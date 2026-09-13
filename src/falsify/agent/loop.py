"""The agent loop, and the five guards that stop it costing money it should not.

The loop is the small part. In full:

    send the conversation to the model
      -> model returns either a final answer, or a request to call tools
      -> if tools: dispatch them, append the results, go round again
      -> if answer: stop

That is the whole of "agentic". The model never executes anything; it asks, and
this module decides whether to comply. Everything interesting is in the
deciding, which lives in `agent.tools`, and in the stopping, which lives here.

WHY THE GUARDS EXIST. One failure mode dominates: the model calls a tool, the
tool errors, the model tries a near-identical call, it errors again, and the
pair loop until something external stops them. Unattended, that is how a budget
disappears overnight. Every guard below is a way for the run to end that does
not depend on the model deciding it is finished:

  1. max_turns          a hard ceiling on model calls, checked before each one.
  2. token budget       accumulated from `response.usage`, which is the ground
                        truth for spend. Not an estimate from string length.
  3. repeated errors    the same tool failing the same way twice ends the run.
                        A model that cannot recover in two attempts will not
                        recover in twenty.
  4. prompt caching     the system prompt and tool schemas are identical on
                        every turn and are most of the input tokens. Cached,
                        they cost a tenth as much to re-read.
  5. fake client        the tests drive this loop with scripted responses and
                        never touch the network, so the suite needs no API key
                        and costs nothing. A test suite that costs money stops
                        being run.

Guards 1 to 3 each end the run with a distinct `stop_reason`, so a run that was
cut short can never be mistaken for one that finished.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from falsify.agent.session import Session
from falsify.agent.tools import ToolError, dispatch, tool_schemas

MODEL = "claude-sonnet-5"

# USD per million tokens for MODEL. Cache writes cost 1.25x a normal input
# token and cache reads 0.1x, which is the entire reason guard 4 is worth the
# few lines it takes.
PRICE_IN = 2.0
PRICE_OUT = 10.0
PRICE_CACHE_WRITE = 2.5
PRICE_CACHE_READ = 0.2

SYSTEM_PROMPT = """\
You are a quantitative research agent. You are given a market hypothesis in \
plain English and your job is to test it honestly, including concluding that \
the evidence does not support a conclusion.

You cannot compute anything yourself. Every number must come from a tool call. \
Never estimate, infer or recall a figure: if you need a number, call the tool \
that produces it. Numbers you state in your final answer must appear verbatim \
in a tool result you received.

The normal sequence is fetch_data, then compute_feature, then run_backtest, \
then analyze_results. You may run several backtests to test variants, but \
every one of them counts as a trial and makes the statistical bar higher.

Before you conclude anything, you must call analyze_results. A Sharpe ratio on \
its own is not a finding. What matters is the deflated Sharpe, which accounts \
for how many strategies were tried, and the minimum track record length, which \
says how much data would be needed before the Sharpe could be distinguished \
from zero at all.

Report honestly. "The evidence is insufficient" is a legitimate and frequent \
answer, and a short sample with a high Sharpe is usually exactly that. Do not \
dress up a weak result. State what was measured, what it means, and what would \
be needed to make it convincing.
"""


@dataclass(frozen=True)
class RunConfig:
    """The guards, as numbers.

    max_turns defaults low. A correct run is roughly four tool calls plus a
    final answer; anything beyond a dozen turns is the model going in circles,
    and paying for twelve turns to discover that is enough.
    """

    model: str = MODEL
    max_turns: int = 12
    max_total_tokens: int = 200_000
    max_output_tokens: int = 2_048
    max_repeated_errors: int = 2


@dataclass
class RunResult:
    """Everything the run produced, including what it cost.

    `stop_reason` is the field to read first. "end_turn" is the only value that
    means the model finished on its own; the rest mean a guard fired, and the
    answer, if any, is partial.
    """

    answer: str
    stop_reason: str
    turns: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    session: Session | None = None

    @property
    def cost_usd(self) -> float:
        """Spend for this run, from reported usage rather than an estimate."""
        return (
            self.input_tokens / 1e6 * PRICE_IN
            + self.output_tokens / 1e6 * PRICE_OUT
            + self.cache_write_tokens / 1e6 * PRICE_CACHE_WRITE
            + self.cache_read_tokens / 1e6 * PRICE_CACHE_READ
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_write_tokens
            + self.cache_read_tokens
        )

    @property
    def completed(self) -> bool:
        """True only if the model stopped because it was done."""
        return self.stop_reason == "end_turn"

    def summary(self) -> str:
        return (
            f"{self.stop_reason} after {self.turns} turns, "
            f"{len(self.tool_calls)} tool calls, "
            f"{self.total_tokens:,} tokens, ${self.cost_usd:.4f}"
        )


def _cached_system() -> list[dict[str, Any]]:
    """System prompt as a cacheable block (guard 4)."""
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _cached_tools() -> list[dict[str, Any]]:
    """Tool schemas with a cache breakpoint on the last one (guard 4).

    Marking the final tool caches everything above it, which is the whole
    schema block: identical on every turn and a large share of the input.
    """
    schemas = [dict(t) for t in tool_schemas()]
    schemas[-1]["cache_control"] = {"type": "ephemeral"}
    return schemas


def _text_of(content: list[Any]) -> str:
    return "".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def _tool_uses(content: list[Any]) -> list[Any]:
    return [b for b in content if getattr(b, "type", None) == "tool_use"]


def _blocks_to_dicts(content: list[Any]) -> list[dict[str, Any]]:
    """Assistant content back into plain dicts for the next request.

    The SDK accepts its own block objects here, but the tests drive this loop
    with fakes, and a transcript of plain dicts is also what makes a run
    inspectable after the fact.
    """
    out: list[dict[str, Any]] = []
    for b in content:
        if getattr(b, "type", None) == "text":
            out.append({"type": "text", "text": b.text})
        elif getattr(b, "type", None) == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def run(
    hypothesis: str,
    client: Any,
    session: Session | None = None,
    config: RunConfig | None = None,
) -> RunResult:
    """Test one hypothesis end to end.

    Args:
        hypothesis: the question, in plain English.
        client: anything exposing `messages.create(...)`. The real
            `anthropic.Anthropic()` in production, a scripted fake in tests.
        session: handle store; a fresh one is made if omitted.
        config: the guards.

    Returns:
        RunResult. Check `stop_reason` before believing `answer`: only
        "end_turn" means the model finished rather than being cut off.

    The client is injected rather than constructed here for one reason: it is
    what lets the entire loop, including every guard, be tested without a
    network call or an API key.
    """
    session = session if session is not None else Session()
    config = config or RunConfig()

    messages: list[dict[str, Any]] = [{"role": "user", "content": hypothesis}]
    result = RunResult(answer="", stop_reason="", turns=0, session=session)
    recent_errors: list[str] = []

    while True:
        # GUARD 1: turn cap, checked BEFORE the call so the ceiling is the
        # number of calls actually paid for.
        if result.turns >= config.max_turns:
            result.stop_reason = "max_turns"
            break

        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_output_tokens,
            system=_cached_system(),
            tools=_cached_tools(),
            messages=messages,
        )
        result.turns += 1

        usage = response.usage
        result.input_tokens += getattr(usage, "input_tokens", 0) or 0
        result.output_tokens += getattr(usage, "output_tokens", 0) or 0
        result.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        result.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

        # The model is done.
        if response.stop_reason != "tool_use":
            result.answer = _text_of(response.content)
            result.stop_reason = "end_turn"
            break

        messages.append({"role": "assistant", "content": _blocks_to_dicts(response.content)})

        tool_results: list[dict[str, Any]] = []
        for call in _tool_uses(response.content):
            try:
                output = dispatch(call.name, call.input, session)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(output, default=str),
                    }
                )
                result.tool_calls.append(
                    {"name": call.name, "input": call.input, "ok": True}
                )
                recent_errors.clear()  # progress resets the error streak
            except ToolError as exc:
                # GUARD 3 material: errors go BACK to the model as data, and
                # the next turn is its chance to correct itself.
                message = str(exc)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": message,
                        "is_error": True,
                    }
                )
                result.tool_calls.append(
                    {"name": call.name, "input": call.input, "ok": False, "error": message}
                )
                recent_errors.append(f"{call.name}: {message}")

        messages.append({"role": "user", "content": tool_results})

        # GUARD 3: the same failure twice means the model is stuck.
        if len(recent_errors) >= config.max_repeated_errors and (
            len(set(recent_errors[-config.max_repeated_errors:])) == 1
        ):
            result.answer = _text_of(response.content)
            result.stop_reason = "repeated_error"
            break

        # GUARD 2: spend, from reported usage.
        if result.total_tokens >= config.max_total_tokens:
            result.answer = _text_of(response.content)
            result.stop_reason = "token_budget"
            break

    result.messages = messages
    return result
