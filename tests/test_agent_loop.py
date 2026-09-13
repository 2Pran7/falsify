"""Tests for the agent loop and its guards. No network, no API key, no spend.

Everything here runs against `FakeClient`, which replays scripted responses.
That is guard 5 from the module docstring, and it is load-bearing: a test suite
that costs money to run is a test suite that stops being run.

The tests that matter are the ones proving a run can be STOPPED by something
other than the model deciding it is finished:

    test_turn_cap_fires
    test_token_budget_fires
    test_repeated_error_ends_the_run
    test_a_stopped_run_is_never_reported_as_completed
"""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import polars as pl
import pytest

from falsify.agent import loop as L
from falsify.agent import tools as T
from falsify.agent.session import Session


# --- the fake client -------------------------------------------------------


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, arguments: dict, block_id: str = "tu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=arguments)


def response(
    content: list,
    stop_reason: str,
    input_tokens: int = 1000,
    output_tokens: int = 100,
    cache_read: int = 0,
    cache_write: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


class FakeClient:
    """Replays a script. Records what it was asked, so the loop can be checked.

    If the script runs out, it keeps returning the last response. That is
    deliberate: a loop with a broken guard would otherwise fail with
    StopIteration, which looks like a test bug rather than a runaway loop.
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.script) > 1:
            return self.script.pop(0)
        return self.script[0]


def _panel(n_tickers: int = 12, n_days: int = 400) -> pl.DataFrame:
    days = pl.date_range(dt.date(2024, 1, 1), dt.date(2026, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)[:n_days]
    rows = []
    for i in range(n_tickers):
        drift = 1.0 + 0.0004 * (i - n_tickers / 2)
        for j, d in enumerate(days):
            rows.append((f"T{i:02d}", d, 100.0 * drift**j))
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}, orient="row"
    )


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(T, "load_panel", lambda *a, **k: _panel())


# --- the happy path --------------------------------------------------------


def test_a_plain_answer_ends_the_run_in_one_turn():
    client = FakeClient([response([text_block("No data needed.")], "end_turn")])
    r = L.run("is momentum real?", client)
    assert r.stop_reason == "end_turn"
    assert r.turns == 1
    assert r.answer == "No data needed."


def test_a_tool_call_is_dispatched_and_the_loop_continues(patched):
    client = FakeClient(
        [
            response([tool_block("fetch_data", {})], "tool_use"),
            response([text_block("Loaded 4800 rows.")], "end_turn"),
        ]
    )
    r = L.run("how much data?", client)
    assert r.turns == 2
    assert r.completed
    assert [c["name"] for c in r.tool_calls] == ["fetch_data"]
    assert r.tool_calls[0]["ok"]


def test_the_full_four_tool_sequence(patched):
    """The shape of a correct run, start to finish."""
    client = FakeClient(
        [
            response([tool_block("fetch_data", {}, "a")], "tool_use"),
            response(
                [tool_block("compute_feature", {"panel_handle": "panel_1",
                                                "feature": "mom_12_1"}, "b")],
                "tool_use",
            ),
            response(
                [tool_block("run_backtest", {"feature_handle": "feature_1"}, "c")],
                "tool_use",
            ),
            response(
                [tool_block("analyze_results", {"backtest_handle": "backtest_1"}, "d")],
                "tool_use",
            ),
            response([text_block("Insufficient evidence.")], "end_turn"),
        ]
    )
    r = L.run("does momentum work?", client)
    assert r.completed
    assert [c["name"] for c in r.tool_calls] == [
        "fetch_data", "compute_feature", "run_backtest", "analyze_results"
    ]
    assert all(c["ok"] for c in r.tool_calls)
    assert r.session.n_backtests == 1


def test_tool_output_is_handed_back_as_a_tool_result(patched):
    client = FakeClient(
        [
            response([tool_block("fetch_data", {}, "xyz")], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )
    L.run("q", client)
    sent = client.calls[1]["messages"]
    results = sent[-1]["content"]
    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "xyz"
    assert "panel_1" in results[0]["content"]


def test_two_tool_calls_in_one_response_are_both_dispatched(patched):
    client = FakeClient(
        [
            response(
                [tool_block("fetch_data", {}, "a"), tool_block("fetch_data", {}, "b")],
                "tool_use",
            ),
            response([text_block("done")], "end_turn"),
        ]
    )
    r = L.run("q", client)
    assert len(r.tool_calls) == 2
    assert r.session.handles("panel") == ["panel_1", "panel_2"]


# --- errors are data, not crashes ------------------------------------------


def test_a_tool_error_is_returned_to_the_model_not_raised(patched):
    """The model's chance to fix itself is the next turn, so it must get there."""
    client = FakeClient(
        [
            response(
                [tool_block("compute_feature",
                            {"panel_handle": "panel_9", "feature": "ret_1d"}, "a")],
                "tool_use",
            ),
            response([text_block("recovered")], "end_turn"),
        ]
    )
    r = L.run("q", client)
    assert r.completed
    assert r.tool_calls[0]["ok"] is False
    sent = client.calls[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True
    assert "panel_9" in sent["content"]


def test_an_invalid_tool_name_does_not_kill_the_run():
    client = FakeClient(
        [
            response([tool_block("drop_tables", {}, "a")], "tool_use"),
            response([text_block("ok, my mistake")], "end_turn"),
        ]
    )
    r = L.run("q", client)
    assert r.completed
    assert "unknown tool" in r.tool_calls[0]["error"]


def test_recovery_after_an_error_clears_the_streak(patched):
    """One error, then progress, then the SAME error again is not a stuck model.

    The identical message both times is what makes this discriminating. With
    different messages the deduplication check passes on its own and the test
    proves nothing about the streak being cleared. Here, only an actual reset
    on success keeps the run alive: a model that hits a wall, works around it,
    and hits the same wall later is still making progress, and ending its run
    would throw away the work already paid for.

    The error used is an unknown TOOL name, whose message is a constant. A bad
    handle will not do: those messages name the handles currently available, so
    the successful call in the middle changes the text and the two errors stop
    being identical for reasons that have nothing to do with the streak.
    """
    client = FakeClient(
        [
            response([tool_block("drop_tables", {}, "a")], "tool_use"),
            response([tool_block("fetch_data", {}, "b")], "tool_use"),
            response([tool_block("drop_tables", {}, "c")], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )
    r = L.run("q", client, config=L.RunConfig(max_repeated_errors=2))
    assert r.completed, "progress between two identical errors is not a loop"


# --- the guards ------------------------------------------------------------


def test_turn_cap_fires():
    """The model never says it is finished. Something else has to stop it."""
    client = FakeClient([response([tool_block("drop_tables", {}, "a")], "tool_use")])
    r = L.run("q", client, config=L.RunConfig(max_turns=3, max_repeated_errors=99))
    assert r.stop_reason == "max_turns"
    assert r.turns == 3


def test_turn_cap_counts_calls_actually_paid_for():
    client = FakeClient([response([tool_block("drop_tables", {}, "a")], "tool_use")])
    r = L.run("q", client, config=L.RunConfig(max_turns=5, max_repeated_errors=99))
    assert len(client.calls) == 5


def test_token_budget_fires():
    """Accumulated from usage, which is the ground truth for spend."""
    client = FakeClient(
        [response([tool_block("drop_tables", {}, "a")], "tool_use",
                  input_tokens=40_000, output_tokens=1_000)]
    )
    r = L.run("q", client, config=L.RunConfig(max_total_tokens=100_000,
                                              max_repeated_errors=99))
    assert r.stop_reason == "token_budget"
    assert r.total_tokens >= 100_000
    assert r.turns == 3


def test_repeated_error_ends_the_run():
    """A model that cannot recover in two attempts will not recover in twenty."""
    client = FakeClient([response([tool_block("drop_tables", {}, "a")], "tool_use")])
    r = L.run("q", client, config=L.RunConfig(max_turns=50, max_repeated_errors=2))
    assert r.stop_reason == "repeated_error"
    assert r.turns == 2


def test_different_errors_are_not_a_repeat(patched):
    """Two different failures is exploration; the same failure twice is a loop."""
    client = FakeClient(
        [
            response([tool_block("drop_tables", {}, "a")], "tool_use"),
            response([tool_block("compute_feature",
                                 {"panel_handle": "x", "feature": "ret_1d"}, "b")],
                     "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )
    r = L.run("q", client, config=L.RunConfig(max_repeated_errors=2))
    assert r.completed


def test_a_stopped_run_is_never_reported_as_completed():
    """`completed` is the field a caller trusts. It must mean exactly one thing.

    A truncated run whose partial answer looked finished would put an
    unsupported conclusion into a research note with nothing marking it.
    """
    client = FakeClient([response([tool_block("drop_tables", {}, "a")], "tool_use")])
    for cfg in (
        L.RunConfig(max_turns=2, max_repeated_errors=99),
        L.RunConfig(max_turns=99, max_repeated_errors=2),
        L.RunConfig(max_total_tokens=1, max_repeated_errors=99),
    ):
        assert not L.run("q", client, config=cfg).completed


# --- cost accounting -------------------------------------------------------


def test_cost_is_computed_from_reported_usage():
    """Hand-derived: 1M input at $2 plus 1M output at $10 is $12."""
    client = FakeClient(
        [response([text_block("hi")], "end_turn",
                  input_tokens=1_000_000, output_tokens=1_000_000)]
    )
    r = L.run("q", client)
    assert r.cost_usd == pytest.approx(12.0)


def test_cached_reads_are_cheaper_than_fresh_input():
    """Guard 4 only pays off if the accounting reflects it: reads are 0.1x."""
    fresh = L.RunResult(answer="", stop_reason="end_turn", turns=1, input_tokens=1_000_000)
    cached = L.RunResult(answer="", stop_reason="end_turn", turns=1,
                         cache_read_tokens=1_000_000)
    assert cached.cost_usd < fresh.cost_usd
    assert cached.cost_usd == pytest.approx(fresh.cost_usd * 0.1)


def test_usage_accumulates_across_turns(patched):
    client = FakeClient(
        [
            response([tool_block("fetch_data", {}, "a")], "tool_use",
                     input_tokens=500, output_tokens=50),
            response([text_block("done")], "end_turn",
                     input_tokens=700, output_tokens=80),
        ]
    )
    r = L.run("q", client)
    assert r.input_tokens == 1200
    assert r.output_tokens == 130


def test_summary_reports_the_stop_reason_and_the_cost():
    client = FakeClient([response([text_block("hi")], "end_turn")])
    s = L.run("q", client).summary()
    assert "end_turn" in s and "$" in s


# --- what the model is actually sent ---------------------------------------


def test_the_four_tools_are_sent():
    client = FakeClient([response([text_block("hi")], "end_turn")])
    L.run("q", client)
    names = {t["name"] for t in client.calls[0]["tools"]}
    assert names == {"fetch_data", "compute_feature", "run_backtest", "analyze_results"}


def test_the_system_prompt_and_tools_are_cached():
    """Identical on every turn and most of the input tokens. Guard 4."""
    client = FakeClient([response([text_block("hi")], "end_turn")])
    L.run("q", client)
    call = client.calls[0]
    assert call["system"][0]["cache_control"]["type"] == "ephemeral"
    assert call["tools"][-1]["cache_control"]["type"] == "ephemeral"


def test_the_system_prompt_forbids_computing():
    prompt = L.SYSTEM_PROMPT.lower()
    assert "cannot compute" in prompt
    assert "tool" in prompt


def test_the_system_prompt_requires_analysis_before_a_conclusion():
    """Without this the model reports a raw Sharpe and calls it a finding."""
    assert "analyze_results" in L.SYSTEM_PROMPT


def test_the_system_prompt_licenses_a_negative_result():
    """falsify exists to kill hypotheses. The model has to know that is allowed."""
    assert "insufficient" in L.SYSTEM_PROMPT.lower()


def test_the_hypothesis_is_the_first_user_message():
    client = FakeClient([response([text_block("hi")], "end_turn")])
    L.run("do stocks that went up keep going up?", client)
    first = client.calls[0]["messages"][0]
    assert first["role"] == "user"
    assert "stocks that went up" in first["content"]


def test_an_external_session_is_used_and_returned(patched):
    """The caller may need the artifacts afterwards, for the research note."""
    s = Session()
    client = FakeClient(
        [
            response([tool_block("fetch_data", {}, "a")], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )
    r = L.run("q", client, session=s)
    assert r.session is s
    assert s.handles("panel") == ["panel_1"]


def test_the_transcript_is_kept_for_inspection(patched):
    client = FakeClient(
        [
            response([tool_block("fetch_data", {}, "a")], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )
    r = L.run("q", client)
    roles = [m["role"] for m in r.messages]
    assert roles == ["user", "assistant", "user"]


def test_assistant_blocks_are_serialisable(patched):
    """The transcript is the debugging surface; it has to survive being saved."""
    client = FakeClient(
        [
            response([tool_block("fetch_data", {}, "a")], "tool_use"),
            response([text_block("done")], "end_turn"),
        ]
    )
    json.dumps(L.run("q", client).messages, default=str)
