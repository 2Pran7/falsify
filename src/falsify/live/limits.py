"""Whether a live run may start. Pure functions, no I/O, so every rule is tested.

Three caps, checked in this order, each a refusal with a reason the visitor sees:

  1. SPEND. The owner pays for live runs, so the hard limit is in dollars, not
     runs: a run starts only if today's spend plus its worst-case cost stays
     within `daily_usd_cap`, so the cap is never crossed. A run count alone would not bound cost, because a run
     that loops to its token budget costs several times a normal one.
  2. RUNS. `daily_run_cap` across all visitors, so a burst of normal runs is
     stopped before it reaches the dollar cap.
  3. PER VISITOR. `per_visitor_cap` per hashed IP per day, so one person cannot
     use up everyone's allowance.

The per-run token budget (`RUN_TOKEN_BUDGET`) is what makes the dollar cap
sound: the most a single run can cost is bounded, so a run that starts under
the cap cannot overshoot it by more than one run's worst case.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

MAX_HYPOTHESIS_CHARS = 300
MIN_HYPOTHESIS_CHARS = 12
MAX_NAME_CHARS = 60

# Per-run guards for live runs, tighter than the CLI defaults.
RUN_MAX_TURNS = 10
RUN_TOKEN_BUDGET = 120_000
RUN_MAX_OUTPUT_TOKENS = 2_048

# The most one run can cost, reserved against the daily cap the moment it is
# QUEUED and replaced by the real figure when it finishes. Without the
# reservation, five queued runs each read "spent $0 so far" and the cap is only
# enforced after the money has gone. Bound: every output token at the output
# price, the remainder of the budget at the input price.
# Rounded UP to the cent: rounding to nearest put the reserve below the bound.
RUN_RESERVE_USD = math.ceil(
    ((RUN_MAX_TURNS * RUN_MAX_OUTPUT_TOKENS) / 1e6 * 10.0 + RUN_TOKEN_BUDGET / 1e6 * 2.0) * 100
) / 100


class Refused(Exception):
    """A run may not start. `str(e)` is shown to the visitor verbatim."""

    def __init__(self, message: str, status: int = 429):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Limits:
    daily_usd_cap: float = 2.0
    daily_run_cap: int = 25
    per_visitor_cap: int = 3
    max_queue: int = 5


@dataclass(frozen=True)
class Today:
    """What has happened so far today (UTC), as the store reports it."""

    spend_usd: float
    runs: int
    runs_by_visitor: int
    queued: int


def clean_hypothesis(text: str) -> str:
    t = " ".join((text or "").split())
    if len(t) < MIN_HYPOTHESIS_CHARS:
        raise Refused(
            f"Ask a full question (at least {MIN_HYPOTHESIS_CHARS} characters), "
            "e.g. 'Do low-volatility stocks outperform?'",
            status=400,
        )
    if len(t) > MAX_HYPOTHESIS_CHARS:
        raise Refused(f"Keep it under {MAX_HYPOTHESIS_CHARS} characters.", status=400)
    return t


def clean_name(text: str | None) -> str | None:
    t = " ".join((text or "").split())[:MAX_NAME_CHARS]
    return t or None


def check(limits: Limits, today: Today) -> None:
    """Raise Refused if a run may not start. Order matters: spend first."""
    # A run starts only if its WORST case still fits under the cap, so the cap
    # is never crossed, rather than crossed by one run and then enforced.
    if today.spend_usd + RUN_RESERVE_USD > limits.daily_usd_cap:
        raise Refused(
            "Today's budget for live runs is used up. It resets at midnight UTC; "
            "the frozen results on the home page are always available."
        )
    if today.runs >= limits.daily_run_cap:
        raise Refused("Today's live runs are used up. It resets at midnight UTC.")
    if today.runs_by_visitor >= limits.per_visitor_cap:
        raise Refused(
            f"That's your {limits.per_visitor_cap} runs for today. Thanks for trying it."
        )
    if today.queued >= limits.max_queue:
        raise Refused("A few runs are already queued. Try again in a minute.", status=503)


def runs_left(limits: Limits, today: Today) -> int:
    if today.spend_usd + RUN_RESERVE_USD > limits.daily_usd_cap:
        return 0
    return max(0, min(limits.daily_run_cap - today.runs, limits.per_visitor_cap - today.runs_by_visitor))


def visitor_id(ip: str, salt: str) -> str:
    """A stable, non-reversible id for rate limiting. The raw IP is never stored."""
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:12]
