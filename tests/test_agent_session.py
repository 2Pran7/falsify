"""Tests for the handle store.

The store is small, so most of these are mechanical. Three are not, and they are
the ones that matter:

    test_summary_cap_rejects_a_payload_sized_summary
    test_unknown_handle_error_lists_what_is_available
    test_n_backtests_counts_every_run_not_the_reported_one

Each defends one of the reasons the store exists at all, rather than the code
it happens to contain.
"""
from __future__ import annotations

import pytest

from falsify.agent.session import (
    MAX_SUMMARY_BYTES,
    Session,
    SummaryTooLarge,
    UnknownHandle,
    WrongKind,
)


@pytest.fixture
def s() -> Session:
    return Session()


# --- storing and resolving -------------------------------------------------


def test_put_returns_a_readable_handle(s):
    h = s.put("panel", payload=[1, 2, 3], summary={"n_rows": 3})
    assert h == "panel_1"


def test_handles_number_upward_within_a_kind(s):
    assert s.put("panel", [], {}) == "panel_1"
    assert s.put("panel", [], {}) == "panel_2"


def test_kinds_number_independently(s):
    s.put("panel", [], {})
    assert s.put("backtest", [], {}) == "backtest_1"


def test_payload_round_trips_by_identity(s):
    obj = object()
    h = s.put("panel", obj, {})
    assert s.payload(h) is obj


def test_summary_round_trips(s):
    h = s.put("panel", [], {"n_rows": 502, "tickers": 548})
    assert s.summary(h) == {"n_rows": 502, "tickers": 548}


def test_unknown_kind_is_rejected(s):
    with pytest.raises(ValueError, match="unknown kind"):
        s.put("portfolio", [], {})


# --- the summary cap: the guard that keeps the boundary real ---------------


def test_summary_cap_rejects_a_payload_sized_summary(s):
    """The easy way to write a tool is to dump the frame into the summary.

    If that succeeded, handles would still be returned and would still look
    like indirection, while the model received the whole payload anyway. The
    cap turns the shortcut into an immediate, loud failure.
    """
    fat = {"rows": [{"ticker": "AAPL", "close": 123.45} for _ in range(500)]}
    with pytest.raises(SummaryTooLarge, match="over the"):
        s.put("panel", [], fat)


def test_summary_just_under_the_cap_is_allowed(s):
    body = "x" * (MAX_SUMMARY_BYTES - 20)
    assert s.put("panel", [], {"note": body})


def test_non_json_safe_summary_is_rejected_by_value_not_type(s):
    """default=str means most objects serialise; a circular one still cannot."""
    circular: dict = {}
    circular["self"] = circular
    with pytest.raises(SummaryTooLarge):
        s.put("panel", [], circular)


def test_a_rejected_put_does_not_consume_a_handle_number(s):
    """A failed store must not leave a gap in the numbering.

    Gaps matter because the transcript is the debugging surface: `panel_1`
    followed by `panel_3` sends you looking for a panel that never existed.
    """
    s.put("panel", [], {"ok": True})
    with pytest.raises(SummaryTooLarge):
        s.put("panel", [], {"big": "y" * (MAX_SUMMARY_BYTES + 1)})
    assert s.put("panel", [], {"ok": True}) == "panel_2"


# --- errors the model has to recover from ----------------------------------


def test_unknown_handle_raises(s):
    with pytest.raises(UnknownHandle):
        s.get("panel_9")


def test_unknown_handle_error_lists_what_is_available(s):
    """The message goes back to the model as a tool error.

    A model shown the valid handles corrects itself on the next turn. A model
    shown only "not found" guesses, and guessing costs turns, which costs
    money and burns the turn cap.
    """
    s.put("panel", [], {})
    s.put("backtest", [], {})
    with pytest.raises(UnknownHandle, match="panel_1, backtest_1"):
        s.get("panel_7")


def test_unknown_handle_on_an_empty_store_says_so(s):
    with pytest.raises(UnknownHandle, match="none yet"):
        s.get("panel_1")


def test_wrong_kind_raises(s):
    h = s.put("panel", [], {})
    with pytest.raises(WrongKind):
        s.get(h, kind="backtest")


def test_wrong_kind_error_names_both_kinds_and_the_alternatives(s):
    s.put("panel", [], {})
    h = s.put("backtest", [], {})
    with pytest.raises(WrongKind, match="is a backtest, not a panel"):
        s.get(h, kind="panel")


def test_right_kind_passes_through(s):
    h = s.put("feature", [], {})
    assert s.get(h, kind="feature").handle == h


# --- the trial counter -----------------------------------------------------


def test_n_backtests_starts_at_zero(s):
    assert s.n_backtests == 0


def test_n_backtests_counts_every_run_not_the_reported_one(s):
    """The deflation trial count comes from the store, never from the model.

    The honest N includes every variant tried and abandoned. That is precisely
    the number a model summarising its own work has an incentive to forget, and
    understating N is how a deflated Sharpe gets quietly re-inflated. So the
    count is taken from the side of the boundary that cannot be talked out of
    it.
    """
    for _ in range(9):
        s.put("backtest", [], {})
    assert s.n_backtests == 9


def test_n_backtests_ignores_other_kinds(s):
    s.put("panel", [], {})
    s.put("feature", [], {})
    s.put("analysis", [], {})
    assert s.n_backtests == 0


# --- listing and introspection ---------------------------------------------


def test_handles_are_in_creation_order(s):
    s.put("panel", [], {})
    s.put("backtest", [], {})
    s.put("panel", [], {})
    assert s.handles() == ["panel_1", "backtest_1", "panel_2"]


def test_handles_filter_by_kind(s):
    s.put("panel", [], {})
    s.put("backtest", [], {})
    s.put("panel", [], {})
    assert s.handles("panel") == ["panel_1", "panel_2"]


def test_len_counts_artifacts(s):
    s.put("panel", [], {})
    s.put("backtest", [], {})
    assert len(s) == 2


def test_iteration_yields_artifacts(s):
    s.put("panel", [1], {"a": 1})
    assert [a.kind for a in s] == ["panel"]


def test_two_sessions_do_not_share_state(s):
    """A run is one conversation. A handle from another run must not resolve.

    The store is deliberately in-memory and per-run: a stale handle silently
    resolving to a previous run's panel is a correctness bug that would be
    invisible in the transcript.
    """
    other = Session()
    h = s.put("panel", [], {})
    with pytest.raises(UnknownHandle):
        other.get(h)


def test_repr_is_useful_when_debugging_a_transcript(s):
    s.put("panel", [], {})
    s.put("backtest", [], {})
    assert "panel=1" in repr(s) and "backtest=1" in repr(s)


def test_repr_of_empty_session(s):
    assert "empty" in repr(s)
