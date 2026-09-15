"""The Note record and the publishable rule.

Every test in the publishable section corresponds to something a REAL run in
this project did. The rule has three conditions, and each one was written
because a run violated it:

  * run 1 produced a note whose numbers were traceable and whose conclusion
    was wrong — provenance is a floor, and the tests here pin that it is only
    a floor;
  * a run cut short by a guard returns an answer that reads exactly like a
    finished one;
  * a run that never calls analyze_results reaches a conclusion from a raw
    Sharpe, which the runner already warns about and which must not be
    publishable.

The forgery tests matter more than the happy path: `publishable` is a computed
property precisely so that nothing — a caller, a JSON payload, a hand-edited
database row — can assert it.
"""
from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from falsify.notes.schema import (
    BacktestRecord,
    Note,
    NoteError,
    ProvenanceVerdict,
    RunMetadata,
    from_run,
)

# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def ok_provenance(checked: int = 21) -> ProvenanceVerdict:
    return ProvenanceVerdict(ok=True, checked=checked, verified=checked, exempt=4)


def bad_provenance() -> ProvenanceVerdict:
    return ProvenanceVerdict(
        ok=False,
        checked=21,
        verified=20,
        exempt=4,
        unverified=("'10.16' in: ...a gap of 10.16pp between...",),
    )


def good_run(**over) -> RunMetadata:
    base = dict(
        model="claude-sonnet-5",
        stop_reason="end_turn",
        turns=5,
        tool_sequence=("fetch_data", "compute_feature", "run_backtest", "analyze_results"),
        n_backtests=1,
        input_tokens=12_000,
        output_tokens=900,
        cost_usd=0.036,
    )
    base.update(over)
    return RunMetadata(**base)


def a_backtest(analysed: bool = True, **over) -> BacktestRecord:
    base = dict(
        handle="backtest_1",
        feature="mom_12_1",
        n_buckets=10,
        long_short=True,
        metrics={
            "total_return": 0.1002,
            "cagr": 0.1064,
            "ann_vol": 0.3928,
            "sharpe": 0.45,
            "max_drawdown": -0.3344,
            "n_invested_days": 239,
            "feature": "mom_12_1",
            "n_buckets": 10,
            "long_short": True,
        },
        statistics=(
            {
                "prob_sharpe_above_zero": 0.7563,
                "prob_beats_best_of_n_trials": 0.4218,
                "min_track_record_length_days": 1654.6,
                "n_observations": 239,
                "n_trials_used": 1,
                "trial_variance_assumption": 0.0009,
            }
            if analysed
            else {}
        ),
    )
    base.update(over)
    return BacktestRecord(**base)


def a_note(**over) -> Note:
    base = dict(
        hypothesis="Do stocks that went up over the last year keep going up?",
        prose="The long/short spread returned 10.02% over 239 invested days.",
        backtests=(a_backtest(),),
        provenance=ok_provenance(),
        run=good_run(),
        assumptions={"trial_variance": 0.0009, "n_trials_for_deflation": 1},
    )
    base.update(over)
    return Note(**base)


# --------------------------------------------------------------------------
# ProvenanceVerdict: the verdict cannot lie about itself
# --------------------------------------------------------------------------

def test_ok_verdict_with_unverified_literals_is_refused():
    with pytest.raises(NoteError, match="claims ok"):
        ProvenanceVerdict(ok=True, checked=3, verified=2, unverified=("'99'",))


def test_failed_verdict_with_nothing_unverified_is_refused():
    with pytest.raises(NoteError, match="claims failure"):
        ProvenanceVerdict(ok=False, checked=3, verified=3)


def test_verified_cannot_exceed_checked():
    with pytest.raises(NoteError, match="exceeds checked"):
        ProvenanceVerdict(ok=True, checked=2, verified=5)


def test_coverage_of_a_note_with_no_numerals_is_one_not_a_zero_division():
    assert ProvenanceVerdict(ok=True, checked=0, verified=0).coverage == 1.0


# --------------------------------------------------------------------------
# the publishable rule, one test per way a real run has broken it
# --------------------------------------------------------------------------

def test_a_clean_run_is_publishable():
    note = a_note()
    assert note.publishable
    assert note.unpublishable_reasons == ()


def test_failed_provenance_blocks_publication():
    note = a_note(provenance=bad_provenance())
    assert not note.publishable
    assert any("provenance failed" in r for r in note.unpublishable_reasons)


def test_a_run_cut_short_by_a_guard_blocks_publication():
    """max_turns, token_budget and repeated_error all produce partial answers
    that read exactly like finished ones."""
    for reason in ("max_turns", "token_budget", "repeated_error"):
        note = a_note(run=good_run(stop_reason=reason))
        assert not note.publishable, reason
        assert any(reason in r for r in note.unpublishable_reasons)


def test_a_run_that_never_analysed_blocks_publication():
    note = a_note(
        run=good_run(tool_sequence=("fetch_data", "compute_feature", "run_backtest"))
    )
    assert not note.publishable
    assert any("analyze_results was never called" in r for r in note.unpublishable_reasons)


def test_analyze_results_that_only_ERRORED_does_not_count_as_called():
    """from_run records successful calls only. A tool the model attempted and
    which failed produced no statistics, so the conclusion still rests on a raw
    Sharpe."""
    note = a_note(
        run=good_run(tool_sequence=("fetch_data", "run_backtest"), n_tool_errors=2)
    )
    assert not note.publishable


def test_backtests_present_but_none_analysed_blocks_publication():
    note = a_note(
        backtests=(a_backtest(analysed=False, analysis_error="series too short"),)
    )
    assert not note.publishable
    assert any("no deflated figure" in r for r in note.unpublishable_reasons)


def test_all_three_failures_are_reported_together_not_just_the_first():
    note = a_note(
        provenance=bad_provenance(),
        run=good_run(stop_reason="max_turns", tool_sequence=("fetch_data",)),
    )
    assert len(note.unpublishable_reasons) >= 3


# --------------------------------------------------------------------------
# publishable cannot be forged
# --------------------------------------------------------------------------

def test_publishable_is_not_a_constructor_argument():
    with pytest.raises(TypeError):
        Note(
            hypothesis="h",
            prose="p",
            backtests=(),
            provenance=bad_provenance(),
            run=good_run(),
            publishable=True,  # type: ignore[call-arg]
        )


def test_a_payload_claiming_publishable_is_ignored_on_reload():
    """The dict carries `publishable` for readers. from_dict recomputes it, so
    a hand-edited row does not become publishable."""
    failed = a_note(provenance=bad_provenance())
    payload = failed.to_dict()
    payload["publishable"] = True
    payload["unpublishable_reasons"] = []
    assert Note.from_dict(payload).publishable is False


def test_the_note_is_frozen_so_a_verdict_cannot_be_edited_after_the_fact():
    note = a_note()
    with pytest.raises(dataclasses.FrozenInstanceError):
        note.prose = "something else"  # type: ignore[misc]


# --------------------------------------------------------------------------
# the record refuses incoherent input
# --------------------------------------------------------------------------

def test_an_empty_hypothesis_is_refused():
    with pytest.raises(NoteError, match="verbatim"):
        a_note(hypothesis="   ")


def test_a_naive_timestamp_is_refused():
    with pytest.raises(NoteError, match="timezone-aware"):
        a_note(created_at=dt.datetime(2026, 9, 15, 12, 0))


def test_a_backtest_cannot_hold_statistics_and_an_analysis_error_at_once():
    with pytest.raises(NoteError, match="one of them is wrong"):
        a_backtest(analysis_error="series too short")


def test_a_run_with_no_stop_reason_is_refused():
    with pytest.raises(NoteError, match="stop_reason"):
        good_run(stop_reason="")


def test_variant_names_the_side_because_long_only_is_not_a_test_of_the_hypothesis():
    assert "long/short" in a_backtest().variant
    assert "long-only" in a_backtest(long_short=False).variant


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------

def test_to_dict_from_dict_round_trips_exactly():
    note = a_note()
    assert Note.from_dict(note.to_dict()) == note


def test_round_trip_preserves_a_failed_note_as_failed():
    """Deleting failures is how an eval suite becomes a highlight reel."""
    failed = a_note(provenance=bad_provenance(), run=good_run(stop_reason="max_turns"))
    back = Note.from_dict(failed.to_dict())
    assert back == failed
    assert not back.publishable
    assert back.provenance.unverified == failed.provenance.unverified


def test_to_dict_is_json_safe():
    import json

    json.dumps(a_note().to_dict())


# --------------------------------------------------------------------------
# from_run, against the real loop objects and a fake session
# --------------------------------------------------------------------------

class _FakeReport:
    ok = True
    checked = 21
    verified = 21
    exempt = 4
    unverified: list = []


def _real_session_with_backtests(n: int):
    """A real Session holding synthetic backtest payloads.

    Real, not a mock: `n_backtests` is the honest trial count and it is the
    thing from_run must read off the store rather than off the model.
    """
    import polars as pl

    from falsify.agent.session import Session

    rng_returns = [0.004, -0.002, 0.006, -0.001, 0.003] * 60
    session = Session()
    for i in range(n):
        invested = pl.DataFrame(
            {
                "ts": [dt.date(2025, 1, 1) + dt.timedelta(days=d) for d in range(300)],
                "ret": rng_returns[:300],
            }
        )
        session.put(
            "backtest",
            {"result": None, "invested": invested},
            {
                "feature": "mom_12_1",
                "n_buckets": 10 + i,
                "long_short": i == 0,
                "sharpe": 0.45,
                "total_return": 0.1002,
                "cagr": 0.1064,
                "ann_vol": 0.3928,
                "max_drawdown": -0.3344,
                "n_invested_days": 300,
            },
        )
    return session


def test_from_run_deflates_every_backtest_at_the_FINAL_trial_count():
    """The bug this prevents is real and happened on run 1: the agent analysed
    backtest_1 when one backtest existed, then ran two more, leaving the first
    row deflated for N=1 when the honest count was 3."""
    from falsify.agent.loop import RunResult

    session = _real_session_with_backtests(3)
    result = RunResult(
        answer="Insufficient evidence.",
        stop_reason="end_turn",
        turns=5,
        tool_calls=[{"name": "analyze_results", "input": {}, "ok": True}],
        session=session,
    )
    note = from_run(result, "does momentum work?", _FakeReport())

    assert len(note.backtests) == 3
    assert all(b.statistics["n_trials_used"] == 3 for b in note.backtests)
    assert note.assumptions["n_trials_for_deflation"] == 3


def test_from_run_takes_the_trial_count_from_the_session_not_the_model():
    from falsify.agent.loop import RunResult

    session = _real_session_with_backtests(2)
    result = RunResult(
        answer="I ran one backtest.",  # the model's self-report, ignored
        stop_reason="end_turn",
        turns=3,
        tool_calls=[{"name": "analyze_results", "input": {"n_trials": 1}, "ok": True}],
        session=session,
    )
    assert from_run(result, "h", _FakeReport()).run.n_backtests == 2


def test_from_run_records_only_successful_tool_calls_in_the_sequence():
    from falsify.agent.loop import RunResult

    result = RunResult(
        answer="a",
        stop_reason="end_turn",
        turns=2,
        tool_calls=[
            {"name": "fetch_data", "input": {}, "ok": True},
            {"name": "analyze_results", "input": {}, "ok": False, "error": "bad handle"},
        ],
        session=_real_session_with_backtests(1),
    )
    note = from_run(result, "h", _FakeReport())
    assert note.run.tool_sequence == ("fetch_data",)
    assert note.run.n_tool_errors == 1
    assert not note.publishable


def test_from_run_stores_the_hypothesis_verbatim():
    from falsify.agent.loop import RunResult

    asked = "Do stocks that went up over the last year keep going up?"
    result = RunResult(
        answer="a", stop_reason="end_turn", turns=1, session=_real_session_with_backtests(1)
    )
    assert from_run(result, asked, _FakeReport()).hypothesis == asked


def test_from_run_carries_the_cost_so_a_quoted_result_can_say_what_it_spent():
    from falsify.agent.loop import RunResult

    result = RunResult(
        answer="a",
        stop_reason="end_turn",
        turns=4,
        input_tokens=15_000,
        output_tokens=1_200,
        session=_real_session_with_backtests(1),
    )
    note = from_run(result, "h", _FakeReport())
    assert note.run.cost_usd == pytest.approx(result.cost_usd, abs=1e-6)
    assert note.run.cost_usd > 0


def test_from_run_records_the_assumptions_that_move_the_numbers():
    from falsify.agent.loop import RunResult
    from falsify.agent.tools import TRIAL_VARIANCE

    result = RunResult(
        answer="a", stop_reason="end_turn", turns=1, session=_real_session_with_backtests(1)
    )
    a = from_run(result, "h", _FakeReport()).assumptions
    assert a["trial_variance"] == TRIAL_VARIANCE
    assert a["trial_variance_is_assumed"] is True
    # Both disclosed elsewhere in the repo and both able to mislead a reader of
    # a stored note who does not have the README in front of them.
    assert a["returns_are_price_returns_not_total_returns"] is True
    assert a["quality_gate_is_point_in_time"] is False
    # The agent has no universe argument, so every run it does is on current
    # constituents and is survivorship-inflated. A stored note that does not
    # say so reads as a point-in-time result when it is quoted later, and this
    # project's headline finding is exactly how big that gap is.
    assert a["universe_is_point_in_time"] is False
    assert a["survivorship_bias_present"] is True


def test_from_run_without_a_session_fails_loudly_rather_than_storing_an_empty_table():
    from falsify.agent.loop import RunResult

    result = RunResult(answer="a", stop_reason="end_turn", turns=1, session=None)
    with pytest.raises(NoteError, match="Session"):
        from_run(result, "h", _FakeReport())


def test_from_run_keeps_an_unpublishable_run_rather_than_raising():
    from falsify.agent.loop import RunResult

    result = RunResult(
        answer="partial",
        stop_reason="max_turns",
        turns=12,
        session=_real_session_with_backtests(1),
    )
    note = from_run(result, "h", _FakeReport())
    assert not note.publishable
    assert note.prose == "partial"
