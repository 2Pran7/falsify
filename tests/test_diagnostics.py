"""Diagnostics, the predicted direction, and the verdict card.

Three things are defended here:

  1. The diagnostics are right on inputs where the answer is known by hand.
  2. The predicted direction is built into the strategy, so a hypothesis that
     the BOTTOM wins is tested as long-bottom, and its probabilities are about
     that strategy rather than its mirror image.
  3. The verdict card is computed from stored numbers by a fixed rule, so the
     one line a reader takes away is never the model's.
"""
from __future__ import annotations

import datetime as dt
import json
import math

import polars as pl
import pytest

from falsify.agent import tools as T
from falsify.agent.session import MAX_SUMMARY_BYTES, Session
from falsify.backtest import diagnostics as D
from falsify.notes.schema import BacktestRecord, Note, ProvenanceVerdict, RunMetadata
from tests.test_agent_tools import _panel


def _days(n: int) -> list[dt.date]:
    d = pl.date_range(dt.date(2025, 1, 1), dt.date(2026, 12, 31), "1d", eager=True)
    return list(d.filter(d.dt.weekday() <= 5)[:n])


# --- 1. hand-derived diagnostics --------------------------------------------


def test_bucket_returns_hand_derived():
    """Four names at 0%, 1%, 2%, 3% a day, two buckets.

    Bucket 1 (lowest signal) holds the 0% and 1% names: mean 0.5% a day,
    annualised 0.005 * 252 = 1.26. Bucket 2: 2.5% a day, 6.3.
    """
    days = _days(6)
    rows = []
    for i, t in enumerate("ABCD"):
        for j, d in enumerate(days):
            rows.append((t, d, 100.0 * (1 + i / 100) ** j))
    prices = pl.DataFrame(rows, schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}, orient="row")
    signal = pl.DataFrame({"ts": [days[0]] * 4, "ticker": list("ABCD"), "sig": [1.0, 2.0, 3.0, 4.0]})
    out = D.bucket_returns(prices, signal, pl.Series([days[0]]), pl.Series(days[:5]), 2)
    assert [b["bucket"] for b in out] == [1, 2]
    assert out[0]["ann_return"] == pytest.approx(0.005 * 252)
    assert out[1]["ann_return"] == pytest.approx(0.025 * 252)
    assert all(b["n_days"] == 5 for b in out)


def test_bucket_membership_is_held_until_the_next_rebalance():
    """Signals flip on the second rebalance; bucket 2 must follow the flip."""
    days = _days(6)
    rows = [(t, d, 100.0 * (1 + r) ** j) for t, r in (("A", 0.0), ("B", 0.02)) for j, d in enumerate(days)]
    prices = pl.DataFrame(rows, schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}, orient="row")
    signal = pl.DataFrame({
        "ts": [days[0], days[0], days[3], days[3]],
        "ticker": ["A", "B", "A", "B"],
        "sig": [1.0, 2.0, 2.0, 1.0],
    })
    out = D.bucket_returns(prices, signal, pl.Series([days[0], days[3]]), pl.Series(days[:5]), 2)
    # Bucket 2 is B for three days (2%) and A for two (0%): mean 1.2% a day.
    assert out[1]["ann_return"] == pytest.approx(0.012 * 252)


def test_monotonicity_is_plus_one_minus_one_and_undefined_below_three():
    up = [{"bucket": i, "ann_return": float(i)} for i in range(1, 6)]
    down = [{"bucket": i, "ann_return": -float(i)} for i in range(1, 6)]
    assert D.monotonicity(up) == pytest.approx(1.0)
    assert D.monotonicity(down) == pytest.approx(-1.0)
    assert D.monotonicity(up[:2]) is None


def test_halves_split_by_count_and_cover_every_day():
    days = _days(9)
    inv = pl.DataFrame({"ts": days, "ret": [0.01, -0.01, 0.02, 0.0, 0.01, -0.02, 0.01, 0.03, -0.01]})
    h = D.halves(inv)
    assert [x["n_days"] for x in h] == [4, 5]
    assert h[0]["last_date"] < h[1]["first_date"]
    assert h[0]["sharpe"] == pytest.approx(
        pl.Series([0.01, -0.01, 0.02, 0.0]).mean() / pl.Series([0.01, -0.01, 0.02, 0.0]).std() * math.sqrt(252)
    )


def test_cost_sensitivity_reprices_from_gross_and_turnover():
    days = _days(4)
    inv = pl.DataFrame({"ts": days, "gross_ret": [0.01, 0.0, 0.01, 0.0]})
    turn = pl.DataFrame({"ts": [days[0], days[2]], "turnover": [2.0, 1.0]})
    c = {x["cost_bps"]: x for x in D.cost_sensitivity(inv, turn)}
    assert set(c) == set(D.COST_GRID_BPS)
    # 25 bps on turnover 2.0 is 0.005 off the first day, 0.0025 off the third.
    net = [0.01 - 0.005, 0.0, 0.01 - 0.0025, 0.0]
    assert c[25]["cagr"] == pytest.approx((math.prod(1 + r for r in net)) ** (252 / 4) - 1)
    assert c[0]["sharpe"] > c[10]["sharpe"] > c[25]["sharpe"]


def test_equity_curve_starts_at_one_ends_at_the_compounded_total_and_is_capped():
    days = _days(400)
    r = [0.001 * math.sin(i) for i in range(400)]
    eq = D.equity_curve(pl.DataFrame({"ts": days, "ret": r}))
    assert eq[0]["equity"] == 1.0
    assert eq[-1]["ts"] == str(days[-1])
    assert eq[-1]["equity"] == pytest.approx(math.prod(1 + x for x in r), abs=1e-5)
    assert len(eq) <= D.EQUITY_POINTS + 2


# --- 2. the predicted direction ---------------------------------------------


@pytest.fixture
def feat(monkeypatch):
    monkeypatch.setattr(T, "load_panel", lambda *a, **k: _panel(30, 600))
    s = Session()
    ph = T.fetch_data(s)["handle"]
    return s, T.compute_feature(s, ph, "mom_12_1")["handle"]


def test_predicting_the_bottom_flips_the_strategy_not_the_buckets(feat):
    s, fh = feat
    top = T.run_backtest(s, fh, n_buckets=5, prediction="top_beats_bottom")
    bot = T.run_backtest(s, fh, n_buckets=5, prediction="bottom_beats_top")
    # The panel's drift rises with momentum, so the top wins: long-top positive,
    # long-bottom negative.
    assert top["sharpe"] > 0 > bot["sharpe"]
    assert top["bucket_ann_return"] == bot["bucket_ann_return"]
    assert bot["legs"].startswith("long the bottom")
    assert T.analyze_results(s, top["handle"])["runs_as_predicted"] is True
    a = T.analyze_results(s, bot["handle"])
    assert a["runs_as_predicted"] is False
    assert a["clears_confirmation_gate"] is False


def test_a_staircase_panel_reads_as_a_perfect_staircase(feat):
    s, fh = feat
    out = T.run_backtest(s, fh, n_buckets=5)
    b = [out["bucket_ann_return"][str(i)] for i in range(1, 6)]
    assert b == sorted(b)
    assert out["bucket_monotonicity_spearman"] == 1.0


def test_an_undeclared_prediction_is_recorded_as_undeclared(feat):
    s, fh = feat
    out = T.run_backtest(s, fh)
    assert out["prediction"] == "top_beats_bottom"
    assert out["prediction_declared"] is False


def test_an_unknown_prediction_is_rejected():
    with pytest.raises(T.ToolError, match="prediction"):
        T.validate_arguments("run_backtest", {"feature_handle": "feature_1", "prediction": "up"})


def test_twenty_buckets_still_fit_the_summary_cap(feat):
    s, fh = feat
    out = T.run_backtest(s, fh, n_buckets=20, prediction="bottom_beats_top")
    assert len(json.dumps(out)) < MAX_SUMMARY_BYTES


def test_the_equity_curve_never_reaches_the_model(feat):
    s, fh = feat
    out = T.run_backtest(s, fh)
    assert "equity" not in json.dumps(out)
    assert len(s.payload(out["handle"])["diagnostics"]["equity"]) > 10


# --- 3. the verdict card ----------------------------------------------------


def _note(*records: BacktestRecord) -> Note:
    return Note(
        hypothesis="Does momentum predict returns?",
        prose="x",
        backtests=tuple(records),
        provenance=ProvenanceVerdict(ok=True, checked=0, verified=0),
        run=RunMetadata(model="m", stop_reason="end_turn", turns=3, tool_sequence=("analyze_results",)),
    )


def _bt(handle="backtest_1", sharpe=0.6, prob=0.5, long_short=True, declared=True, pred="top_beats_bottom", diag=None):
    return BacktestRecord(
        handle=handle,
        feature="mom_12_1",
        n_buckets=10,
        long_short=long_short,
        metrics={"sharpe": sharpe, "prediction": pred, "prediction_declared": declared, "n_invested_days": 900},
        statistics={
            "sharpe_annualised": sharpe,
            "prob_beats_best_of_n_trials": prob,
            "clears_confirmation_gate": prob >= 0.95,
            "runs_as_predicted": sharpe > 0,
            "n_trials_used": 2,
        },
        diagnostics=diag or {},
    )


@pytest.mark.parametrize(
    "sharpe, prob, outcome",
    [(0.9, 0.97, "supported"), (0.6, 0.5, "not_confirmed"), (-0.4, 0.01, "contradicted"), (-0.4, 0.99, "contradicted")],
)
def test_verdict_outcome_rule(sharpe, prob, outcome):
    assert _note(_bt(sharpe=sharpe, prob=prob)).verdict["outcome"] == outcome


def test_no_analysed_backtest_is_no_verdict():
    rec = BacktestRecord(handle="backtest_1", feature="f", n_buckets=10, long_short=True, analysis_error="x")
    assert _note(rec).verdict["outcome"] == "no_verdict"


def test_headline_prefers_a_declared_long_short_over_an_earlier_long_only():
    lo = _bt("backtest_1", sharpe=1.5, prob=0.99, long_short=False)
    ls = _bt("backtest_2", sharpe=-0.2, prob=0.1)
    n = _note(lo, ls)
    assert n.headline.handle == "backtest_2"
    assert n.verdict["outcome"] == "contradicted"


def test_verdict_reads_the_diagnostics_when_present_and_none_when_absent():
    diag = {
        "monotonicity": 0.9,
        "halves": [{"sharpe": 0.4}, {"sharpe": -0.1}],
        "costs": [{"cost_bps": 0, "sharpe": 0.7}, {"cost_bps": 25, "sharpe": 0.2}],
    }
    v = _note(_bt(diag=diag)).verdict
    assert (v["monotonicity"], v["both_halves_as_predicted"], v["survives_25bps"]) == (0.9, False, True)
    old = _note(_bt()).verdict
    assert old["monotonicity"] is None and old["both_halves_as_predicted"] is None


def test_diagnostics_round_trip_and_old_records_still_load():
    n = _note(_bt(diag={"monotonicity": 0.5, "equity": [{"ts": "2025-01-02", "equity": 1.0}]}))
    assert Note.from_dict(n.to_dict()).to_dict() == n.to_dict()
    d = n.to_dict()
    for b in d["backtests"]:
        del b["diagnostics"]
    assert Note.from_dict(d).backtests[0].diagnostics == {}


def test_a_stored_row_cannot_claim_its_own_verdict():
    d = _note(_bt(sharpe=-0.3, prob=0.01)).to_dict()
    d["verdict"] = {"outcome": "supported"}
    assert Note.from_dict(d).verdict["outcome"] == "contradicted"


def test_an_undeclared_run_never_claims_a_direction():
    """Old notes were built long-top whatever the question said."""
    v = _note(_bt(sharpe=1.1, prob=0.5, declared=False)).verdict
    assert v["outcome"] == "not_confirmed" and v["runs_as_predicted"] is None
    assert _note(_bt(sharpe=1.1, prob=0.99, declared=False)).verdict["outcome"] == "no_verdict"
