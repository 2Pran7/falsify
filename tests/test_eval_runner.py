"""The suite runner. Two tests here are worth more than the rest combined:

    test_all_six_anomalies_share_one_session
    test_nothing_is_analysed_until_every_backtest_has_run

Both defend the trial count, and both guard against an implementation that
looks correct, runs cleanly, and reports every anomaly as less deflated than it
should be. A fresh Session per anomaly makes N=1 six times over; analysing as
each finishes holds six rows to six different standards, ordered by nothing
more meaningful than the registry tuple. Neither leaves a mark in the output.
"""
from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from falsify.agent import tools as T
from falsify.eval import registry as R
from falsify.eval.runner import compare_universes, measured_trial_variance, run_suite


def _panel(n_tickers: int = 14, n_days: int = 400, with_market: bool = True):
    """The same deterministic sine-wave panel as test_tools_universe."""
    days = pl.date_range(dt.date(2024, 1, 1), dt.date(2026, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)[:n_days]
    rows = []
    for i in range(n_tickers):
        drift = 0.0004 * (i - n_tickers / 2)
        phase = i * 0.7
        for j, d in enumerate(days):
            rows.append((f"T{i:02d}", d,
                         100.0 * math.exp(drift * j) * (1 + 0.05 * math.sin(0.11 * j + phase))))
    if with_market:
        for j, d in enumerate(days):
            rows.append(("SPY", d,
                         400.0 * math.exp(0.0002 * j) * (1 + 0.03 * math.sin(0.09 * j))))
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}, orient="row"
    )


def _snaps(dates: pl.Series, tickers: list[str], n: int = 4) -> pl.DataFrame:
    ds = dates.unique().sort()
    picks = [ds[int(i * len(ds) / n)] for i in range(n)]
    rows = []
    for k, d in enumerate(picks):
        members = tickers if k == 0 else [t for t in tickers if t != "T00"]
        rows += [(t, "SP500", d) for t in members]
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "index_name": pl.Utf8, "as_of": pl.Date},
        orient="row",
    )


@pytest.fixture
def wired(monkeypatch):
    panel = _panel()
    snaps = _snaps(panel["ts"], sorted(panel["ticker"].unique().to_list()))
    monkeypatch.setattr(T, "load_panel", lambda *a, **k: panel)
    monkeypatch.setattr(T, "load_snapshots", lambda *a, **k: snaps)
    return panel


# --- the two that carry the file -------------------------------------------


def test_all_six_anomalies_share_one_session(wired):
    """A fresh Session per anomaly would make N=1 every time.

    `Session.n_backtests` is the trial count for deflation and it is
    per-Session. With one Session per anomaly, every anomaly is the only trial
    that ever ran, no deflation is applied at all, and every row still looks
    like a deflated result.
    """
    run = run_suite()
    assert run.n_trials == 5, (
        "five of the six backtest on a two-year panel; long_term_reversal needs "
        "three years and never runs"
    )
    for a in run.analyses.values():
        assert a["n_trials_used"] == run.n_trials


def test_nothing_is_analysed_until_every_backtest_has_run(wired, monkeypatch):
    """THE ORDERING BUG THAT LEAVES NO TRACE.

    `analyze_results` takes the trial count WHEN IT IS CALLED. Analysing each
    anomaly as it finishes deflates the first against N=1 and the last against
    N=5: six rows held to six different standards, ordered by the registry
    tuple and nothing else. This records the call order and asserts every
    backtest precedes every analysis.
    """
    order: list[str] = []
    real_bt, real_an = T.run_backtest, T.analyze_results

    monkeypatch.setattr(
        T, "run_backtest",
        lambda *a, **k: (order.append("backtest"), real_bt(*a, **k))[1],
    )
    monkeypatch.setattr(
        T, "analyze_results",
        lambda *a, **k: (order.append("analyse"), real_an(*a, **k))[1],
    )

    run_suite()
    first_analysis = order.index("analyse")
    assert "backtest" not in order[first_analysis:], (
        "a backtest ran after an analysis: the earlier verdicts were deflated "
        "against a trial count that was still growing"
    )


# --- the suite runs the registry, all of it ---------------------------------


def test_every_registered_anomaly_gets_a_measurement_and_a_score(wired):
    run = run_suite()
    assert len(run.measurements) == len(R.ANOMALIES)
    assert [s.key for s in run.score] == [a.key for a in R.ANOMALIES]


def test_the_run_carries_the_registry_digest(wired):
    """Without it a stored verdict cannot be tied to the prediction it judged."""
    assert run_suite().registry_sha == R.registry_digest()


def test_long_term_reversal_is_insufficient_not_failed_on_a_two_year_panel(wired):
    """The honest outcome on this panel, and the reason there are four verdicts.

    The feature needs 756 trading days. `compute_feature` raises, the runner
    records the tool's own message, and the anomaly scores insufficient_data.
    A suite that reported this as a failure would be manufacturing a rejection
    out of a short sample.
    """
    run = run_suite()
    s = run.score["long_term_reversal"]
    assert s.verdict == "insufficient_data"
    assert "long_term_reversal" in run.errors
    assert "needs more history" in run.errors["long_term_reversal"]


def test_a_failing_anomaly_does_not_cost_the_others_their_verdicts(wired, monkeypatch):
    real = T.compute_feature

    def boom(session, handle, feature):
        if feature == "vol_63d":
            raise T.ToolError("synthetic failure")
        return real(session, handle, feature)

    monkeypatch.setattr(T, "compute_feature", boom)
    run = run_suite()
    assert run.score["low_volatility"].verdict == "insufficient_data"
    assert "synthetic failure" in run.errors["low_volatility"]
    assert sum(1 for s in run.score if s.verdict != "insufficient_data") >= 3


def test_the_tool_error_is_recorded_rather_than_swallowed(wired, monkeypatch):
    monkeypatch.setattr(
        T, "compute_feature",
        lambda *a, **k: (_ for _ in ()).throw(T.ToolError("everything is broken")),
    )
    run = run_suite()
    assert len(run.errors) == len(R.ANOMALIES)
    assert all(s.verdict == "insufficient_data" for s in run.score)


# --- orientation -------------------------------------------------------------


def test_the_p_value_is_computed_on_the_oriented_sharpe(wired):
    """`sharpe_pvalue` is one-sided "greater".

    Passing the raw spread would score every direction=-1 anomaly as
    insignificant exactly when its effect was strongest, and four of the six
    predict a negative spread.
    """
    run = run_suite()
    for mm in run.measurements:
        if mm.realised_sharpe is None or mm.p_value is None:
            continue
        oriented = mm.realised_sharpe * R.get(mm.key).direction
        # A positive oriented Sharpe must give p < 0.5 under a one-sided test,
        # and a negative one p > 0.5. Nothing about the raw sign is asserted.
        assert (mm.p_value < 0.5) == (oriented > 0), mm.key


def test_the_measurement_keeps_the_sharpe_signed_as_measured(wired):
    """So a reader can reconcile a stored row against the backtest output."""
    run = run_suite()
    for mm in run.measurements:
        if mm.realised_sharpe is None:
            continue
        assert mm.realised_sharpe == run.backtests[mm.key]["sharpe"]


# --- the universe -----------------------------------------------------------


def test_the_suite_runs_on_either_universe(wired):
    assert run_suite(universe="current").universe == "current"
    assert run_suite(universe="point_in_time").universe == "point_in_time"


def test_the_point_in_time_suite_holds_fewer_names(wired):
    cur = run_suite(universe="current")
    pit = run_suite(universe="point_in_time")
    key = "momentum_12_1"
    assert (
        pit.backtests[key]["n_investable_rows"] < cur.backtests[key]["n_investable_rows"]
    )


def test_compare_universes_reports_a_gap_per_anomaly(wired):
    """The Module 3 survivorship measurement generalised from one strategy to six."""
    out = compare_universes()
    assert set(out) == {"current", "point_in_time", "gaps"}
    assert len(out["gaps"]) == len(R.ANOMALIES)
    for g in out["gaps"]:
        assert set(g) == {
            "key", "current_sharpe", "pit_sharpe", "sharpe_gap",
            "current_verdict", "pit_verdict",
        }


def test_the_gap_is_current_minus_pit_on_the_oriented_sharpe(wired):
    out = compare_universes()
    for g in out["gaps"]:
        if g["sharpe_gap"] is None:
            continue
        assert g["sharpe_gap"] == pytest.approx(g["current_sharpe"] - g["pit_sharpe"])


def test_an_untestable_anomaly_has_no_gap_rather_than_a_gap_of_zero(wired):
    """A gap of 0.0 would read as "survivorship did not matter here"."""
    out = compare_universes()
    g = next(x for x in out["gaps"] if x["key"] == "long_term_reversal")
    assert g["sharpe_gap"] is None


# --- the trial variance -----------------------------------------------------


def test_the_measured_trial_variance_is_reported(wired):
    run = run_suite()
    assert run.trial_variance_measured is not None
    assert run.trial_variance_measured >= 0


def test_the_assumed_trial_variance_is_not_replaced_by_the_measured_one(wired):
    """Substituting a six-point estimate would replace a LABELLED assumption
    with an unlabelled one, invisible in every number downstream of it."""
    run = run_suite()
    assert run.trial_variance_assumed == T.TRIAL_VARIANCE
    for a in run.analyses.values():
        assert a["trial_variance_assumption"] == T.TRIAL_VARIANCE


def test_the_trial_variance_is_per_period_not_annualised():
    """The unit `expected_max_sharpe` consumes. An annualised variance is 252x
    too large and would inflate the benchmark enormously."""
    analyses = {
        "a": {"sharpe_per_period": 0.10},
        "b": {"sharpe_per_period": 0.20},
        "c": {"sharpe_per_period": 0.30},
    }
    # Sample variance of 0.1, 0.2, 0.3 is 0.01 exactly.
    assert measured_trial_variance(analyses) == pytest.approx(0.01)


def test_the_trial_variance_is_none_below_two_trials():
    assert measured_trial_variance({}) is None
    assert measured_trial_variance({"a": {"sharpe_per_period": 0.1}}) is None


# --- suite-level bookkeeping ------------------------------------------------


def test_the_panel_summary_records_the_trading_day_count(wired):
    """`history_days` in every verdict comes from here."""
    run = run_suite()
    assert run.panel["n_trading_days"] > 0
    for s in run.score:
        assert s.history_days == run.panel["n_trading_days"]


def test_all_six_share_one_bucket_count_and_one_cost(wired):
    """Varying either per anomaly would be a free parameter chosen after seeing
    the data, and six anomalies at three bucket counts is eighteen trials."""
    run = run_suite(n_buckets=5, cost_bps=0.0)
    for bt in run.backtests.values():
        assert bt["n_buckets"] == 5
        assert bt["cost_bps"] == 0.0


def test_every_backtest_is_long_short(wired):
    """The dollar-neutral spread is the correct test of a ranking hypothesis;
    long-only carries market beta that no tool here can separate out."""
    for bt in run_suite().backtests.values():
        assert bt["long_short"] is True
