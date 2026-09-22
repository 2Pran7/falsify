"""The universe argument, the Panel, and the three Module 6 features.

The disconnect this closes: before Module 6, `fetch_data` had no universe
argument at all. It loaded every ingested ticker, which IS the current S&P 500
membership, so every agent run was survivorship-inflated -- while
`stats/survivorship.py`, the module that measures exactly that, was reachable
only from a script. The project's headline finding and its headline artifact
did not touch.

The four tests that carry this file, because each defends a place where a
plausible alternative silently produces a wrong number:

    test_the_mask_restricts_holdings_and_never_the_price_history
    test_the_mask_is_applied_before_ranking_not_after
    test_point_in_time_refuses_to_run_on_a_single_snapshot_date
    test_idio_vol_is_null_everywhere_without_the_market_series

THE PANEL USED HERE IS NOT THE ONE IN test_agent_tools.py. That one is a pure
exponential drift, which has ZERO rolling return variance -- a constant daily
log return -- so any beta against it is undefined. Harmless everywhere it was
already used, and fatal for `ivol_63d`, which needs variation to have a
residual at all. This one is built from a sine wave: still fully deterministic,
because a sampled panel makes a test that fails once a month, and a test that
fails once a month gets deleted.
"""
from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from falsify.agent import tools as T
from falsify.agent.session import Session
from falsify.features import library as feat


# --- fixtures ---------------------------------------------------------------


def _wavy_panel(n_tickers: int = 12, n_days: int = 400, with_market: bool = True):
    """Deterministic prices with real return variation, plus an optional SPY.

    Ticker i gets a drift plus a sine wave at its own phase, so cross-sectional
    ranks move over time and every rolling variance is strictly positive.
    """
    days = pl.date_range(dt.date(2024, 1, 1), dt.date(2026, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)[:n_days]
    rows = []
    for i in range(n_tickers):
        drift = 0.0004 * (i - n_tickers / 2)
        phase = i * 0.7
        for j, d in enumerate(days):
            level = 100.0 * math.exp(drift * j) * (1.0 + 0.05 * math.sin(0.11 * j + phase))
            rows.append((f"T{i:02d}", d, level))
    if with_market:
        for j, d in enumerate(days):
            rows.append(("SPY", d, 400.0 * math.exp(0.0002 * j) * (1.0 + 0.03 * math.sin(0.09 * j))))
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}, orient="row"
    )


def _snapshots(dates: pl.Series, tickers: list[str], n_snaps: int = 4) -> pl.DataFrame:
    """Dated membership snapshots that DROP a ticker part way through.

    T00 is a member in the first snapshot and absent from every later one, so
    the point-in-time universe is genuinely smaller than the current one and
    the mask has something to do.
    """
    ds = dates.unique().sort()
    picks = [ds[int(i * len(ds) / n_snaps)] for i in range(n_snaps)]
    rows = []
    for k, d in enumerate(picks):
        members = tickers if k == 0 else [t for t in tickers if t != "T00"]
        rows += [(t, "SP500", d) for t in members]
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "index_name": pl.Utf8, "as_of": pl.Date},
        orient="row",
    )


@pytest.fixture
def s() -> Session:
    return Session()


@pytest.fixture
def pit(monkeypatch):
    """A panel plus four membership snapshots, both patched in at the loader."""
    panel = _wavy_panel()
    snaps = _snapshots(panel["ts"], sorted(panel["ticker"].unique().to_list()))
    monkeypatch.setattr(T, "load_panel", lambda *a, **k: panel)
    monkeypatch.setattr(T, "load_snapshots", lambda *a, **k: snaps)
    return panel, snaps


@pytest.fixture
def one_snapshot(monkeypatch):
    """What run_ingest.py leaves behind on its own: a single as_of date."""
    panel = _wavy_panel()
    tickers = sorted(panel["ticker"].unique().to_list())
    snaps = pl.DataFrame(
        [(t, "SP500", panel["ts"].max()) for t in tickers],
        schema={"ticker": pl.Utf8, "index_name": pl.Utf8, "as_of": pl.Date},
        orient="row",
    )
    monkeypatch.setattr(T, "load_panel", lambda *a, **k: panel)
    monkeypatch.setattr(T, "load_snapshots", lambda *a, **k: snaps)
    return panel


# --- the closed menu still closes ------------------------------------------


def test_universe_is_a_closed_set():
    assert T.UNIVERSES == ("current", "point_in_time")


def test_an_unknown_universe_is_rejected_by_the_validator():
    with pytest.raises(T.ToolError, match="unknown universe"):
        T.validate_arguments("fetch_data", {"universe": "survivorship_free"})


def test_the_fetch_data_schema_advertises_the_universe_enum():
    schema = next(t for t in T.tool_schemas() if t["name"] == "fetch_data")
    prop = schema["input_schema"]["properties"]["universe"]
    assert prop["enum"] == list(T.UNIVERSES)
    assert "survivorship" in prop["description"].lower(), (
        "the description is the only place the model learns that the default is biased"
    )


def test_the_default_universe_is_current_and_says_so(s, pit):
    out = T.fetch_data(s)
    assert out["universe"] == "current"
    assert s.payload(out["handle"], "panel").membership is None


# --- the Panel carries the universe ----------------------------------------


def test_the_panel_carries_frame_universe_and_membership(s, pit):
    h = T.fetch_data(s, universe="point_in_time")["handle"]
    p = s.payload(h, "panel")
    assert isinstance(p.frame, pl.DataFrame)
    assert p.universe == "point_in_time"
    assert set(p.membership.columns) == {"ts", "ticker"}


def test_the_universe_survives_compute_feature(s, pit):
    """A frame alone carries no record of which universe produced it, and three
    tools later nothing can tell the difference."""
    ph = T.fetch_data(s, universe="point_in_time")["handle"]
    out = T.compute_feature(s, ph, "mom_12_1")
    assert out["universe"] == "point_in_time"
    assert s.payload(out["handle"], "feature").membership is not None


def test_the_universe_is_reported_on_the_backtest(s, pit):
    ph = T.fetch_data(s, universe="point_in_time")["handle"]
    fh = T.compute_feature(s, ph, "mom_12_1")["handle"]
    assert T.run_backtest(s, fh)["universe"] == "point_in_time"


# --- the two decisions that are the whole of the argument -------------------


def test_the_mask_restricts_holdings_and_never_the_price_history(s, pit):
    """A company that joined the index in March had prices in February.

    Filtering the PRICE frame to member-days would destroy that history and
    leave every entrant unscored for its first year: a lookahead bug in
    reverse, which would make the point-in-time universe look worse than it is
    for a reason unrelated to survivorship.
    """
    panel, _ = pit
    cur = T.fetch_data(s, universe="current")
    pitout = T.fetch_data(s, universe="point_in_time")
    assert pitout["n_rows"] == cur["n_rows"] == len(panel)
    assert pitout["n_tickers"] == cur["n_tickers"]

    # ... and the feature is computed over the whole frame, T00 included.
    fh = T.compute_feature(s, pitout["handle"], "mom_12_1")["handle"]
    frame = s.payload(fh, "feature").frame
    assert "T00" in frame.drop_nulls("mom_12_1")["ticker"].unique().to_list()


def test_the_mask_is_applied_before_ranking_not_after(s, pit, monkeypatch):
    """Bucket edges must come from the names investable that day.

    Filtering after ranking leaves the deciles defined by a universe the
    strategy could not have traded -- a non-member still pushes a real holding
    out of the top bucket -- and the output still looks exactly like a
    backtest. This test captures the frame `decile_weights` actually receives.
    """
    seen: list[pl.DataFrame] = []
    real = T.decile_weights

    def spy(signal, *a, **k):
        seen.append(signal)
        return real(signal, *a, **k)

    monkeypatch.setattr(T, "decile_weights", spy)

    ph = T.fetch_data(s, universe="point_in_time")["handle"]
    fh = T.compute_feature(s, ph, "mom_12_1")["handle"]
    T.run_backtest(s, fh)

    assert seen, "decile_weights was never called"
    ranked = seen[0]
    membership = s.payload(ph, "panel").membership
    dropped_after_first = membership.filter(pl.col("ticker") == "T00")["ts"]
    late = ranked.filter(pl.col("ticker") == "T00")["ts"]
    assert set(late.to_list()) <= set(dropped_after_first.to_list()), (
        "T00 was ranked on a date it was not a member: the mask ran after the sort"
    )


def test_the_point_in_time_run_holds_fewer_names_than_the_current_one(s, pit):
    """The measurable consequence of the mask. If this ever came out equal, the
    mask is not being applied and every downstream comparison is of a universe
    with itself."""
    ph = T.fetch_data(s, universe="current")["handle"]
    fh = T.compute_feature(s, ph, "mom_12_1")["handle"]
    cur = T.run_backtest(s, fh)

    ph2 = T.fetch_data(s, universe="point_in_time")["handle"]
    fh2 = T.compute_feature(s, ph2, "mom_12_1")["handle"]
    pitr = T.run_backtest(s, fh2)

    assert pitr["n_investable_rows"] < cur["n_investable_rows"]
    assert pitr["n_scored_rows"] == cur["n_scored_rows"], (
        "the same rows were SCORED; only what may be held differs"
    )


def test_point_in_time_refuses_to_run_on_a_single_snapshot_date(s, one_snapshot):
    """THE FAILURE MODE THAT FAILS BY LOOKING HEALTHY.

    Membership that never changes IS today's membership. It reproduces the
    current-constituents result exactly, with full coverage, no gap, and a
    survivorship audit that measures zero -- and every number in the output
    looks entirely reasonable. A check that refuses to run is worth more than a
    check that warns.
    """
    with pytest.raises(T.ToolError, match="point_in_time needs dated membership"):
        T.fetch_data(s, universe="point_in_time")


def test_the_single_snapshot_error_says_what_to_run(s, one_snapshot):
    """A tool error is read by the model and by Brev; both need the next step."""
    with pytest.raises(T.ToolError, match="build_pit_universe"):
        T.fetch_data(s, universe="point_in_time")


def test_current_still_works_when_there_are_no_snapshots_at_all(s, monkeypatch):
    """The default path must not acquire a dependency on the snapshot table."""
    panel = _wavy_panel()
    monkeypatch.setattr(T, "load_panel", lambda *a, **k: panel)
    monkeypatch.setattr(
        T, "load_snapshots",
        lambda *a, **k: pl.DataFrame(
            schema={"ticker": pl.Utf8, "index_name": pl.Utf8, "as_of": pl.Date}
        ),
    )
    assert T.fetch_data(s, universe="current")["universe"] == "current"


# --- the feature-column map -------------------------------------------------


def test_the_feature_column_map_is_identity_unless_stated():
    assert T.feature_column("mom_12_1") == "mom_12_1"
    assert T.feature_column("pct_52w_high") == "pct_252d_high"


def test_a_feature_whose_column_differs_from_its_key_backtests_end_to_end(s, pit):
    """The bug found while building: `compute_feature` assumed the output column
    always equalled the menu key. `pct_52w_high` appends `pct_252d_high`, and
    the failure would have been a KeyError three frames down rather than a
    usable tool error."""
    ph = T.fetch_data(s)["handle"]
    out = T.compute_feature(s, ph, "pct_52w_high")
    assert out["feature"] == "pct_52w_high"
    assert out["feature_column"] == "pct_252d_high"
    bt = T.run_backtest(s, out["handle"])
    assert bt["feature"] == "pct_52w_high"
    assert bt["n_invested_days"] > 0


def test_a_builder_renamed_without_the_map_fails_with_both_names(s, pit, monkeypatch):
    monkeypatch.setitem(
        T._BUILDERS, "vol_21d",
        lambda df: feat.add_rolling_vol(df, 21).rename({"vol_21d": "moved"}),
    )
    ph = T.fetch_data(s)["handle"]
    with pytest.raises(T.ToolError, match="did not produce column 'vol_21d'"):
        T.compute_feature(s, ph, "vol_21d")


# --- the three new features -------------------------------------------------


def test_the_three_new_features_are_on_the_menu():
    for k in ("rev_36_12", "pct_52w_high", "ivol_63d"):
        assert k in T.FEATURES and k in T._BUILDERS


def test_rev_36_12_skips_the_most_recent_year():
    """The skip is the construction, not a convenience: without it the window
    holds momentum and reversal, which run opposite ways over that horizon."""
    panel = _wavy_panel(n_tickers=2, n_days=900, with_market=False)
    out = feat.add_reversal_36_12(panel)
    row = out.filter(pl.col("ticker") == "T00").sort("ts")
    closes = row["close"].to_list()
    last = len(closes) - 1
    assert last >= 756, "the fixture must be long enough for a three-year window"
    # Hand-derived, not read off the implementation: the value at t is the
    # return from t-756 to t-252, so the most recent YEAR is excluded entirely.
    expected = closes[last - 252] / closes[last - 756] - 1
    assert row["rev_36_12"].to_list()[last] == pytest.approx(expected)

    # And the discriminating half: it is NOT the plain three-year return.
    plain = closes[last] / closes[last - 756] - 1
    assert row["rev_36_12"].to_list()[last] != pytest.approx(plain)


def test_rev_36_12_is_null_on_a_two_year_panel(s, pit):
    """The reason long_term_reversal is pre-registered as insufficient_data."""
    ph = T.fetch_data(s)["handle"]
    with pytest.raises(T.ToolError, match="needs more history"):
        T.compute_feature(s, ph, "rev_36_12")


def test_pct_52w_high_is_bounded_in_zero_to_one():
    panel = _wavy_panel(n_tickers=3, n_days=400, with_market=False)
    col = feat.add_pct_52w_high(panel)["pct_252d_high"].drop_nulls()
    assert col.min() > 0.0
    assert col.max() == pytest.approx(1.0), "a name at its own rolling max scores 1"
    assert col.max() <= 1.0 + 1e-12


def test_idio_vol_is_null_everywhere_without_the_market_series():
    """IT MUST NOT FALL BACK TO TOTAL VOLATILITY.

    A fallback would score the low-volatility effect a second time under a
    different name and let the suite count it as independent evidence: two rows
    that look like two findings. That is the worst failure an eval suite can
    have, and it is the tempting implementation.
    """
    panel = _wavy_panel(n_tickers=4, n_days=300, with_market=False)
    out = feat.add_idio_vol(panel, 63)
    assert out["ivol_63d"].null_count() == len(out)


def test_idio_vol_differs_from_total_vol_when_the_market_is_present():
    """The positive half of the same point: with SPY there, the residual
    volatility is not the total volatility."""
    panel = _wavy_panel(n_tickers=4, n_days=300, with_market=True)
    out = feat.add_idio_vol(feat.add_rolling_vol(panel, 63), 63).drop_nulls("ivol_63d")
    assert len(out) > 0
    diffs = (out["ivol_63d"] - out["vol_63d"]).abs()
    assert diffs.max() > 1e-6


def test_idio_vol_obeys_the_least_squares_projection_inequality():
    """The analytic identity that pins the arithmetic: SSE <= sum(r^2).

    Projecting onto the market can only REMOVE sum of squares, so the residual
    cannot exceed the uncentred total. Note the comparison is against the
    UNCENTRED root-mean-square, not against `vol_63d`: vol_63d is a deviation
    from the rolling MEAN, while this regression has NO INTERCEPT and its
    residual is about zero. Where the mean return is non-zero, ivol can
    legitimately exceed vol_63d, and asserting otherwise would be asserting a
    property this construction does not have.
    """
    panel = _wavy_panel(n_tickers=4, n_days=300, with_market=True)
    ref = (
        feat.add_log_returns(panel)
        .sort(["ticker", "ts"])
        .with_columns((pl.col("logret_1d") ** 2).alias("rr"))
        .with_columns(
            ((pl.col("rr").rolling_sum(63).over("ticker") / 62) ** 0.5 * (252**0.5))
            .alias("uncentred_vol")
        )
        .select(["ticker", "ts", "uncentred_vol"])
    )
    out = (
        feat.add_idio_vol(panel, 63)
        .join(ref, on=["ticker", "ts"], how="left")
        .drop_nulls(["ivol_63d", "uncentred_vol"])
    )
    assert len(out) > 0
    assert (out["ivol_63d"] <= out["uncentred_vol"] + 1e-9).all()


def test_idio_vol_excludes_the_market_from_its_own_output():
    """SPY's residual against itself is identically zero, and a zero-volatility
    name sorts to the top of a low-vol ranking forever."""
    panel = _wavy_panel(n_tickers=3, n_days=300, with_market=True)
    out = feat.add_idio_vol(panel, 63)
    spy = out.filter(pl.col("ticker") == "SPY")
    assert spy["ivol_63d"].null_count() == len(spy)


def test_ivol_backtests_end_to_end_on_the_wavy_panel(s, pit):
    ph = T.fetch_data(s)["handle"]
    fh = T.compute_feature(s, ph, "ivol_63d")["handle"]
    assert T.run_backtest(s, fh)["n_invested_days"] > 0


# --- the summary cap still holds -------------------------------------------


def test_no_universe_tool_result_leaks_a_payload(s, pit):
    """Rule 3 of the tool boundary, re-checked for the new return keys."""
    import json

    from falsify.agent.session import MAX_SUMMARY_BYTES

    outs = [T.fetch_data(s, universe="point_in_time")]
    outs.append(T.compute_feature(s, outs[0]["handle"], "mom_12_1"))
    outs.append(T.run_backtest(s, outs[1]["handle"]))
    for o in outs:
        assert len(json.dumps(o, default=str)) < MAX_SUMMARY_BYTES
        for v in o.values():
            assert not isinstance(v, (list, tuple, pl.DataFrame, pl.Series))


# --- found by the eval suite's first real run -------------------------------
#
# These belong to `analyze_results`, which is a Module 4 file, but they are here
# because the eval suite is what exposed the defect and the eval suite is why it
# matters. Four of the six registered anomalies predict a NEGATIVE raw spread.


def _losing_series(n: int = 400) -> pl.Series:
    """A deterministic return series with a negative Sharpe and real variance."""
    return pl.Series("ret", [-0.0005 + 0.01 * math.sin(0.3 * i) for i in range(n)])


def test_a_negative_sharpe_still_gets_its_probability_statistics():
    """THE BUG THE FIRST REAL EVAL RUN FOUND.

    "How long before this Sharpe is distinguishable from zero?" has no finite
    answer when the Sharpe is BELOW zero, so `min_track_record_length` raises,
    correctly. The original `analyze_results` wrapped all five statistics in one
    try, so one legitimately-undefined quantity returned NOTHING at all.

    PSR and the deflated Sharpe are perfectly computable on a losing series. A
    PSR of 0.05 is a real and useful statement, and it was being thrown away.
    """
    from falsify.stats import deflated

    r = _losing_series()
    st = deflated.sharpe_stats(r)
    assert st["sr"] < 0

    # The two that must survive.
    assert 0.0 <= deflated.probabilistic_sharpe(
        st["sr"], st["skew"], st["kurt"], int(st["n"])
    ) <= 1.0
    assert 0.0 <= deflated.deflated_sharpe(r, 5, T.TRIAL_VARIANCE) <= 1.0

    # The one that is genuinely undefined, and stays that way.
    with pytest.raises(ValueError, match="MinTRL is infinite"):
        deflated.min_track_record_length(st["sr"], st["skew"], st["kurt"])


def test_analyze_results_survives_an_undefined_min_track_record_length(s, monkeypatch):
    """The whole call must still return, with the undefined field set to None.

    WHY THIS IS NOT COSMETIC: a working low-volatility or reversal anomaly
    produces exactly the negative raw spread that triggers this. The eval suite
    would then see no deflation statistic, apply its own rule that a missing
    statistic is not a satisfied condition, and score a REDISCOVERED ANOMALY AS
    A FAILURE — with a reason that reads as principled.
    """
    r = _losing_series()
    handle = s.put(
        "backtest",
        {"result": None, "invested": pl.DataFrame({"ret": r})},
        {"sharpe": -1.0},
    )
    out = T.analyze_results(s, handle)

    assert out["sharpe_per_period"] < 0
    assert out["min_track_record_length_days"] is None
    assert 0.0 <= out["prob_sharpe_above_zero"] <= 1.0
    assert 0.0 <= out["prob_beats_best_of_n_trials"] <= 1.0


def test_the_undefined_track_record_carries_a_reason_rather_than_a_bare_none(s):
    """A None with nothing attached reads as "we failed to compute it" when it
    means "the answer is infinite". Same rule as every other verdict in this
    project: the number is useless without why."""
    handle = s.put(
        "backtest",
        {"result": None, "invested": pl.DataFrame({"ret": _losing_series()})},
        {"sharpe": -1.0},
    )
    out = T.analyze_results(s, handle)
    note = out.get("min_track_record_length_note", "")
    assert "undefined" in note and "not an error" in note


def test_a_positive_sharpe_still_reports_a_finite_track_record_length(s):
    """The control. If this ever came back None too, the fix above would have
    turned a real statistic into a silent None for every strategy."""
    r = pl.Series("ret", [0.002 + 0.01 * math.sin(0.3 * i) for i in range(400)])
    handle = s.put(
        "backtest", {"result": None, "invested": pl.DataFrame({"ret": r})}, {"sharpe": 1.0}
    )
    out = T.analyze_results(s, handle)
    assert out["min_track_record_length_days"] is not None
    assert out["min_track_record_length_days"] > 0
    assert "min_track_record_length_note" not in out


def test_a_working_negative_direction_anomaly_is_not_failed_for_a_missing_statistic():
    """The end-to-end consequence, at the scorer.

    An anomaly that WORKS in the direction its paper predicted — raw spread
    negative, oriented spread positive — must reach the deflation gate with a
    real number. Before the fix it reached it with None and was failed.
    """
    from falsify.eval import registry as R
    from falsify.eval.score import Measurement, score_anomaly

    worked = Measurement(
        key="low_volatility",
        realised_sharpe=-0.9,      # raw: the BOTTOM bucket won, as predicted
        p_value=0.001,
        deflated_psr=0.97,         # available, because MinTRL no longer kills it
        history_days=800,
        n_invested_days=400,
        n_trials=5,
    )
    assert score_anomaly(worked, R.get("low_volatility"), 0.001).verdict == "pass"

    starved = Measurement(**{**worked.__dict__, "deflated_psr": None})
    starved_score = score_anomaly(starved, R.get("low_volatility"), 0.001)
    assert starved_score.verdict == "fail"
    assert any("missing statistic" in r for r in starved_score.reasons), (
        "this is what the bug produced: a rediscovered anomaly failed for a "
        "reason that reads as principled"
    )
