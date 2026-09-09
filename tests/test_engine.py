"""Engine tests.

Two bracket the timing rule from both sides. The positive control
(test_oracle_signal_prints_an_absurd_sharpe) shows the engine can express a
profitable strategy at all, so a flat result elsewhere means "correctly earned
nothing" rather than "silently broken". The negative control
(test_alternating_series_strategy_must_lose) shows it does not credit
tomorrow's return today.
"""
from __future__ import annotations

import datetime as dt
import math
import random

import polars as pl
import pytest

from falsify.backtest import metrics as m
from falsify.backtest.engine import (
    BacktestConfig,
    compute_turnover,
    forward_returns,
    run_backtest,
)

D0 = dt.date(2024, 1, 1)


def _dates(n: int) -> list[dt.date]:
    return [D0 + dt.timedelta(days=i) for i in range(n)]


def _prices(paths: dict[str, list[float]]) -> pl.DataFrame:
    """{'AAA': [100, 110, ...], 'BBB': [...]} -> long (ticker, ts, close) frame."""
    n = len(next(iter(paths.values())))
    ds = _dates(n)
    rows = [(tk, ds[i], float(p)) for tk, path in paths.items() for i, p in enumerate(path)]
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}, orient="row"
    ).sort(["ticker", "ts"])


def _weights(rows: list[tuple[dt.date, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"ts": pl.Date, "ticker": pl.Utf8, "w": pl.Float64}, orient="row"
    ).sort(["ts", "ticker"])


def _long_the_higher_signal(signal: pl.DataFrame) -> pl.DataFrame:
    """signal: (ts, ticker, sig). Go 100% long the top-ranked name each date.
    Dates where every signal is null produce no weights at all."""
    s = signal.drop_nulls("sig")
    if s.is_empty():
        return _weights([])
    top = (
        s.filter(pl.col("sig") == pl.col("sig").max().over("ts"))
        .sort(["ts", "ticker"])
        .unique(subset=["ts"], keep="first", maintain_order=True)
    )
    return _weights([(r["ts"], r["ticker"], 1.0) for r in top.iter_rows(named=True)])


# --------------------------------------------------------------------------
# forward_returns
# --------------------------------------------------------------------------

def test_forward_returns_looks_exactly_one_day_ahead():
    prices = _prices({"AAA": [100.0, 110.0, 121.0]})
    out = forward_returns(prices).sort("ts")

    assert out["fwd_ret"].to_list()[:2] == pytest.approx([0.10, 0.10])
    assert out["fwd_ret"].to_list()[2] is None, "last observation has no tomorrow"


def test_forward_returns_does_not_leak_across_tickers():
    """The classic long-format bug: ticker A's final row reaching into ticker
    B's first. test_features.py guards the backward shift; same failure mode
    here on the forward one."""
    prices = _prices({"AAA": [100.0, 200.0], "BBB": [50.0, 25.0]})
    out = forward_returns(prices)

    a = out.filter(pl.col("ticker") == "AAA").sort("ts")["fwd_ret"].to_list()
    b = out.filter(pl.col("ticker") == "BBB").sort("ts")["fwd_ret"].to_list()

    assert a[0] == pytest.approx(1.0)
    assert a[1] is None
    assert b[0] == pytest.approx(-0.5)
    assert b[1] is None


# --------------------------------------------------------------------------
# turnover
# --------------------------------------------------------------------------

def test_turnover_first_date_is_the_full_position():
    ds = _dates(1)
    w = _weights([(ds[0], "AAA", 0.6), (ds[0], "BBB", -0.4)])
    out = compute_turnover(w).sort("ts")
    assert out["turnover"].to_list() == pytest.approx([1.0])


def test_turnover_counts_absolute_changes_and_treats_exits_as_changes():
    ds = _dates(2)
    w = _weights([
        (ds[0], "AAA", 1.0),
        (ds[1], "BBB", 1.0),          # AAA fully exited, BBB fully entered
    ])
    out = compute_turnover(w).sort("ts")
    assert out["turnover"].to_list() == pytest.approx([1.0, 2.0])


def test_turnover_is_zero_when_weights_are_unchanged():
    ds = _dates(3)
    w = _weights([(d, "AAA", 1.0) for d in ds])
    out = compute_turnover(w).sort("ts")
    assert out["turnover"].to_list() == pytest.approx([1.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# run_backtest, basic accounting
# --------------------------------------------------------------------------

def test_flat_prices_earn_exactly_zero():
    prices = _prices({"AAA": [100.0] * 5, "BBB": [100.0] * 5})
    ds = _dates(5)
    w = _weights([(d, "AAA", 1.0) for d in ds])

    res = run_backtest(prices, w)
    assert res.ret.to_list() == pytest.approx([0.0] * 4)


def test_single_ticker_known_path():
    """100 -> 110 -> 121, fully invested. Total return must be exactly 21%."""
    prices = _prices({"AAA": [100.0, 110.0, 121.0]})
    ds = _dates(3)
    w = _weights([(d, "AAA", 1.0) for d in ds])

    res = run_backtest(prices, w)
    assert res.ret.to_list() == pytest.approx([0.10, 0.10])
    assert m.total_return(res.ret) == pytest.approx(0.21, rel=1e-12)


def test_final_date_is_dropped_not_zeroed():
    """The final date has no forward return. Dropping it and recording 0.0 are
    not equivalent: a silent zero drags every downstream mean and volatility."""
    prices = _prices({"AAA": [100.0, 110.0, 121.0, 133.1]})
    ds = _dates(4)
    w = _weights([(d, "AAA", 1.0) for d in ds])

    res = run_backtest(prices, w)
    assert len(res.returns) == 3
    assert res.returns["ts"].max() == ds[2]


def test_dates_without_weights_are_flat_but_still_reported():
    """Out-of-position days belong in the series. Omitting them would report
    the Sharpe of a strategy that was never actually run."""
    prices = _prices({"AAA": [100.0, 110.0, 121.0, 133.1]})
    ds = _dates(4)
    w = _weights([(ds[2], "AAA", 1.0)])          # only in the market on day 2

    res = run_backtest(prices, w)
    assert len(res.returns) == 3
    assert res.ret.to_list() == pytest.approx([0.0, 0.0, 0.10])


def test_costs_are_charged_on_turnover():
    """Flat prices, so gross return is 0 and the only thing left is the cost.
    Entering a 100% position is turnover 1.0; at 10bps that is exactly -0.0010."""
    prices = _prices({"AAA": [100.0] * 3})
    ds = _dates(3)
    w = _weights([(d, "AAA", 1.0) for d in ds])

    res = run_backtest(prices, w, BacktestConfig(cost_bps=10.0))
    assert res.ret.to_list() == pytest.approx([-0.0010, 0.0])
    assert res.returns["gross_ret"].to_list() == pytest.approx([0.0, 0.0])


def test_zero_cost_config_is_the_default():
    prices = _prices({"AAA": [100.0] * 3})
    ds = _dates(3)
    w = _weights([(d, "AAA", 1.0) for d in ds])
    assert run_backtest(prices, w).ret.to_list() == pytest.approx([0.0, 0.0])


# --------------------------------------------------------------------------
# Timing controls
# --------------------------------------------------------------------------

def test_oracle_signal_prints_an_absurd_sharpe():
    """Positive control.

    The signal at t is the return from t to t+1: perfect foresight, which a
    correct engine awards an enormous Sharpe. The point is not that foresight
    is interesting. It establishes that a 0.0 elsewhere means the engine
    correctly earned nothing rather than returning zero for everything, without
    which a flat result proves nothing.
    """
    rng = random.Random(42)
    n = 300
    paths = {}
    for tk in ("AAA", "BBB"):
        p, path = 100.0, [100.0]
        for _ in range(n - 1):
            p *= 1 + rng.gauss(0, 0.02)
            path.append(p)
        paths[tk] = path
    prices = _prices(paths)

    fwd = forward_returns(prices)
    signal = fwd.select(["ts", "ticker", pl.col("fwd_ret").alias("sig")])
    w = _long_the_higher_signal(signal)

    res = run_backtest(prices, w)
    assert m.sharpe(res.ret) > 5.0, "engine cannot express a profitable strategy at all"
    assert m.total_return(res.ret) > 1.0


def test_alternating_series_strategy_must_lose():
    """Negative control.

    AAA alternates +10%, -10% forever; BBB never moves. The signal is
    YESTERDAY's realised return, strictly backward-looking, and the strategy
    buys whichever name rose most recently. On an alternating series that is a
    machine for buying the top: every time it goes long AAA, AAA's next move is
    -10%, so it must bleed out.

    Under an off-by-one that applies each date's weights to that date's
    already-realised return, the same strategy compounds at +10% on half the
    days and the result flips from catastrophic to spectacular. Inspecting a
    Sharpe ratio would not reveal that; one assertion here does.
    """
    n = 41
    path, p = [100.0], 100.0
    for i in range(n - 1):
        p *= 1.1 if i % 2 == 0 else 0.9
        path.append(p)
    prices = _prices({"AAA": path, "BBB": [100.0] * n})

    # Signal = the return that already happened, from t-1 to t.
    signal = (
        prices.sort(["ticker", "ts"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("sig")
        )
        .select(["ts", "ticker", "sig"])
    )
    w = _long_the_higher_signal(signal)

    # Check the setup before trusting the verdict: on every date where AAA has
    # just risen, the strategy should be holding AAA.
    held = dict(zip(w["ts"].to_list(), w["ticker"].to_list()))
    rose = (
        signal.filter((pl.col("ticker") == "AAA") & (pl.col("sig") > 0))["ts"].to_list()
    )
    assert all(held.get(d) == "AAA" for d in rose), "test setup is wrong, not the engine"

    res = run_backtest(prices, w)

    assert m.total_return(res.ret) < -0.5, (
        "buying the most recent winner on a strictly alternating series made "
        "money. The engine is applying weights to the contemporaneous return "
        "instead of the forward return."
    )
    assert m.sharpe(res.ret) < 0.0


def test_negative_control_is_actually_discriminating():
    """Guards the guard.

    Reproduces the off-by-one deliberately and confirms the same setup turns
    strongly positive. Were this to fail, the negative control above would no
    longer discriminate and would be giving false assurance.
    """
    n = 41
    path, p = [100.0], 100.0
    for i in range(n - 1):
        p *= 1.1 if i % 2 == 0 else 0.9
        path.append(p)
    prices = _prices({"AAA": path, "BBB": [100.0] * n})

    realised = prices.sort(["ticker", "ts"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("sig")
    )
    w = _long_the_higher_signal(realised.select(["ts", "ticker", "sig"]))

    # The bug: pair each date's weight with that date's OWN realised return.
    buggy = (
        w.join(realised.select(["ts", "ticker", "sig"]), on=["ts", "ticker"], how="left")
        .with_columns((pl.col("w") * pl.col("sig")).alias("r"))
        .group_by("ts").agg(pl.col("r").sum())
        .sort("ts")
    )
    assert m.total_return(buggy["r"]) > 0.5, (
        "the alternating construction no longer separates correct from buggy"
    )


def test_result_object_carries_the_full_return_series():
    """The statistics layer consumes the daily series, the weight history and
    the config, so the result object's shape is pinned by a test."""
    prices = _prices({"AAA": [100.0, 110.0, 121.0]})
    ds = _dates(3)
    w = _weights([(d, "AAA", 1.0) for d in ds])

    res = run_backtest(prices, w, BacktestConfig(cost_bps=5.0))

    assert set(res.returns.columns) >= {"ts", "gross_ret", "cost", "ret"}
    assert set(res.weights.columns) >= {"ts", "ticker", "w"}
    assert set(res.turnover.columns) >= {"ts", "turnover"}
    assert res.config.cost_bps == 5.0
    assert res.returns["ts"].is_sorted()
    assert not math.isnan(m.total_return(res.ret))
