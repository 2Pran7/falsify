"""Module 2's closing test: does the engine reproduce a result we can verify
independently, on real market data?

Two tests, deliberately:

  1. A synthetic reconciliation that ALWAYS runs. It proves the engine's
     buy-and-hold output equals a direct cumprod of the same prices.
  2. The same reconciliation against real SPY bars from the database. Skipped
     automatically if Docker is down or SPY has not been ingested, so the suite
     stays green on a fresh clone.

Note what this does NOT do: compare against a number someone remembered. It
recomputes the answer a second, independent way and demands they agree. A
stored constant would go stale as the data window is extended;
an independent recomputation does not.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from falsify.backtest import metrics as m
from falsify.backtest.engine import run_backtest
from falsify.backtest.portfolio import fixed_weights


def _direct_buy_and_hold(prices: pl.DataFrame) -> float:
    """The answer computed WITHOUT the engine: last close over first close.

    Buy at the first close, sell at the last. No weights, no joins, no shifts.
    If the engine disagrees with this, the engine is wrong.
    """
    p = prices.sort("ts")["close"]
    return float(p[-1] / p[0] - 1.0)


# ---------------------------------------------------------------------------
# 1. Synthetic. Always runs.
# ---------------------------------------------------------------------------

def test_engine_matches_direct_buy_and_hold_synthetic():
    import random

    rng = random.Random(7)
    n = 500
    p, path = 100.0, [100.0]
    for _ in range(n - 1):
        p *= 1 + rng.gauss(0.0004, 0.011)
        path.append(p)

    ds = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    prices = pl.DataFrame(
        {"ticker": ["SPY"] * n, "ts": ds, "close": path},
        schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64},
    )

    res = run_backtest(prices, fixed_weights(prices["ts"], "SPY"))

    assert m.total_return(res.ret) == pytest.approx(_direct_buy_and_hold(prices), rel=1e-10)


def test_costs_only_reduce_returns():
    """Charging for trading cannot increase returns. Catches a sign flip
    on the cost term, which zero-cost tests would never notice."""
    ds = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(50)]
    path = [100.0 * (1.001 ** i) for i in range(50)]
    prices = pl.DataFrame(
        {"ticker": ["SPY"] * 50, "ts": ds, "close": path},
        schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64},
    )
    w = fixed_weights(prices["ts"], "SPY")

    from falsify.backtest.engine import BacktestConfig

    free = m.total_return(run_backtest(prices, w).ret)
    paid = m.total_return(run_backtest(prices, w, BacktestConfig(cost_bps=25.0)).ret)
    assert paid < free


# ---------------------------------------------------------------------------
# 2. Real data. Skips cleanly when the DB is not there.
# ---------------------------------------------------------------------------

def _load_spy() -> pl.DataFrame | None:
    try:
        from falsify.backtest.loader import load_prices

        df = load_prices(["SPY"])
        return df if len(df) > 100 else None
    except Exception:
        return None


spy = _load_spy()
needs_spy = pytest.mark.skipif(
    spy is None, reason="SPY not in daily_bars (run scripts/run_ingest.py SPY, Docker up)"
)


@needs_spy
def test_engine_reproduces_real_spy_buy_and_hold():
    res = run_backtest(spy, fixed_weights(spy["ts"], "SPY"))
    assert m.total_return(res.ret) == pytest.approx(_direct_buy_and_hold(spy), rel=1e-10)


@needs_spy
def test_real_spy_volatility_is_plausible():
    """A band, not a point. SPY's annualised vol has sat roughly between 10%
    and 25% for decades. If the engine reports 80%, something is wrong no
    matter how good the cumulative return looks."""
    res = run_backtest(spy, fixed_weights(spy["ts"], "SPY"))
    vol = m.ann_vol(res.ret)
    assert 0.05 < vol < 0.45, f"SPY annualised vol came out at {vol:.1%}"


@needs_spy
def test_real_spy_has_no_gaps_in_the_return_series():
    """Every trading day except the last must appear exactly once. A missing
    day means a silent join failure, which is how a backtest ends up reporting
    the Sharpe of a strategy that was not always invested."""
    res = run_backtest(spy, fixed_weights(spy["ts"], "SPY"))
    assert len(res.returns) == spy["ts"].n_unique() - 1
    assert res.returns["ts"].n_unique() == len(res.returns)
