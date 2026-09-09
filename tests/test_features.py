"""Feature library tests on synthetic data with hand-computable answers.

Expected values are derived by hand rather than captured from the
implementation, so the tests constrain the features rather than describe them.
"""
import datetime as dt

import polars as pl
import pytest

from falsify.features.library import (
    add_momentum_12_1,
    add_returns,
    add_rolling_vol,
    add_sma,
    add_zscore,
    build_standard_features,
)


def make_synthetic(n_days: int = 300, tickers: tuple[str, ...] = ("AAA", "BBB")) -> pl.DataFrame:
    """Deterministic prices: AAA grows 1%/day exactly, BBB is flat at 50."""
    start = dt.date(2024, 1, 1)
    rows = []
    for t in tickers:
        price = 100.0
        for i in range(n_days):
            price = price * 1.01 if t == "AAA" else 50.0
            rows.append({"ticker": t, "ts": start + dt.timedelta(days=i), "close": price})
    return pl.DataFrame(rows)


def test_returns_exact():
    df = add_returns(make_synthetic(), 1)
    aaa = df.filter(pl.col("ticker") == "AAA").drop_nulls("ret_1d")
    # AAA compounds at exactly 1% per day
    assert aaa["ret_1d"].abs().min() == pytest.approx(0.01, abs=1e-12)
    assert aaa["ret_1d"].abs().max() == pytest.approx(0.01, abs=1e-12)
    bbb = df.filter(pl.col("ticker") == "BBB").drop_nulls("ret_1d")
    assert bbb["ret_1d"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_no_cross_ticker_leakage():
    """The first return of the second ticker must be null, not computed from
    the previous ticker's last price: the classic long-format bug."""
    df = add_returns(make_synthetic(), 1).sort(["ticker", "ts"])
    first_bbb = df.filter(pl.col("ticker") == "BBB").head(1)
    assert first_bbb["ret_1d"][0] is None


def test_momentum_12_1_construction():
    df = add_momentum_12_1(make_synthetic(400))
    aaa = df.filter(pl.col("ticker") == "AAA").drop_nulls("mom_12_1")
    # For constant 1% daily growth: close_{t-21}/close_{t-252} - 1 = 1.01^231 - 1
    expected = 1.01**231 - 1
    assert aaa["mom_12_1"][0] == pytest.approx(expected, rel=1e-9)
    # Needs 252 prior rows: first 252 rows must be null
    n_null = df.filter(pl.col("ticker") == "AAA")["mom_12_1"].null_count()
    assert n_null == 252


def test_sma_flat_series():
    df = add_sma(make_synthetic(), 50)
    bbb = df.filter(pl.col("ticker") == "BBB").drop_nulls("sma_50d")
    assert bbb["sma_50d"].min() == pytest.approx(50.0)
    assert bbb["sma_50d"].max() == pytest.approx(50.0)


def test_vol_zero_for_constant_growth():
    # Constant log-return series has zero std -> zero vol
    df = add_rolling_vol(make_synthetic(), 21)
    aaa = df.filter(pl.col("ticker") == "AAA").drop_nulls("vol_21d")
    assert aaa["vol_21d"].max() == pytest.approx(0.0, abs=1e-9)


def test_zscore_and_full_pipeline_run():
    df = build_standard_features(make_synthetic(400))
    # Pipeline produces all expected columns and doesn't explode
    for c in ["ret_1d", "ret_5d", "ret_21d", "vol_21d", "vol_63d",
              "mom_12_1", "sma_50d", "sma_200d", "z_mom_12_1_252d"]:
        assert c in df.columns
