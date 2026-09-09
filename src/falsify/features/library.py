"""Feature library.

Every function takes a long-format Polars frame (ticker, ts, close, ...) sorted
by (ticker, ts) and appends ONE column. Chainable. Windows are TRADING DAYS.

METHODOLOGY RULE: features at time t use data up to and INCLUDING t, never
after. Lagging for trade use is the caller's job (the backtester does it), so
lookahead bias lives in exactly one place.
"""
from __future__ import annotations

import polars as pl

OVER = "ticker"  # every rolling op is per-ticker


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort([OVER, "ts"])


def add_returns(df: pl.DataFrame, periods: int = 1, col: str = "close") -> pl.DataFrame:
    """Simple return over `periods` trading days: close_t / close_{t-p} - 1."""
    name = f"ret_{periods}d"
    return _sorted(df).with_columns(
        (pl.col(col) / pl.col(col).shift(periods).over(OVER) - 1).alias(name)
    )


def add_log_returns(df: pl.DataFrame, col: str = "close") -> pl.DataFrame:
    return _sorted(df).with_columns(
        (pl.col(col) / pl.col(col).shift(1).over(OVER)).log().alias("logret_1d")
    )


def add_rolling_vol(df: pl.DataFrame, window: int = 21) -> pl.DataFrame:
    """Annualised rolling volatility of daily log returns (sqrt(252) scaling)."""
    if "logret_1d" not in df.columns:
        df = add_log_returns(df)
    name = f"vol_{window}d"
    return df.with_columns(
        (pl.col("logret_1d").rolling_std(window).over(OVER) * (252**0.5)).alias(name)
    )


def add_momentum_12_1(df: pl.DataFrame, col: str = "close") -> pl.DataFrame:
    """Jegadeesh-Titman momentum: 12-month return SKIPPING the most recent month.

    (close_{t-21} / close_{t-252}) - 1. The 1-month skip keeps short-term
    reversal out of the signal: the published construction, not a convenience.
    """
    return _sorted(df).with_columns(
        (pl.col(col).shift(21).over(OVER) / pl.col(col).shift(252).over(OVER) - 1)
        .alias("mom_12_1")
    )


def add_sma(df: pl.DataFrame, window: int, col: str = "close") -> pl.DataFrame:
    return _sorted(df).with_columns(
        pl.col(col).rolling_mean(window).over(OVER).alias(f"sma_{window}d")
    )


def add_zscore(df: pl.DataFrame, source_col: str, window: int = 252) -> pl.DataFrame:
    """Rolling z-score of any existing column, per ticker."""
    mean = pl.col(source_col).rolling_mean(window).over(OVER)
    std = pl.col(source_col).rolling_std(window).over(OVER)
    return _sorted(df).with_columns(
        ((pl.col(source_col) - mean) / std).alias(f"z_{source_col}_{window}d")
    )


def build_standard_features(df: pl.DataFrame) -> pl.DataFrame:
    """The default feature set, chained."""
    df = add_returns(df, 1)
    df = add_returns(df, 5)
    df = add_returns(df, 21)
    df = add_log_returns(df)
    df = add_rolling_vol(df, 21)
    df = add_rolling_vol(df, 63)
    df = add_momentum_12_1(df)
    df = add_sma(df, 50)
    df = add_sma(df, 200)
    df = add_zscore(df, "mom_12_1", 252)
    return df
