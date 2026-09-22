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


def add_reversal_36_12(df: pl.DataFrame, col: str = "close") -> pl.DataFrame:
    """De Bondt-Thaler long-term reversal: the 3-year return SKIPPING the last year.

    (close_{t-252} / close_{t-756}) - 1, appended as `rev_36_12`.

    THE SKIP IS THE WHOLE CONSTRUCTION, for the mirror of the reason
    `add_momentum_12_1` skips a month. Over a 36-month formation window without
    the skip, the most recent year carries momentum (past winners keep winning)
    while the earlier two carry reversal (past winners revert). They run
    OPPOSITE ways, so the sign of the combined signal says nothing about either
    effect, and a backtest on it would be a measurement of their net balance in
    this particular sample.

    Needs 756 trading days per ticker — three years — before it produces a
    single value. On a two-year panel it is null everywhere, which is why the
    eval suite scores it `insufficient_data` rather than `fail`.
    """
    return _sorted(df).with_columns(
        (pl.col(col).shift(252).over(OVER) / pl.col(col).shift(756).over(OVER) - 1)
        .alias("rev_36_12")
    )


def add_pct_52w_high(df: pl.DataFrame, window: int = 252, col: str = "close") -> pl.DataFrame:
    """George-Hwang 52-week high: today's close as a fraction of its rolling max.

    close_t / max(close over the trailing `window` days), appended as
    `pct_{window}d_high`. A name at its 52-week high scores 1.0; one that has
    halved from it scores 0.5. Bounded in (0, 1], which makes it the only
    feature here with a scale a reader can interpret without a distribution.

    NOTE THE COLUMN NAME. The menu key in agent/tools.py is `pct_52w_high` and
    the column this appends is `pct_252d_high`. That mismatch is deliberate —
    the window is a parameter, and naming the column after the weeks would lie
    for any window but 252 — and it is exactly the assumption `compute_feature`
    used to make. `_FEATURE_COLUMNS` now maps the two explicitly.

    The rolling max includes t itself, which is correct: whether a stock is at
    its high TODAY is knowable today. Nothing here looks forward.
    """
    return _sorted(df).with_columns(
        (pl.col(col) / pl.col(col).rolling_max(window).over(OVER))
        .alias(f"pct_{window}d_high")
    )


def add_idio_vol(
    df: pl.DataFrame,
    window: int = 63,
    market: str = "SPY",
) -> pl.DataFrame:
    """Ang-Hodrick-Xing-Zhang idiosyncratic volatility, appended as `ivol_{window}d`.

    Annualised standard deviation of the residual from regressing each ticker's
    daily log return on the market's, over a rolling window, WITH NO INTERCEPT:

        beta_t = sum(r*m) / sum(m*m)        over the trailing window
        SSE_t  = sum(r*r) - beta_t * sum(r*m)
        ivol_t = sqrt(SSE_t / (window - 1)) * sqrt(252)

    Three decisions, each of which a plausible alternative gets wrong:

    1. NO INTERCEPT. The published residual comes from a three-factor
       regression; those factors are not in this database, and a stated
       simplification beats an invented factor series. Over 63 days the
       estimated alpha is indistinguishable from zero, and subtracting a noisy
       estimate of zero ADDS variance to the exact quantity being measured.

    2. IF THE MARKET SERIES IS ABSENT, THE FEATURE IS NULL EVERYWHERE. It does
       not fall back to total volatility. A fallback would score the
       low-volatility anomaly a second time under a different name and let the
       suite count it as independent evidence, which is the worst failure an
       eval suite can have: two rows that look like two findings.

    3. The market row is joined on `ts`, so a ticker with a trading day the
       market series lacks contributes a null rather than a zero market return.

    The market ticker is EXCLUDED from its own output (its residual against
    itself is identically zero, and a zero-volatility name sorts to the top of
    a low-vol ranking forever).
    """
    name = f"ivol_{window}d"
    frame = _sorted(df)

    if "logret_1d" not in frame.columns:
        frame = add_log_returns(frame)

    mkt = frame.filter(pl.col(OVER) == market).select(
        ["ts", pl.col("logret_1d").alias("_m")]
    )
    # Absent market series -> null everywhere. Deliberately not a fallback.
    if mkt.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))

    out = (
        frame.join(mkt.unique(subset=["ts"]), on="ts", how="left")
        .sort([OVER, "ts"])
        .with_columns(
            (pl.col("logret_1d") * pl.col("_m")).alias("_rm"),
            (pl.col("_m") * pl.col("_m")).alias("_mm"),
            (pl.col("logret_1d") * pl.col("logret_1d")).alias("_rr"),
        )
        .with_columns(
            pl.col("_rm").rolling_sum(window).over(OVER).alias("_srm"),
            pl.col("_mm").rolling_sum(window).over(OVER).alias("_smm"),
            pl.col("_rr").rolling_sum(window).over(OVER).alias("_srr"),
        )
        .with_columns(
            # Guard the divide: a window in which the market never moved leaves
            # beta undefined, and a silent 0.0 beta would report TOTAL vol here.
            pl.when(pl.col("_smm") > 0)
            .then(pl.col("_srm") / pl.col("_smm"))
            .otherwise(None)
            .alias("_beta")
        )
        .with_columns(
            (pl.col("_srr") - pl.col("_beta") * pl.col("_srm")).alias("_sse")
        )
        .with_columns(
            pl.when((pl.col("_sse") >= 0) & (pl.col(OVER) != market))
            .then((pl.col("_sse") / (window - 1)).sqrt() * (252**0.5))
            .otherwise(None)
            .alias(name)
        )
        .drop(["_m", "_rm", "_mm", "_rr", "_srm", "_smm", "_srr", "_beta", "_sse"])
    )
    return out


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
