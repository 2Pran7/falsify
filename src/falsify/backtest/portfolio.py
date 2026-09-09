"""Signal -> weights. The strategy layer.

The engine knows nothing about ranking, deciles or rebalancing. That all lives
here, which is why adding a new anomaly in Module 6 means writing a new signal
and reusing everything else untouched.

Input everywhere is a long frame (ts, ticker, sig). Output everywhere is a long
frame (ts, ticker, w) in exactly the shape run_backtest expects.

Methodology note: nothing here shifts anything in time. Signals passed in
must already be computed from data up to and including ts (which is what
features/library.py guarantees). The engine does the shift. One place, once.
"""
from __future__ import annotations

import warnings

import polars as pl

W_SCHEMA = {"ts": pl.Date, "ticker": pl.Utf8, "w": pl.Float64}


def decile_weights(
    signal: pl.DataFrame,
    n_buckets: int = 10,
    long_short: bool = True,
) -> pl.DataFrame:
    """Rank each date's cross-section and take the extremes.

    Args:
        signal:     long frame (ts, ticker, sig).
        n_buckets:  10 = deciles, 5 = quintiles.
        long_short: True  -> long the top bucket, short the bottom, +1/k and
                             -1/k each. Weights sum to 0, gross exposure 2.0.
                    False -> long the top bucket only, +1/k each, sums to 1.0.

    Returns:
        Long frame (ts, ticker, w). Dates and tickers with a null signal simply
        do not appear.

    Null handling is the thing to get right here, not the ranking. mom_12_1
    needs 252 days of history, so every ticker is null for its first trading
    year. If nulls silently ranked as zero they would pile into the bottom
    bucket, producing a short book of names too young to score,
    which is a real portfolio that produces plausible-looking numbers and means
    nothing. So nulls are dropped BEFORE ranking, and the bucket size is
    computed from how many names actually scored that day.
    """
    s = signal.drop_nulls("sig")
    if s.is_empty():
        return pl.DataFrame(schema=W_SCHEMA)

    s = s.with_columns(
        pl.col("sig").rank("ordinal", descending=True).over("ts").alias("rk"),
        pl.len().over("ts").cast(pl.Int64).alias("n"),
    )
    # Bucket size, at least 1 so a thin cross-section degrades instead of
    # producing an empty portfolio.
    s = s.with_columns(
        pl.max_horizontal(
            pl.lit(1, dtype=pl.Int64), (pl.col("n") // n_buckets).cast(pl.Int64)
        ).alias("k")
    )

    if long_short:
        # Need at least two names, otherwise the same ticker is both the top
        # and the bottom of its own cross-section.
        s = s.filter(pl.col("n") >= 2)
        s = s.with_columns(
            pl.when(pl.col("rk") <= pl.col("k"))
            .then(1.0 / pl.col("k"))
            .when(pl.col("rk") > pl.col("n") - pl.col("k"))
            .then(-1.0 / pl.col("k"))
            .otherwise(None)
            .alias("w")
        )
    else:
        s = s.with_columns(
            pl.when(pl.col("rk") <= pl.col("k"))
            .then(1.0 / pl.col("k"))
            .otherwise(None)
            .alias("w")
        )

    return (
        s.drop_nulls("w")
        .select(["ts", "ticker", "w"])
        .cast(W_SCHEMA)
        .sort(["ts", "ticker"])
    )


def fixed_weights(dates: pl.Series, ticker: str, w: float = 1.0) -> pl.DataFrame:
    """Hold one ticker at a constant weight on every given date.

    This is what the SPY buy-and-hold smoke test uses. Trivial, but it belongs
    here rather than being hand-rolled inside a test.
    """
    ds = dates.unique().sort()
    return pl.DataFrame(
        {"ts": ds, "ticker": [ticker] * len(ds), "w": [float(w)] * len(ds)}
    ).cast(W_SCHEMA)


def month_end_dates(dates: pl.Series) -> pl.Series:
    """The last TRADING day of each month present in `dates`.

    Not the calendar month-end: 31 August might be a Sunday. Rebalancing on a
    date the market was shut is a classic way to quietly lose a day of returns.
    """
    df = pl.DataFrame({"ts": dates.unique().sort()})
    return (
        df.with_columns(pl.col("ts").dt.truncate("1mo").alias("m"))
        .group_by("m")
        .agg(pl.col("ts").max())
        .sort("ts")["ts"]
    )


def hold_until_next_rebalance(
    weights: pl.DataFrame, all_dates: pl.Series
) -> pl.DataFrame:
    """Carry each rebalance's weights forward until the next one.

    Monthly momentum picks its names once a month, but the engine wants a
    weight for every trading day. On any date, the held weights are
    whatever the most recent rebalance on or before that date decided.

    Uses an as-of join, which is the same primitive exchanges use to match a
    quote to the trade that followed it. Crucially it looks BACKWARD only, so
    it can never carry a future rebalance's decision into the past.
    """
    if weights.is_empty():
        return pl.DataFrame(schema=W_SCHEMA)

    tickers = weights["ticker"].unique().sort()
    spine = (
        pl.DataFrame({"ts": all_dates.unique().sort()})
        .join(pl.DataFrame({"ticker": tickers}), how="cross")
        .sort(["ticker", "ts"])
    )

    with warnings.catch_warnings():
        # Polars cannot verify sortedness when `by` groups are used, and warns.
        # Both frames ARE sorted by (ticker, ts) two lines above, so the warning
        # is noise. Suppressed narrowly rather than globally.
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        out = spine.join_asof(
            weights.sort(["ticker", "ts"]),
            on="ts",
            by="ticker",
            strategy="backward",
        )

    return (
        out.drop_nulls("w")
        .filter(pl.col("w") != 0.0)
        .select(["ts", "ticker", "w"])
        .cast(W_SCHEMA)
        .sort(["ts", "ticker"])
    )
