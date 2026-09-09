"""Signal -> weights. The strategy layer.

Ranking, deciles and rebalancing all live here, so a new anomaly means a new
signal and the rest of the pipeline unchanged. Input is a long frame
(ts, ticker, sig); output is (ts, ticker, w) in the shape run_backtest expects.

Nothing here shifts anything in time. Signals are already computed from data up
to and including ts; the engine applies the shift, in one place, once.
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

    Null handling matters more here than the ranking. mom_12_1 needs 252 days
    of history, so every ticker is null for its first trading year. Nulls
    ranked as zero would collect in the bottom bucket and short names that are
    merely too young to score: a real portfolio with plausible-looking returns
    and no meaning. Nulls are dropped BEFORE ranking, and bucket size comes
    from the names that actually scored that date.
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
        # Below two names the same ticker is both top and bottom of its own
        # cross-section.
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

    Used by the buy-and-hold reconciliation test.
    """
    ds = dates.unique().sort()
    return pl.DataFrame(
        {"ts": ds, "ticker": [ticker] * len(ds), "w": [float(w)] * len(ds)}
    ).cast(W_SCHEMA)


def month_end_dates(dates: pl.Series) -> pl.Series:
    """The last TRADING day of each month present in `dates`.

    Not the calendar month-end: 31 August may fall on a Sunday. Rebalancing on
    a date the market was closed silently loses a day of returns.
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

    Monthly momentum selects names once a month; the engine needs a weight for
    every trading day. On any date the held weights are those set by the most
    recent rebalance on or before it.

    Each rebalance is a COMPLETE portfolio snapshot: a name absent from one gets
    an explicit 0.0 that date, so dropping out of the selection closes the
    position. Carrying weights forward per ticker instead would hold exited
    names indefinitely and let gross exposure accumulate at every rebalance.

    Implemented as a backward as-of join, so a future rebalance can never be
    carried into the past.
    """
    if weights.is_empty():
        return pl.DataFrame(schema=W_SCHEMA)

    tickers = pl.DataFrame({"ticker": weights["ticker"].unique().sort()})

    # Complete every rebalance date into a full snapshot across all tickers.
    weights = (
        pl.DataFrame({"ts": weights["ts"].unique().sort()})
        .join(tickers, how="cross")
        .join(weights, on=["ts", "ticker"], how="left")
        .with_columns(pl.col("w").fill_null(0.0))
    )

    spine = (
        pl.DataFrame({"ts": all_dates.unique().sort()})
        .join(tickers, how="cross")
        .sort(["ticker", "ts"])
    )

    with warnings.catch_warnings():
        # Polars cannot verify sortedness when `by` groups are used, and warns.
        # Both frames ARE sorted by (ticker, ts), so the warning is noise.
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
