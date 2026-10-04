"""What a quant checks after the headline Sharpe, computed by code.

A single long/short Sharpe answers one question: did the extremes differ. A
reader who knows the field asks four more before believing it, and every one of
them is answered here, by the pipeline, so the model can quote them but never
produce them:

  1. BUCKET RETURNS. Is the effect a staircase across the whole cross-section,
     or one odd bucket at an end? A real ranking effect is roughly monotonic.
     A spread driven by the bottom decile alone is a different (and weaker)
     claim. Reported low signal to high signal, with the Spearman correlation
     between bucket number and return as one number for the shape.
  2. TWO HALVES. Does the result hold in both halves of the sample, or did one
     regime carry it? A Sharpe that is all first half is a period effect.
  3. COST SENSITIVITY. The Sharpe at 0, 10 and 25 bps one-way. An effect that
     dies between 10 and 25 bps is not tradeable at realistic cost.
  4. THE EQUITY CURVE. The path, downsampled, so a reader sees one jump versus
     a steady climb. A chart, never a model-visible number.

Everything here is a pure function of frames the backtest already built. No
timing rule is restated: forward returns come from engine.forward_returns, the
one place the codebase looks forward, and bucket membership is decided on the
same rebalance dates as the strategy's weights and held until the next one.
"""
from __future__ import annotations

import math
from typing import Any

import polars as pl

from falsify.backtest import metrics as m
from falsify.backtest.engine import forward_returns

COST_GRID_BPS: tuple[int, ...] = (0, 10, 25)
EQUITY_POINTS = 160


def bucket_returns(
    prices: pl.DataFrame,
    signal: pl.DataFrame,
    rebalance_dates: pl.Series,
    held_dates: pl.Series,
    n_buckets: int,
) -> list[dict[str, Any]]:
    """Equal-weight return of every bucket, bucket 1 = LOWEST signal.

    Args:
        prices: (ticker, ts, close).
        signal: (ts, ticker, sig), already restricted to investable names.
        rebalance_dates: the strategy's rebalance dates.
        held_dates: the dates the strategy held positions (the invested window).
        n_buckets: as in the backtest.

    Returns:
        One dict per bucket: bucket, ann_return (arithmetic mean daily return x
        252, GROSS of costs), n_days. Empty if no date had enough names.

    The whole cross-section is partitioned, so the extreme buckets here are
    close to, but not identical with, the strategy's legs: decile_weights takes
    n // n_buckets names from each end, and this assigns every name.
    """
    s = signal.join(pl.DataFrame({"ts": rebalance_dates}), on="ts", how="semi")
    s = s.with_columns(
        pl.col("sig").rank("ordinal").over("ts").cast(pl.Int64).alias("rk"),
        pl.len().over("ts").cast(pl.Int64).alias("n"),
    ).filter(pl.col("n") >= n_buckets)
    if s.is_empty():
        return []
    s = s.with_columns(
        (((pl.col("rk") - 1) * n_buckets) // pl.col("n") + 1).alias("bucket")
    ).select(pl.col("ts").alias("period"), "ticker", "bucket")

    # Each held date belongs to the most recent rebalance on or before it, the
    # same backward rule hold_until_next_rebalance applies to the weights.
    periods = pl.DataFrame({"ts": held_dates.unique().sort()}).join_asof(
        pl.DataFrame({"period": s["period"].unique().sort()}).with_columns(
            pl.col("period").alias("ts")
        ),
        on="ts",
        strategy="backward",
    ).drop_nulls("period")

    fwd = forward_returns(prices.select(["ticker", "ts", "close"])).drop_nulls("fwd_ret")
    daily = (
        fwd.join(periods, on="ts", how="inner")
        .join(s, on=["period", "ticker"], how="inner")
        .group_by(["ts", "bucket"])
        .agg(pl.col("fwd_ret").mean().alias("r"))
    )
    out = (
        daily.group_by("bucket")
        .agg(
            (pl.col("r").mean() * m.TRADING_DAYS).alias("ann_return"),
            pl.len().alias("n_days"),
        )
        .sort("bucket")
    )
    return [
        {"bucket": int(r["bucket"]), "ann_return": float(r["ann_return"]), "n_days": int(r["n_days"])}
        for r in out.iter_rows(named=True)
    ]


def monotonicity(buckets: list[dict[str, Any]]) -> float | None:
    """Spearman correlation of bucket number with bucket return, in [-1, 1].

    +1 is a perfect staircase rising with the signal, -1 a perfect staircase
    falling with it, near 0 no shape at all. None below three buckets, where a
    rank correlation carries no information about shape.
    """
    if len(buckets) < 3:
        return None
    x = pl.Series([float(b["bucket"]) for b in buckets]).rank()
    y = pl.Series([b["ann_return"] for b in buckets]).rank()
    c = pl.DataFrame({"x": x, "y": y}).select(pl.corr("x", "y")).item()
    return None if c is None or math.isnan(c) else float(c)


def halves(invested: pl.DataFrame) -> list[dict[str, Any]]:
    """Sharpe of the first and second half of the invested window, by count."""
    n = len(invested)
    if n < 4:
        return []
    out = []
    for part in (invested.head(n // 2), invested.tail(n - n // 2)):
        out.append(
            {
                "first_date": str(part["ts"].min()),
                "last_date": str(part["ts"].max()),
                "sharpe": m.sharpe(part["ret"]),
                "n_days": len(part),
            }
        )
    return out


def cost_sensitivity(invested: pl.DataFrame, turnover: pl.DataFrame) -> list[dict[str, Any]]:
    """Sharpe and CAGR of the same weights at each cost in COST_GRID_BPS.

    Recomputed from the gross return and the turnover, so it is the strategy
    actually run, repriced, not a rescaled net figure.
    """
    g = invested.select(["ts", "gross_ret"]).join(turnover, on="ts", how="left").with_columns(
        pl.col("turnover").fill_null(0.0)
    )
    out = []
    for bps in COST_GRID_BPS:
        r = g["gross_ret"] - g["turnover"] * bps / 10_000.0
        out.append({"cost_bps": bps, "sharpe": m.sharpe(r), "cagr": m.cagr(r)})
    return out


def equity_curve(invested: pl.DataFrame, points: int = EQUITY_POINTS) -> list[dict[str, Any]]:
    """Growth of 1 over the invested window, downsampled to about `points`.

    Starts at 1.0 on the day before the first return, keeps the last day, and
    takes evenly spaced days between. A chart input, never shown to the model.
    """
    if invested.is_empty():
        return []
    eq = (invested["ret"] + 1.0).cum_prod()
    ts = invested["ts"]
    n = len(eq)
    step = max(1, math.ceil(n / points))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [{"ts": str(ts[0]), "equity": 1.0}] + [
        {"ts": str(ts[i]), "equity": round(float(eq[i]), 5)} for i in idx
    ]


def _r(x: float | None, places: int) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), places)


def compute(
    prices: pl.DataFrame,
    raw_signal: pl.DataFrame,
    rebalance_dates: pl.Series,
    invested: pl.DataFrame,
    turnover: pl.DataFrame,
    n_buckets: int,
) -> dict[str, Any]:
    """Every diagnostic, full precision where it matters. Stored with the run.

    `raw_signal` must be the UN-negated signal, so bucket 1 is always the
    lowest feature value whichever leg the strategy held.
    """
    b = bucket_returns(prices, raw_signal, rebalance_dates, invested["ts"], n_buckets)
    return {
        "buckets": [
            {**x, "ann_return": _r(x["ann_return"], 6)} for x in b
        ],
        "monotonicity": _r(monotonicity(b), 4),
        "halves": [{**h, "sharpe": _r(h["sharpe"], 4)} for h in halves(invested)],
        "costs": [
            {**c, "sharpe": _r(c["sharpe"], 4), "cagr": _r(c["cagr"], 6)}
            for c in cost_sensitivity(invested, turnover)
        ],
        "equity": equity_curve(invested),
    }


def model_view(d: dict[str, Any]) -> dict[str, Any]:
    """The compact digest the model reads: rounded, no curve, well under 1 KB.

    Rounded here, at the boundary, so the figures the note quotes are already
    at reading precision and the provenance check still traces each one.
    """
    return {
        # A dict keyed by bucket number, not a list: no tool result carries a
        # sequence (rule 3 in agent/tools.py), and twenty aggregates keyed by
        # name read the same to the model without opening that door.
        "bucket_ann_return": {str(x["bucket"]): _r(x["ann_return"], 4) for x in d["buckets"]},
        "bucket_note": (
            "Gross annualised mean return of each signal bucket, 1 = LOWEST "
            "feature value. A real ranking effect is roughly a staircase."
        ),
        "bucket_monotonicity_spearman": _r(d["monotonicity"], 2),
        "sharpe_first_half": _r(d["halves"][0]["sharpe"], 2) if d["halves"] else None,
        "sharpe_second_half": _r(d["halves"][1]["sharpe"], 2) if d["halves"] else None,
        "split_date": d["halves"][1]["first_date"] if d["halves"] else None,
        "sharpe_at_cost_bps": {str(c["cost_bps"]): _r(c["sharpe"], 2) for c in d["costs"]},
    }
