"""Data-quality gates for the price panel.

Vendor price series are keyed by ticker string, not a permanent security id.
A delisted, merged or renamed listing frees its ticker for reassignment to an
entirely different company, and a naive series then splices two securities end
to end: a fictitious several-hundred-percent return and, worse, a momentum
score computed across the join that stays wrong for a full year.

The checks are deliberately structural, not statistical. A screen on "unusually
large returns" cannot work: real equities move 30-40% on earnings and over 100%
on takeover news, so any threshold low enough to catch a splice also discards
genuine events, which biases a momentum backtest in a flattering direction.

Two signatures identify a splice without touching the return distribution:

  coverage gap  a real listing trades every session; a multi-month hole means
                the ticker was dormant and the rows either side are different
                securities.
  level shift   a single-session price ratio beyond about 4x. Splits and
                dividends are already adjusted at ingest, and no continuously
                listed equity moves that far in one session.

LIMITATION, disclosed rather than hidden: THIS GATE IS NOT POINT-IN-TIME.
`drop_suspect_tickers` evaluates the whole panel at once and removes an
offending ticker from every date in it, so a splice that only becomes
detectable in 2026 also removes that ticker from a 2025 backtest. That is
future information shaping the universe, which is the same class of bias
`stats/survivorship.py` exists to measure.

It is kept for now because the alternative is worse: a fictitious price formed
by splicing two different companies produces a momentum score that stays wrong
for a full year. The bias runs toward a cleaner universe rather than a more
profitable one, so it does not flatter returns the way a return-magnitude
screen would. The correct fix is a point-in-time variant that excludes a ticker
only from the date its defect becomes detectable, keeping its earlier, genuine
history in the sample. Scheduled for Module 6, where the window lengthens and
the number of affected names grows.
"""
from __future__ import annotations

import polars as pl

# Longest plausible market closure, in calendar days. Weekends are 3, holiday
# weekends 4, and the longest modern unscheduled closure (Sandy, 9/11) was 6.
MAX_GAP_DAYS = 10

# Single-session price ratio treated as impossible. The largest genuine
# one-day moves on record are roughly 3x; splits arrive pre-adjusted.
MAX_PRICE_RATIO = 4.0


def coverage_gaps(panel: pl.DataFrame, max_gap_days: int = MAX_GAP_DAYS) -> pl.DataFrame:
    """Tickers with a hole between consecutive observations.

    Returns (ticker, ts, gap_days) for each gap, where `ts` is the last
    observation before the hole.
    """
    return (
        panel.sort(["ticker", "ts"])
        .with_columns(
            (pl.col("ts").shift(-1).over("ticker") - pl.col("ts")).dt.total_days().alias("gap_days")
        )
        .filter(pl.col("gap_days") > max_gap_days)
        .select(["ticker", "ts", "gap_days"])
        .sort("gap_days", descending=True)
    )


def level_shifts(panel: pl.DataFrame, max_ratio: float = MAX_PRICE_RATIO) -> pl.DataFrame:
    """Consecutive closes whose ratio is outside [1/max_ratio, max_ratio].

    Returns (ticker, ts, close, next_close, ratio) where `ts` is the last
    observation before the shift.
    """
    return (
        panel.sort(["ticker", "ts"])
        .with_columns(pl.col("close").shift(-1).over("ticker").alias("next_close"))
        .drop_nulls("next_close")
        .with_columns((pl.col("next_close") / pl.col("close")).alias("ratio"))
        .filter((pl.col("ratio") > max_ratio) | (pl.col("ratio") < 1.0 / max_ratio))
        .select(["ticker", "ts", "close", "next_close", "ratio"])
        .sort("ratio", descending=True)
    )


def flag_suspect_tickers(
    panel: pl.DataFrame,
    max_gap_days: int = MAX_GAP_DAYS,
    max_ratio: float = MAX_PRICE_RATIO,
) -> pl.DataFrame:
    """Tickers whose series is not a single continuous security.

    Returns (ticker, reason, detail), one row per ticker, reporting the first
    reason found. Callers exclude these from the universe and disclose the
    exclusion rather than silently dropping them.
    """
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for r in level_shifts(panel, max_ratio).iter_rows(named=True):
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        rows.append(
            (
                r["ticker"],
                "level_shift",
                f"{r['close']:.2f} -> {r['next_close']:.2f} ({r['ratio']:.1f}x) at {r['ts']}",
            )
        )

    for r in coverage_gaps(panel, max_gap_days).iter_rows(named=True):
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        rows.append(
            (r["ticker"], "coverage_gap", f"{r['gap_days']} day hole after {r['ts']}")
        )

    schema = {"ticker": pl.Utf8, "reason": pl.Utf8, "detail": pl.Utf8}
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row").sort("ticker")


def drop_suspect_tickers(panel: pl.DataFrame, **kwargs) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Panel with suspect tickers removed, alongside the exclusion report."""
    flagged = flag_suspect_tickers(panel, **kwargs)
    if flagged.is_empty():
        return panel, flagged
    clean = panel.filter(~pl.col("ticker").is_in(flagged["ticker"].to_list()))
    return clean, flagged
