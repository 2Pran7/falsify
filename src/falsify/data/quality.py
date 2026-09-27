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

TWO GATES, AND ONLY ONE OF THEM IS POINT-IN-TIME.

`drop_suspect_tickers` (Modules 2-3) evaluates the whole panel at once and
removes an offending ticker from EVERY date, so a splice first detectable in
2026 also removes that ticker from a 2025 backtest. That is future information
shaping the universe: the same class of bias `stats/survivorship.py` exists to
measure, in the tool written to prevent a different one. It is kept only so
the Module 3 figures can be reproduced exactly (`--whole-sample-gate`).

`truncate_suspect_tickers` (Module 7) is the fix. A ticker keeps every row up
to and including the last observation before its FIRST defect, and loses
everything from the bar on which that defect becomes visible. Nothing that
happens after a date can change what the gate keeps on that date, and that is
stated as a test rather than asserted here: gating a panel cut at any date T
gives exactly the full-panel gate cut at T (`test_truncation_is_point_in_time`).

What truncation gives up, disclosed: the rows AFTER a splice may be a genuine
successor security (the ticker's new owner), and they are dropped rather than
re-keyed as a new instrument. That costs sample, not correctness: the bias runs
toward a smaller universe, and it cannot reach a return, because the return
across the join is the one that is never formed.

This is the gate `agent/tools.fetch_data` applies, so the agent and the eval
suite see the same cleaned panel. Until Module 7 they saw no gate at all.
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


def truncate_suspect_tickers(
    panel: pl.DataFrame,
    max_gap_days: int = MAX_GAP_DAYS,
    max_ratio: float = MAX_PRICE_RATIO,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The point-in-time gate: cut each suspect ticker at its first defect.

    Args:
        panel: long price frame with at least ticker, ts, close.
        max_gap_days, max_ratio: as for `flag_suspect_tickers`.

    Returns:
        (clean, report). `clean` is the panel with, for each suspect ticker,
        every row AFTER its first defect removed. `report` has one row per
        truncated ticker: ticker, reason, detail, last_kept (the last
        observation before the defect), rows_dropped.

    WHY "AFTER THE EVENT" AND NOT "FROM THE EVENT". Both detectors report `ts`
    as the last observation BEFORE the hole or the jump. That row is genuine:
    it is the old security's last close. The defect is only visible on the NEXT
    row, when the ticker reappears after the hole or prints at 4x. So the cut is
    strictly after `last_kept`, and the one return that spans the join is never
    formed.

    WHY THE FIRST DEFECT. Once a ticker has changed hands once, nothing after
    that point can be trusted to be the security the earlier history belongs
    to, so a second defect changes nothing about what is kept.
    """
    events = pl.concat(
        [
            level_shifts(panel, max_ratio).select(
                "ticker",
                "ts",
                pl.lit("level_shift").alias("reason"),
                pl.format(
                    "{} -> {} ({}x) after {}",
                    pl.col("close").round(2),
                    pl.col("next_close").round(2),
                    pl.col("ratio").round(1),
                    pl.col("ts"),
                ).alias("detail"),
            ),
            coverage_gaps(panel, max_gap_days).select(
                "ticker",
                "ts",
                pl.lit("coverage_gap").alias("reason"),
                pl.format("{} day hole after {}", pl.col("gap_days"), pl.col("ts")).alias(
                    "detail"
                ),
            ),
        ]
    )
    report_schema = {
        "ticker": pl.Utf8,
        "reason": pl.Utf8,
        "detail": pl.Utf8,
        "last_kept": panel.schema["ts"],
        "rows_dropped": pl.UInt32,
    }
    if events.is_empty():
        return panel, pl.DataFrame(schema=report_schema)

    first = (
        events.sort(["ticker", "ts", "reason"])
        .group_by("ticker", maintain_order=True)
        .first()
        .rename({"ts": "last_kept"})
    )
    joined = panel.join(first.select("ticker", "last_kept"), on="ticker", how="left")
    keep = pl.col("last_kept").is_null() | (pl.col("ts") <= pl.col("last_kept"))
    clean = joined.filter(keep).select(panel.columns)
    dropped = (
        joined.filter(~keep)
        .group_by("ticker")
        .agg(pl.len().cast(pl.UInt32).alias("rows_dropped"))
    )
    report = (
        first.join(dropped, on="ticker", how="left")
        .with_columns(pl.col("rows_dropped").fill_null(0))
        .select(list(report_schema))
        .sort("ticker")
    )
    return clean, report
