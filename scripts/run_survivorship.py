"""Survivorship audit: the same momentum strategy run on two universes.

The biased run uses TODAY's constituent list, which is what almost every
retail backtest does. The honest run uses point-in-time membership
reconstructed from the git history of the maintained constituents CSV. The
difference between the two IS the survivorship bias, expressed in Sharpe
rather than in a disclaimer sentence.

Usage
-----
    python scripts/run_survivorship.py
    python scripts/run_survivorship.py --since 2024-09-01
    python scripts/run_survivorship.py --force      # ignore the coverage gate

Three design decisions worth being able to defend:

1. **The coverage gate refuses to print a number it cannot trust.** A dropped
   ticker whose prices were never ingested contributes nothing to the
   point-in-time run, so the audit would silently under-measure the bias and
   report a comfortingly small gap. That failure mode is invisible unless you
   look for it, so this script looks for it and exits non-zero rather than
   printing. `--force` overrides, and says so loudly in the output.

   The gate distinguishes the kinds of missing day rather than counting them,
   because they do not mean the same thing. Prices that STOP while a name is
   still an index member are a delisting: expected, unfixable, and the reason
   the gap is a lower bound. Prices that START late are the ingest window. A
   hole in the MIDDLE is a defect. Only the last kind, and a total absence of
   prices, block the run.

2. **The quality gate runs ONCE, on the combined universe, before the split.**
   Running it separately per universe could exclude different tickers from each
   run, and the measured gap would then be part survivorship and part
   data-cleaning. One gate, applied identically, leaves membership as the only
   difference between the two runs.

3. **Both runs share one `momentum_weights`.** Imported from run_momentum.py
   rather than reimplemented here, for the same reason.

LIMITATION, and it belongs in the write-up: restoring a dropped name to the
universe is not the same as capturing its delisting return. A company acquired
for cash simply stops having prices, so the final move to the takeout price is
missing. The measured gap is a LOWER BOUND on the true bias, not an estimate of
it. Shumway (1997) covers the size of the missing piece.

On the current 233-day invested window this output is a DIAGNOSTIC, not a
result. The publishable version needs the 5-year backfill at Module 6.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import polars as pl

from build_pit_universe import read_revisions, sync_repo
from run_momentum import BENCHMARK, COST_BPS, _fmt, momentum_weights

from falsify.backtest import metrics as m
from falsify.backtest.engine import BacktestConfig, run_backtest
from falsify.backtest.loader import load_panel
from falsify.data.pit_universe import members_as_of, membership_panel, parse_snapshots
from falsify.data.quality import drop_suspect_tickers
from falsify.stats.survivorship import coverage_report, survivorship_gap

DEFAULT_SINCE = dt.date(2024, 9, 1)

# A member-day with no price is not one thing. Classified against the ticker's
# own first and last priced day:
#   leading   membership starts before the price history does. The ingest
#             window, not a defect: the free Polygon tier goes back two years
#             from the day the fetch ran, and two ingests run a week apart give
#             two different windows.
#   trailing  prices stop while the name is still an index member. A real
#             delisting. Expected, and it is exactly Shumway's missing piece.
#   interior  a hole between two priced days. Nothing legitimate produces this
#             at scale: suspect a failed fetch or a ticker splice.
# Only `interior` and a total absence of prices block the run. Tolerating a few
# interior days absorbs genuine trading halts without waving through a gap.
INTERIOR_TOLERANCE_DAYS = 5
DELISTING_MIN_DAYS = 1
TOP_N = 8


def _classify_missing(membership: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    """Member-days with no price, labelled leading / trailing / interior.

    Returns a frame (ticker, kind, days, last_px). `coverage_report` counts
    missing days; this says which KIND they are, which is what decides whether
    the run is safe to trust.
    """
    mem = membership.select(["ts", "ticker"]).unique()
    px = universe.select(["ts", "ticker"]).unique()
    span = px.group_by("ticker").agg(
        pl.col("ts").min().alias("first_px"), pl.col("ts").max().alias("last_px")
    )
    gaps = mem.join(px, on=["ts", "ticker"], how="anti").join(span, on="ticker", how="left")
    if gaps.is_empty():
        return pl.DataFrame(
            schema={"ticker": pl.Utf8, "kind": pl.Utf8, "days": pl.UInt32, "last_px": pl.Date}
        )
    return (
        gaps.with_columns(
            pl.when(pl.col("first_px").is_null()).then(pl.lit("none"))
            .when(pl.col("ts") < pl.col("first_px")).then(pl.lit("leading"))
            .when(pl.col("ts") > pl.col("last_px")).then(pl.lit("trailing"))
            .otherwise(pl.lit("interior"))
            .alias("kind")
        )
        .group_by(["ticker", "kind"])
        .agg(pl.len().alias("days").cast(pl.UInt32), pl.col("last_px").first())
        .sort(["days", "ticker"], descending=[True, False])
    )


def _tickers_with(kinds: pl.DataFrame, kind: str, min_days: int) -> pl.DataFrame:
    """Rows of one kind at or above a day threshold, worst first."""
    return kinds.filter((pl.col("kind") == kind) & (pl.col("days") >= min_days))


def _run(panel: pl.DataFrame, membership: pl.DataFrame | None, label: str):
    """One backtest. Returns (daily returns frame, invested dates frame)."""
    weights = momentum_weights(panel, membership=membership)
    if weights.is_empty():
        raise SystemExit(
            f"{label}: no dates carry a 12-1 momentum score. mom_12_1 needs 252 "
            "trading days of history per ticker."
        )
    res = run_backtest(
        panel.select(["ticker", "ts", "close"]),
        weights,
        BacktestConfig(cost_bps=COST_BPS),
    )
    invested = weights.select("ts").unique()
    print(
        f"  {label:<22} {weights['ticker'].n_unique():>4} names traded, "
        f"{invested.height:>4} invested days"
    )
    return res.returns, invested


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=dt.date.fromisoformat, default=DEFAULT_SINCE,
                    help="ignore index snapshots before this date (YYYY-MM-DD)")
    ap.add_argument("--force", action="store_true",
                    help="print the gap even when price coverage is incomplete")
    args = ap.parse_args()

    panel = load_panel()
    if panel.is_empty():
        raise SystemExit("daily_bars is empty. Run scripts/run_ingest.py first.")
    universe = panel.filter(pl.col("ticker") != BENCHMARK)
    print(
        f"ingested: {universe['ticker'].n_unique()} tickers, "
        f"{universe['ts'].min()} to {universe['ts'].max()}"
    )

    # --- point-in-time membership ------------------------------------------
    snapshots = parse_snapshots(read_revisions(sync_repo(), since=args.since))
    if snapshots.is_empty():
        raise SystemExit("No usable snapshots parsed — the source format has changed.")
    dates = snapshots["as_of"].unique().sort()
    latest = snapshots["as_of"].max()
    current_members = members_as_of(snapshots, latest)
    print(
        f"snapshots: {len(dates)} from {dates[0]} to {dates[-1]}, "
        f"{snapshots['ticker'].n_unique()} distinct tickers, "
        f"{len(current_members)} current members"
    )

    membership = membership_panel(snapshots, universe["ts"].unique())

    # --- coverage gate ------------------------------------------------------
    cov = coverage_report(membership, universe)
    kinds = _classify_missing(membership, universe)
    never = cov.filter(pl.col("n_price_days") == 0)["ticker"].to_list()
    interior = _tickers_with(kinds, "interior", INTERIOR_TOLERANCE_DAYS + 1)
    trailing = _tickers_with(kinds, "trailing", DELISTING_MIN_DAYS)
    leading = _tickers_with(kinds, "leading", DELISTING_MIN_DAYS)

    print(f"\nCOVERAGE: {cov.height} point-in-time members checked")
    print(f"  fully covered:                 {cov.filter(pl.col('missing_days') == 0).height}")
    print(f"  no prices at all:              {len(never)}   <- blocks")
    print(f"  interior holes >{INTERIOR_TOLERANCE_DAYS}d:            {interior.height}   <- blocks")
    print(f"  price history ends early:      {trailing.height}   (delistings, expected)")
    print(f"  price history starts late:     {leading.height}   (ingest window, expected)")

    if not trailing.is_empty():
        print(f"\n  Names whose prices stop while still index members (top {TOP_N}):")
        for r in trailing.head(TOP_N).iter_rows(named=True):
            print(f"    {r['ticker']:<6} {r['days']:>4}d of membership unpriced after {r['last_px']}")
        print(
            "  This is Shumway's missing piece, visible in the data: an acquired\n"
            "  company stops having prices and its delisting return is unrecoverable.\n"
            "  It is why the measured gap is a LOWER BOUND, not an estimate."
        )

    blocked = bool(never) or not interior.is_empty()

    if never:
        print(f"\n  {len(never)} member(s) have no ingested prices at all:")
        print("    " + ", ".join(never[:20]) + (" ..." if len(never) > 20 else ""))
        print("  Ingest them, or the point-in-time run silently omits them and the")
        print("  gap comes back smaller than the truth:")
        print(f"    python scripts/run_ingest.py {' '.join(never[:10])}")

    if not interior.is_empty():
        print(f"\n  {interior.height} member(s) have holes INSIDE their price history:")
        for r in interior.head(TOP_N).iter_rows(named=True):
            print(f"    {r['ticker']:<6} {r['days']:>4}d missing between first and last price")
        print("  A hole in the middle is not a delisting. Suspect a failed fetch or a")
        print("  ticker splice, and check data/quality.py before trusting this run.")

    if blocked:
        if not args.force:
            raise SystemExit("\nRefusing to report a gap on incomplete coverage. --force overrides.")
        print("\n  --force: reporting anyway. THIS NUMBER UNDER-MEASURES THE BIAS.")

    # --- one quality gate, applied to both runs -----------------------------
    universe, excluded = drop_suspect_tickers(universe)
    if not excluded.is_empty():
        print(f"\nquality gate excluded {excluded.height} ticker(s) from BOTH runs:")
        for r in excluded.iter_rows(named=True):
            print(f"  {r['ticker']:<6} {r['reason']:<14} {r['detail']}")

    biased_panel = universe.filter(pl.col("ticker").is_in(list(current_members)))

    # --- the two runs -------------------------------------------------------
    print("\nRUNS")
    ret_biased, inv_biased = _run(biased_panel, None, "current constituents")
    ret_pit, inv_pit = _run(universe, membership, "point-in-time")

    common = inv_biased.join(inv_pit, on="ts", how="inner").sort("ts")
    if common.is_empty():
        raise SystemExit("The two runs share no invested days; nothing to compare.")
    print(
        f"  common invested window: {common.height} days, "
        f"{common['ts'].min()} to {common['ts'].max()}"
    )

    b = ret_biased.join(common, on="ts", how="semi").sort("ts")
    p = ret_pit.join(common, on="ts", how="semi").sort("ts")

    print("\nCURRENT CONSTITUENTS (the biased run)")
    print(_fmt(m.summary(b["ret"])))
    print("\nPOINT-IN-TIME MEMBERSHIP (the honest run)")
    print(_fmt(m.summary(p["ret"])))

    # --- the gap ------------------------------------------------------------
    gap = survivorship_gap(b["ret"], p["ret"])
    print(f"\n{'SURVIVORSHIP GAP (current - point-in-time)':<44}")
    print(f"  {'metric':<16}{'current':>12}{'PIT':>12}{'gap':>12}")
    for name, pct in (
        ("total_return", True), ("cagr", True), ("ann_vol", True),
        ("sharpe", False), ("max_drawdown", True),
    ):
        c, pv, g = gap[f"{name}_current"], gap[f"{name}_pit"], gap[f"{name}_gap"]
        fmt = "{:>11.2%}" if pct else "{:>11.2f} "
        print(f"  {name:<16}" + fmt.format(c) + fmt.format(pv) + fmt.format(g))

    sharpe_gap = gap["sharpe_gap"]
    print(
        f"\n  Reading it: the biased run's Sharpe is inflated by {sharpe_gap:+.2f} "
        f"purely by\n  not being able to see the companies that left the index."
        if sharpe_gap > 0 else
        f"\n  The gap is NEGATIVE ({sharpe_gap:+.2f}). Not a bug to suppress: it says the\n"
        "  dropped names were ones momentum was correctly short in this sample.\n"
        "  That is a finding about the sample, not a refutation of the bias."
    )
    print(
        "\n  LOWER BOUND. A cash acquisition stops having prices, so the delisting\n"
        "  return is missing from the point-in-time run too. The true bias is larger.\n"
        "  DIAGNOSTIC ONLY on this window — publishable after the M6 backfill."
    )


if __name__ == "__main__":
    main()
