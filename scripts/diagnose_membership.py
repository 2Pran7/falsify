"""What the point-in-time membership panel is actually made of.

Usage
-----
    python scripts/diagnose_membership.py
    python scripts/diagnose_membership.py --ticker WBA

WRITTEN to explain an apparent "374 member-days against a ~500-day panel", and
THE FIRST RUN (22 Sep 2026) SHOWED THAT PREMISE WAS FALSE. The median ticker
has the full 507, the maximum is 507, and there are 44 distinct counts. 374 is
not a universal shortfall -- it is a CLUSTER, and a cluster of identical counts
has one plausible cause: a group of tickers whose membership all ends on the
same day, because a single snapshot after a long gap noticed every one of their
removals at once.

That is a worse finding than a uniform shortfall, and it is already inside the
sample rather than waiting at its edge. A name removed from the index during a
snapshot gap is carried as a MEMBER for the whole gap. Index removals are
disproportionately fallers. Over that window the point-in-time leg is not
point-in-time; it is stale membership, and the survivorship gap measured
against it inherits the error.

THREE CANDIDATE CAUSES, and they call for three different responses, which is
why this script distinguishes them rather than counting the missing days:

  A. THE BACKWARD AS-OF JOIN WORKING AS DESIGNED. `membership_panel` gives each
     date the latest snapshot at or before it, and a date earlier than every
     snapshot inherits nothing -- deliberately, because falling back to the
     earliest snapshot would import future membership into the past, which is
     the exact bias the module exists to remove. If the missing days are all
     BEFORE the first snapshot, there is no defect. Fix: ingest prices starting
     no earlier than the first snapshot, or accept a shorter PIT window and say
     so.

  B. A GENUINE SNAPSHOT GAP: a stretch with no snapshot within it. Harmless for
     a monthly rebalance if short, and a real defect if long, because every
     removal inside it is dated to the snapshot that closed it. Fix: re-run
     build_pit_universe.py and establish whether the gap is in the SOURCE (the
     maintainers committed nothing) or in the INGEST (commits exist and were
     not walked). Only the second is fixable, and it is fixable for free.

  C. run_ingest.py's SINGLE SNAPSHOT MASQUERADING AS POINT-IN-TIME. One
     snapshot dated the day the ingest ran means membership never changes, so
     the PIT leg reproduces the current-constituents result exactly, with full
     coverage and a survivorship audit that measures ZERO. This is the failure
     mode that FAILS BY LOOKING HEALTHY. `tools.fetch_data` already refuses it;
     this reports it.

The script prints which one it is, identifies the cluster and the snapshot that
created it, and exits non-zero for B or C.
"""
from __future__ import annotations

import argparse
import sys

import polars as pl

from falsify.backtest.loader import load_panel, load_snapshots
from falsify.data.pit_universe import membership_panel
from falsify.stats.survivorship import coverage_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=None, help="drill into one ticker's timeline")
    ap.add_argument("--index", default="SP500")
    args = ap.parse_args()

    snapshots = load_snapshots(args.index)
    panel = load_panel()

    if panel.is_empty():
        print("daily_bars is empty. Run scripts/run_ingest.py first.", file=sys.stderr)
        return 2
    if snapshots.is_empty():
        print("universe_snapshot is empty. Run scripts/build_pit_universe.py.",
              file=sys.stderr)
        return 2

    dates = panel["ts"].unique().sort()
    snap_dates = snapshots["as_of"].unique().sort()
    n_snap = len(snap_dates)

    print("\n=== PANEL AND SNAPSHOTS ===")
    print(f"price panel:  {len(dates)} trading days, {dates.min()} to {dates.max()}")
    print(f"snapshots:    {n_snap} dated, {snap_dates.min()} to {snap_dates.max()}")

    # --- cause C ---------------------------------------------------------
    if n_snap < 2:
        print("\nCAUSE C: A SINGLE SNAPSHOT MASQUERADING AS POINT-IN-TIME.")
        print(
            f"  universe_snapshot holds {n_snap} distinct as_of date(s), which is what\n"
            "  run_ingest.py leaves behind on its own. Membership that never changes IS\n"
            "  today's membership: the point-in-time leg would reproduce the\n"
            "  current-constituents result exactly, with full coverage, no gap, and a\n"
            "  survivorship audit that measures ZERO. It fails by looking healthy.\n"
            "  Fix: python scripts/build_pit_universe.py"
        )
        return 1

    membership = membership_panel(snapshots, dates)
    cov = coverage_report(membership, panel)

    print("\n=== MEMBER-DAY DISTRIBUTION ===")
    md = membership.group_by("ticker").len().rename({"len": "n_member_days"})
    print(
        md.select(
            pl.col("n_member_days").min().alias("min"),
            pl.col("n_member_days").median().alias("median"),
            pl.col("n_member_days").max().alias("max"),
            pl.col("n_member_days").n_unique().alias("distinct_values"),
            pl.len().alias("n_tickers"),
        )
    )

    # --- cause A: days before the first snapshot -------------------------
    first_snap = snap_dates.min()
    before = dates.filter(dates < first_snap)
    covered = dates.filter(dates >= first_snap)

    print("\n=== CAUSE A: DATES BEFORE THE FIRST SNAPSHOT ===")
    print(f"  trading days before {first_snap}: {len(before)}")
    print(f"  trading days on or after it:      {len(covered)}")
    if len(before):
        print(
            "  These inherit NO membership, by design: a backward as-of join that fell\n"
            "  back to the earliest snapshot would import future membership into the\n"
            "  past. If the shortfall equals this number, the join is working and the\n"
            "  only decision is whether to ingest prices from the first snapshot date."
        )
    max_member_days = int(md["n_member_days"].max())
    print(f"\n  busiest ticker has {max_member_days} member-days against "
          f"{len(covered)} covered trading days "
          f"({len(covered) - max_member_days} short)")

    # --- cause B: interior snapshot gaps ---------------------------------
    print("\n=== CAUSE B: INTERIOR SNAPSHOT GAPS ===")
    gaps = (
        pl.DataFrame({"as_of": snap_dates})
        .with_columns((pl.col("as_of") - pl.col("as_of").shift(1)).alias("gap"))
        .drop_nulls("gap")
        .with_columns(pl.col("gap").dt.total_days().alias("gap_days"))
        .filter(pl.col("gap_days") > 45)
        .sort("gap_days", descending=True)
    )
    if gaps.is_empty():
        print("  no gap between consecutive snapshots exceeds 45 days.")
    else:
        print(f"  {len(gaps)} gap(s) over 45 days between consecutive snapshots:")
        print(gaps.select(["as_of", "gap_days"]).head(10))

    # --- CAUSE B, CONTINUED: what the member-day CLUSTER actually is -----
    #
    # The premise this script was written to investigate -- "every ticker shows
    # 374 member-days" -- turned out to be false: the median is the full panel
    # length and there are dozens of distinct values. 374 is a CLUSTER, and a
    # cluster of identical counts has exactly one plausible cause: a group of
    # tickers whose membership all ends on the same day, because one snapshot
    # after a long gap noticed all of their removals at once.
    #
    # That matters far more than a uniform shortfall would. A name removed from
    # the index during the gap is carried as a MEMBER for the whole gap, so the
    # point-in-time leg holds it long after it actually left -- and index
    # removals are disproportionately fallers. The PIT universe is not
    # point-in-time over that window; it is stale-membership, and the
    # survivorship gap measured against it inherits the error.
    print("\n=== CAUSE B, CONTINUED: THE MEMBER-DAY CLUSTER ===")
    full = len(covered)
    clusters = (
        md.filter(pl.col("n_member_days") < full)
        .group_by("n_member_days").len().rename({"len": "n_tickers"})
        .sort("n_tickers", descending=True)
    )
    if clusters.is_empty():
        print("  no ticker is short of the full panel; nothing to explain.")
    else:
        print(f"  full panel is {full} trading days. Largest short clusters:")
        print(clusters.head(5))
        top = int(clusters["n_member_days"][0])
        names = (
            md.filter(pl.col("n_member_days") == top)
            .sort("ticker")["ticker"].to_list()
        )
        print(f"\n  {len(names)} tickers share exactly {top} member-days: "
              f"{', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")

        last_days = (
            membership.filter(pl.col("ticker").is_in(names))
            .group_by("ticker").agg(pl.col("ts").max().alias("last"))
        )
        distinct_last = last_days["last"].unique().sort()
        print(f"  their membership ends on {len(distinct_last)} distinct date(s): "
              f"{', '.join(str(d) for d in distinct_last.to_list()[:5])}")

        if len(distinct_last) == 1:
            ends = distinct_last[0]
            after = snap_dates.filter(snap_dates > ends)
            print(f"\n  CONFIRMED: all {len(names)} end on {ends}, the last trading day")
            if len(after):
                nxt = after.min()
                before = snap_dates.filter(snap_dates <= ends)
                prev = before.max() if len(before) else None
                gap = (nxt - prev).days if prev else None
                print(f"  before the snapshot of {nxt}.")
                print(f"  The previous snapshot was {prev}, a gap of {gap} days.")
                print(
                    f"\n  SO: every removal that happened between {prev} and {nxt} was\n"
                    f"  recorded as happening on {nxt}. Those names were carried as index\n"
                    f"  MEMBERS for up to {gap} days after they actually left. Index removals\n"
                    "  are disproportionately fallers, so over that window the\n"
                    "  point-in-time leg is not point-in-time -- it is stale membership, and\n"
                    "  the survivorship gap measured against it inherits the error.\n"
                    "\n  THE GAP IS ALREADY INSIDE THE SAMPLE. It is not a pre-snapshot\n"
                    "  window that costs nothing; it sits in the middle of the panel and it\n"
                    "  is affecting the headline number NOW."
                )
                print(
                    f"\n  Fix, in order: re-run scripts/build_pit_universe.py and see whether\n"
                    f"  the {gap}-day gap is in the SOURCE (the maintainers committed nothing)\n"
                    "  or in the INGEST (commits exist and were not walked). Only the second\n"
                    "  is fixable, and it is fixable for free."
                )
        else:
            print("\n  NOT a single-date cluster. The identical counts are a coincidence of")
            print("  length rather than a common removal date; investigate individually.")

    print("\n=== PRICE COVERAGE OF POINT-IN-TIME MEMBERS ===")
    print("  (a member-day with no ingested price contributes nothing to the PIT leg,")
    print("   so the audit would silently UNDER-measure the bias)")
    missing = cov.filter(pl.col("missing_days") > 0)
    print(f"  tickers with at least one missing member-day: {len(missing)} of {len(cov)}")
    print(missing.head(15))

    if args.ticker:
        t = args.ticker.upper()
        print(f"\n=== {t} TIMELINE ===")
        mine = membership.filter(pl.col("ticker") == t)["ts"]
        prices = panel.filter(pl.col("ticker") == t)["ts"]
        if mine.is_empty():
            print(f"  {t} is not a point-in-time member on any date in this panel.")
        else:
            print(f"  member-days: {len(mine)}, {mine.min()} to {mine.max()}")
            print(f"  price-days:  {len(prices)}, "
                  f"{prices.min() if len(prices) else '-'} to "
                  f"{prices.max() if len(prices) else '-'}")
            inside = mine.filter((mine >= prices.min()) & (mine <= prices.max())) \
                if len(prices) else mine.head(0)
            holes = len(inside) - len(
                pl.Series(sorted(set(inside.to_list()) & set(prices.to_list())))
            )
            print(f"  INTERIOR holes (member, priced window, no price): {holes}")

    verdict = 0
    if not gaps.is_empty():
        worst = int(gaps["gap_days"][0])
        print(f"\nVERDICT: CAUSE B. The largest interior snapshot gap is {worst} days, and")
        print("  it is INSIDE the price panel, not at its edge. Every index removal in that")
        print("  window is dated to the snapshot that closed it, so the point-in-time leg")
        print("  holds those names for up to that long after they left the index.")
        print("  Backfill the missing commits BEFORE lengthening the price window, and do")
        print("  not re-quote the survivorship gap until this is closed.")
        verdict = 1
    elif max_member_days >= len(covered) - 2:
        print("\nVERDICT: CAUSE A, working as designed. The shortfall is the pre-snapshot")
        print("  window and nothing else. Ingest from the first snapshot date, or accept")
        print("  the shorter PIT window and state it.")
    else:
        print("\nVERDICT: UNEXPLAINED. The shortfall is larger than the pre-snapshot window")
        print("  and there are no interior snapshot gaps. Do not backfill until this is")
        print("  understood: it would move an unexplained deficit inside the sample.")
        verdict = 1
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
