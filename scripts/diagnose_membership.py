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
    #
    # A GAP ONLY COSTS SOMETHING WHERE THERE ARE PRICES. The snapshot history
    # now reaches back to 2012 while the price panel covers two years, so most
    # gaps sit entirely outside the sample and affect nothing. The first version
    # of this script reported the largest gap anywhere and called it "inside the
    # price panel", which was false and was about to be used to make a decision.
    #
    # So gaps are split in two: those that OVERLAP the panel, which are
    # affecting results now, and those that do not, which are a forecast of what
    # a longer price window will run into.
    print("\n=== CAUSE B: INTERIOR SNAPSHOT GAPS ===")
    all_gaps = (
        pl.DataFrame({"as_of": snap_dates})
        .with_columns(pl.col("as_of").shift(1).alias("prev"))
        .drop_nulls("prev")
        .with_columns((pl.col("as_of") - pl.col("prev")).dt.total_days().alias("gap_days"))
        .filter(pl.col("gap_days") > 45)
        .sort("gap_days", descending=True)
    )
    first_day, last_day = dates.min(), dates.max()
    # Overlap test on the OPEN interval the gap spans: (prev, as_of].
    inside = all_gaps.filter(
        (pl.col("as_of") > first_day) & (pl.col("prev") < last_day)
    )
    outside = all_gaps.filter(
        ~((pl.col("as_of") > first_day) & (pl.col("prev") < last_day))
    )

    print(f"  price panel spans {first_day} to {last_day}")
    print(f"  gaps over 45 days, anywhere in the snapshot history: {len(all_gaps)}")
    print(f"\n  AFFECTING THIS PANEL ({len(inside)}):")
    if inside.is_empty():
        print("    none. Every snapshot gap sits outside the priced window.")
    else:
        print(inside.select(["prev", "as_of", "gap_days"]))

    print(f"\n  OUTSIDE THIS PANEL ({len(outside)}) — a forecast, not a current defect:")
    if outside.is_empty():
        print("    none.")
    else:
        print(outside.select(["prev", "as_of", "gap_days"]).head(8))
        print("    These cost nothing today and every one of them moves INSIDE the")
        print("    sample the moment the price window is lengthened. Check this list")
        print("    against the window you are about to buy.")

    # MEMBERSHIP RESOLUTION: the number to disclose when a gap cannot be closed.
    # A backward as-of join dates every removal to the next snapshot, so the
    # error on any membership date is bounded by the local snapshot spacing.
    # Quoting that bound is honest; claiming daily point-in-time membership from
    # a source committed every few weeks is not.
    in_window = snap_dates.filter(
        (snap_dates >= first_day) & (snap_dates <= last_day)
    )
    if len(in_window) >= 2:
        spacing = (
            pl.DataFrame({"as_of": in_window})
            .with_columns(
                (pl.col("as_of") - pl.col("as_of").shift(1)).dt.total_days().alias("d")
            )
            .drop_nulls("d")["d"]
        )
        print(f"\n  MEMBERSHIP RESOLUTION over the priced window:")
        print(f"    {len(in_window)} snapshots · median spacing {spacing.median():.0f} days"
              f" · worst {spacing.max()} days")
        print(f"    Membership is accurate to within the LOCAL spacing, so a removal is")
        print(f"    dated up to {spacing.max()} days late in the worst case. That is the")
        print("    number to put in the write-up, not a claim of daily accuracy.")

    gaps = inside  # only overlapping gaps drive the verdict

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
    print("\n=== CAUSE B, CONTINUED: MEMBER-DAY CLUSTERS ===")
    full = len(covered)
    clusters = (
        md.filter(pl.col("n_member_days") < full)
        .group_by("n_member_days").len().rename({"len": "n_tickers"})
        .sort("n_tickers", descending=True)
    )
    if clusters.is_empty():
        print("  no ticker is short of the full panel; nothing to explain.")
    else:
        # A cluster is a group of tickers whose membership all ends on the same
        # day. There are TWO reasons that happens and they are opposites:
        #
        #   ENDS AT THE PANEL'S LAST DATE -> these are still members. They JOINED
        #   late, and their short count is their tenure, not a removal. The first
        #   version of this script reported them as removals and alarmed about
        #   the gap before the panel's final snapshot, which is nonsense: nobody
        #   left.
        #
        #   ENDS BEFORE IT -> a genuine removal, dated to the snapshot that
        #   noticed. Only these inherit the snapshot gap's error.
        #
        # And they are ranked by the SIZE OF THE GAP that created them, not by
        # how many tickers they contain. A 13-day lag on a monthly rebalance is
        # immaterial; a 204-day one is the finding. Ranking by count put the
        # harmless cluster first and attached 204-day language to it. A checker
        # that cries wolf gets switched off.
        last_day = dates.max()
        rows = []
        for n_days in clusters["n_member_days"].to_list():
            names = md.filter(pl.col("n_member_days") == n_days)["ticker"].to_list()
            ends = (
                membership.filter(pl.col("ticker").is_in(names))
                .group_by("ticker").agg(pl.col("ts").max().alias("last"))["last"]
                .unique()
            )
            if len(ends) != 1:
                continue
            end = ends[0]
            after = snap_dates.filter(snap_dates > end)
            before = snap_dates.filter(snap_dates <= end)
            gap = (
                int((after.min() - before.max()).days)
                if len(after) and len(before) else None
            )
            rows.append({
                "n_member_days": int(n_days), "n_tickers": len(names),
                "ends": end, "kind": "joined late" if end == last_day else "removed",
                "gap_days": gap, "names": names,
            })

        joined = [r for r in rows if r["kind"] == "joined late"]
        removed = sorted(
            [r for r in rows if r["kind"] == "removed"],
            key=lambda r: r["gap_days"] or 0, reverse=True,
        )

        if joined:
            n = sum(r["n_tickers"] for r in joined)
            print(f"\n  STILL MEMBERS, joined during the window: {n} ticker(s) in "
                  f"{len(joined)} cluster(s).")
            print("    Their short member-day count is tenure, not a removal. Not a defect.")
            for r in joined[:3]:
                print(f"      {r['n_tickers']:>3} tickers, {r['n_member_days']} days: "
                      f"{', '.join(r['names'][:8])}{' ...' if len(r['names']) > 8 else ''}")

        if not removed:
            print("\n  No removal cluster in this window.")
        else:
            print(f"\n  REMOVAL CLUSTERS, worst gap first ({len(removed)}):")
            for r in removed[:5]:
                print(f"    {r['n_tickers']:>3} tickers ended {r['ends']} "
                      f"after a {r['gap_days']}-day snapshot gap")
            worst = removed[0]
            g = worst["gap_days"] or 0
            print(f"\n    Worst: {worst['n_tickers']} tickers ending {worst['ends']} — "
                  f"{', '.join(worst['names'][:8])}"
                  f"{' ...' if len(worst['names']) > 8 else ''}")
            if g > 45:
                print(
                    f"\n    MATERIAL. Every removal between the previous snapshot and\n"
                    f"    {worst['ends']} is dated to the snapshot that closed a {g}-day gap,\n"
                    f"    so those names were carried as index MEMBERS for up to {g} days\n"
                    "    after they left. Index removals are disproportionately fallers, so\n"
                    "    over that window the point-in-time leg is stale membership rather\n"
                    "    than point-in-time, and the survivorship gap inherits the error."
                )
            else:
                print(
                    f"\n    IMMATERIAL. A {g}-day lag is within normal snapshot spacing and\n"
                    "    well inside a monthly rebalance, so no trade is affected. Recorded\n"
                    "    for completeness, not as a defect."
                )

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
        when = gaps["as_of"][0]
        print(f"\nVERDICT: CAUSE B. The largest snapshot gap OVERLAPPING the priced")
        print(f"  window is {worst} days, closing {when}. Every index removal inside it")
        print("  is dated to the snapshot that closed it, so the point-in-time leg holds")
        print("  those names for up to that long after they left the index.")
        print()
        print("  IF scripts/build_pit_universe.py HAS ALREADY BEEN RUN and this gap is")
        print("  still here, the commits do not exist: the gap is in the SOURCE, not the")
        print("  ingest, and it cannot be closed for free. It is then a LIMIT ON")
        print("  MEMBERSHIP RESOLUTION, to be disclosed and bounded rather than fixed.")
        print("  The honest statement is that membership is accurate to within the local")
        print("  snapshot spacing, and that spacing is printed above.")
        verdict = 1
    elif max_member_days >= len(covered) - 2:
        print("\nVERDICT: CAUSE A, working as designed. The shortfall is the pre-snapshot")
        print("  window and nothing else. Ingest from the first snapshot date, or accept")
        print("  the shorter PIT window and state it.")
    else:
        print("\nVERDICT: UNEXPLAINED. The shortfall is larger than the pre-snapshot window")
        print("  and no snapshot gap overlaps the sample. Do not backfill until this is")
        print("  understood: it would move an unexplained deficit inside the sample.")
        verdict = 1
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
