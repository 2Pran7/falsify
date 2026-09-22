"""Ingest prices for the names that LEFT the index. The other half of the audit.

Usage
-----
    python scripts/ingest_dropped.py --dry-run
    python scripts/ingest_dropped.py --years 5
    python scripts/ingest_dropped.py --years 5 --since 2020-01-01

WHY THIS SCRIPT EXISTS. `run_ingest.py` calls `fetch_sp500_tickers()`, which
returns TODAY's constituents, and ingests those. That is correct for what it
does and it is exactly half of what the survivorship audit needs. The other
half is the names that were in the index during the sample and are not in it
now — 326 of them across the reconstructed history — and nothing in the project
ever fetched their prices.

The consequence is the failure mode `stats/survivorship.py` warns about in its
own docstring, arriving through the back door. A point-in-time member with no
ingested price contributes NOTHING to the point-in-time leg: it is silently
absent from the cross-section on every date it should have been in it. The
audit then compares today's constituents against a thinned-out version of
today's constituents and reports a comfortingly small gap.

**So the measured survivorship bias is bounded below by price coverage, not
just by delisting returns.** Both bounds run the same way — toward
under-measurement — and only this one is fixable.

WHAT IT STILL CANNOT FIX. A company acquired for cash simply stops having
prices, and no API returns them. Restoring a dropped name to the universe is
not the same as capturing its delisting return; Shumway (1997) covers the size
of the missing piece. This closes the gap between "we never asked for the
prices" and "the prices do not exist", which are very different admissions.

COST. Polygon's free tier is 5 requests per minute and two years of history, so
326 tickers is roughly 65 minutes and still only reaches back two years. On the
Starter tier, set POLYGON_RPM in .env to something like 300 and the same job
takes a few minutes over five years. Ingest is idempotent per
(ticker, from_date, to_date), so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

sys.path.insert(0, "src")

import polars as pl

from falsify.backtest.loader import coverage, load_snapshots
from falsify.data.ingest import ingest_universe
from falsify.data.pit_universe import dropped_tickers


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--years", type=int, default=5,
                    help="years of history to request per ticker (default 5)")
    ap.add_argument("--since", type=dt.date.fromisoformat, default=None,
                    help="only names that were members on or after this date. Without "
                         "it, every name ever dropped since the first snapshot is "
                         "included, which reaches back to 2012 and is almost certainly "
                         "more than the price window needs.")
    ap.add_argument("--index", default="SP500")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be fetched and exit. Run this first.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of tickers, for a timed trial run")
    args = ap.parse_args()

    snapshots = load_snapshots(args.index)
    if snapshots.is_empty():
        print("universe_snapshot is empty. Run scripts/build_pit_universe.py first.",
              file=sys.stderr)
        return 2

    n_snap = snapshots["as_of"].n_unique()
    print(f"snapshots: {n_snap} dated, {snapshots['as_of'].min()} to {snapshots['as_of'].max()}")

    dropped = dropped_tickers(snapshots, since=args.since)
    if not dropped:
        print("no dropped tickers in this window; nothing to ingest.")
        return 0

    # Skip anything already priced. The ingest is idempotent anyway, but a list
    # of 326 tickers that is really a list of 40 is a misleading thing to print
    # before a job measured in minutes.
    have = coverage()
    priced = set() if have.is_empty() else set(have["ticker"].to_list())
    todo = [t for t in dropped if t not in priced]
    partial = [t for t in dropped if t in priced]

    print(f"dropped and never priced: {len(todo)}")
    print(f"dropped but already have some prices: {len(partial)}")
    if partial:
        print("  (these are re-requested only if the date range differs; ingest_log "
              "is keyed on (ticker, from_date, to_date))")

    if args.limit:
        todo = todo[: args.limit]
        print(f"limited to {len(todo)} for this run")

    print(f"\nwould fetch {len(todo)} tickers x {args.years}y:")
    print("  " + ", ".join(todo[:20]) + (" ..." if len(todo) > 20 else ""))

    from falsify.config import settings
    rpm = settings.polygon_rpm
    minutes = len(todo) / rpm if rpm else float("inf")
    print(f"\nPOLYGON_RPM={rpm} -> about {minutes:.0f} minute(s) for {len(todo)} calls")
    if rpm <= 5:
        print("  NOTE: 5 rpm is the FREE tier, which also caps history at ~2 years.")
        print("  On Starter, raise POLYGON_RPM in .env before running this.")

    if args.dry_run:
        print("\n--dry-run: nothing fetched.")
        return 0

    ingest_universe(todo, years=args.years)

    # Report what actually landed, rather than trusting the loop's own count.
    after = coverage()
    got = 0 if after.is_empty() else len(
        after.filter(pl.col("ticker").is_in(todo))
    )
    print(f"\n{got} of {len(todo)} requested tickers now have prices.")
    if got < len(todo):
        print("  The remainder returned no bars. For a name acquired for cash that is "
              "the correct answer and not a failure: the prices do not exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
