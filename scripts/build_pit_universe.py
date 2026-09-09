"""Reconstruct point-in-time S&P 500 membership and load it into universe_snapshot.

Walks the git history of the maintained constituents CSV that `universe.py`
already depends on. Every commit touching the file is a dated snapshot of index
membership; ~190 of them reach back to 2012. See `data/pit_universe.py` for the
reasoning and the limitations.

Usage
-----
    python scripts/build_pit_universe.py                  # reconstruct + load
    python scripts/build_pit_universe.py --dry-run        # report, write nothing
    python scripts/build_pit_universe.py --since 2021-09-01
    python scripts/build_pit_universe.py --dropped-out dropped.txt

`--dropped-out` writes the tickers that were members during the window and are
absent today: exactly the names the current-constituents backtest cannot see,
and the ingest list that makes the survivorship audit possible.

The clone is cached under `.cache/` and re-pulled on later runs, so a rebuild
costs one fetch rather than one clone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import subprocess
import sys

sys.path.insert(0, "src")

import polars as pl
import psycopg

from falsify.config import settings
from falsify.data.pit_universe import dropped_tickers, parse_snapshots

REPO_URL = "https://github.com/datasets/s-and-p-500-companies.git"
CSV_PATH = "data/constituents.csv"
CACHE = pathlib.Path(".cache/s-and-p-500-companies")


def _git(*args: str, cwd: pathlib.Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def sync_repo(cache: pathlib.Path = CACHE) -> pathlib.Path:
    """Clone the dataset repo, or fetch into an existing clone."""
    if (cache / ".git").exists():
        print(f"Updating {cache} ...")
        _git("fetch", "--quiet", "origin", cwd=cache)
        _git("reset", "--quiet", "--hard", "origin/HEAD", cwd=cache)
    else:
        print(f"Cloning {REPO_URL} -> {cache} ...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", REPO_URL, str(cache))
    return cache


def read_revisions(
    cache: pathlib.Path, since: dt.date | None = None
) -> list[tuple[dt.date, str]]:
    """Every revision of the constituents file, as (commit date, CSV text).

    The commit date is used as `as_of`, not the author date: it is the moment
    the change entered the maintained file, which is the closest available
    proxy for when the information was publicly known.
    """
    log = _git(
        "log", "--format=%H %cd", "--date=short", "--", CSV_PATH, cwd=cache
    ).strip().splitlines()

    revisions: list[tuple[dt.date, str]] = []
    for line in log:
        sha, date_str = line.split()
        as_of = dt.date.fromisoformat(date_str)
        if since and as_of < since:
            continue
        try:
            text = _git("show", f"{sha}:{CSV_PATH}", cwd=cache)
        except subprocess.CalledProcessError:
            print(f"  skipping {sha[:8]} ({as_of}): file not readable at that commit")
            continue
        revisions.append((as_of, text))
    return revisions


def write_snapshots(snapshots, dsn: str | None = None) -> int:
    """Insert into universe_snapshot. Idempotent via the composite primary key."""
    rows = snapshots.select(["ticker", "index_name", "as_of"]).rows()
    with psycopg.connect(dsn or settings.db_dsn) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO universe_snapshot (ticker, index_name, as_of) "
                "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                rows,
            )
        conn.commit()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=dt.date.fromisoformat, default=None,
                    help="ignore commits before this date (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the reconstruction, write nothing to the database")
    ap.add_argument("--dropped-out", type=pathlib.Path, default=None,
                    help="write the dropped-ticker ingest list to this file")
    args = ap.parse_args()

    cache = sync_repo()
    revisions = read_revisions(cache, since=args.since)
    print(f"Found {len(revisions)} revisions of {CSV_PATH}")

    snapshots = parse_snapshots(revisions)
    if snapshots.is_empty():
        raise SystemExit("No usable snapshots parsed — the source format has changed.")

    dates = snapshots["as_of"].unique().sort()
    print(f"Parsed {len(dates)} usable snapshots: {dates[0]} -> {dates[-1]}")
    print(f"Distinct tickers ever seen: {snapshots['ticker'].n_unique()}")

    dropped = dropped_tickers(snapshots, since=args.since)
    latest = snapshots["as_of"].max()
    current = snapshots.filter(pl.col("as_of") == latest)["ticker"].n_unique()
    print(f"Members in the latest snapshot: {current}")
    print(f"Members during the window, absent today: {len(dropped)}")
    print(f"  -> survivorship exposure: {len(dropped) / max(current, 1):.1%} of the universe")
    if dropped:
        print(f"  first 15: {', '.join(dropped[:15])}")

    if args.dropped_out:
        args.dropped_out.write_text("\n".join(dropped) + "\n")
        print(f"Wrote {len(dropped)} tickers to {args.dropped_out}")
        print(f"  ingest them with: python scripts/run_ingest.py $(cat {args.dropped_out})")

    if args.dry_run:
        print("\n--dry-run: nothing written to the database.")
        return

    n = write_snapshots(snapshots)
    print(f"\nWrote {n} rows to universe_snapshot (duplicates ignored).")


if __name__ == "__main__":
    main()
