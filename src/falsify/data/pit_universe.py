"""Point-in-time S&P 500 membership, reconstructed from version-control history.

`universe.py` fetches the CURRENT constituent list, so a backtest on it is
survivorship-biased: every company that was in the index during the sample and
has since been dropped, acquired or delisted is silently absent, and those are
disproportionately the losers.

Point-in-time membership is normally a paid dataset. It is not here. The
maintained `datasets/s-and-p-500-companies` CSV that `universe.py` already
depends on lives in git, and every commit touching `data/constituents.csv` is a
dated snapshot of membership. Walking the history yields ~190 snapshots back to
2012 for the cost of a clone.

Limitations, disclosed rather than buried:
  1. as_of is the COMMIT date, not the effective date of the index change.
     Maintainers update within days, which is immaterial for a monthly
     rebalance and would not be for a daily one.
  2. Coverage is only as good as the maintainers'.
  3. This recovers WHO was in the index, not the delisting return a name earned
     on its way out. Prices for dropped tickers must be ingested separately,
     and a cash acquisition simply stops having prices.
  4. Ticker strings are not permanent identifiers, so `data/quality.py` runs
     downstream of this.

No git commands here: `parse_snapshots` takes (date, csv_text) pairs, so the
module is testable with no network, no clone and no database.
`scripts/build_pit_universe.py` does the git walk.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import warnings

import polars as pl

SNAPSHOT_SCHEMA = {"ticker": pl.Utf8, "index_name": pl.Utf8, "as_of": pl.Date}

# A plausible US equity ticker after normalisation: 1-5 capitals, optionally a
# single class suffix (BRK.B). Anything else is a parse artifact, not a ticker.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

# Below this, the file was a stub, a partial write or a failed parse rather
# than a real constituent list. Such a snapshot must be dropped, not trusted:
# a short snapshot silently narrows the point-in-time universe and would look
# like a mass exit from the index.
MIN_CONSTITUENTS = 400


def normalise_ticker(raw: str) -> str | None:
    """Vendor ticker string -> the form used in `daily_bars`, or None if junk.

    Three quirks seen in this source's history:
        BRK-B                     hyphenated class share; Polygon uses a dot,
                                  so the panel would never join.
        RVTY (Previously PKI)     rename annotated in the symbol cell.
        American Airlines Group   column-shift artifact, a NAME in the symbol
                                  column. Rejected by the pattern.

    Returns None rather than raising: one malformed row in one commit should
    cost that row, not the whole snapshot.
    """
    s = (raw or "").strip().upper()
    if not s:
        return None
    s = s.split("(")[0].strip()   # drop "(Previously PKI)" style annotations
    s = s.replace("-", ".")       # BRK-B -> BRK.B
    return s if _TICKER_RE.match(s) else None


def parse_constituents(csv_text: str) -> set[str]:
    """Ticker set from one revision of the constituents CSV.

    The header has changed shape three times (`Security` / `Name` / `Company`
    for the second column) but the symbol column has always been `Symbol`, so
    keying on the name survives the reshuffles. Positional indexing would not.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames or "Symbol" not in reader.fieldnames:
        return set()
    out = set()
    for row in reader:
        t = normalise_ticker(row.get("Symbol", ""))
        if t:
            out.add(t)
    return out


def parse_snapshots(
    revisions: list[tuple[dt.date, str]],
    index_name: str = "SP500",
    min_constituents: int = MIN_CONSTITUENTS,
) -> pl.DataFrame:
    """(commit date, CSV text) pairs -> long frame (ticker, index_name, as_of).

    Args:
        revisions: one entry per commit touching the file, any order.
        min_constituents: revisions yielding fewer tickers are discarded as
            malformed — a truncated commit would otherwise read as a mass exit.

    Returns:
        Long frame in `universe_snapshot` shape, one row per (ticker, as_of).
        Same-day commits collapse, since the table is keyed by
        (ticker, index_name, as_of).
    """
    by_date: dict[dt.date, set[str]] = {}
    for as_of, text in revisions:
        tickers = parse_constituents(text)
        if len(tickers) >= min_constituents:
            by_date[as_of] = tickers

    if not by_date:
        return pl.DataFrame(schema=SNAPSHOT_SCHEMA)

    rows = [(t, index_name, d) for d, ts in by_date.items() for t in ts]
    return (
        pl.DataFrame(rows, schema=SNAPSHOT_SCHEMA, orient="row")
        .sort(["as_of", "ticker"])
    )


def members_as_of(snapshots: pl.DataFrame, date: dt.date) -> set[str]:
    """Membership in force on `date`: the latest snapshot at or before it.

    Backward-only. A date earlier than every snapshot returns the EMPTY set,
    not the earliest one: falling back would import future membership into the
    past, which is the bias this module exists to remove.
    """
    if snapshots.is_empty():
        return set()
    prior = snapshots.filter(pl.col("as_of") <= date)
    if prior.is_empty():
        return set()
    latest = prior["as_of"].max()
    return set(prior.filter(pl.col("as_of") == latest)["ticker"].to_list())


def membership_panel(snapshots: pl.DataFrame, dates: pl.Series) -> pl.DataFrame:
    """Expand snapshots to one row per (ts, ticker) that was a member on ts.

    Args:
        snapshots: frame from `parse_snapshots`.
        dates: the trading days to expand over.

    Returns:
        Frame (ts, ticker) of every membership in force on every date. Join a
        signal against this to restrict it to the point-in-time universe.

    A backward as-of join, the same primitive as
    `portfolio.hold_until_next_rebalance`: each day inherits the latest
    snapshot at or before it, and a day before the first inherits nothing.
    """
    if snapshots.is_empty():
        return pl.DataFrame(schema={"ts": pl.Date, "ticker": pl.Utf8})

    universe = pl.DataFrame({"ticker": snapshots["ticker"].unique().sort()})
    snaps = (
        pl.DataFrame({"as_of": snapshots["as_of"].unique().sort()})
        .join(universe, how="cross")
        .join(
            snapshots.select(["as_of", "ticker"]).with_columns(
                pl.lit(True).alias("member")
            ),
            on=["as_of", "ticker"],
            how="left",
        )
        # Absent from a snapshot means explicitly NOT a member that day, which
        # is what makes an exit visible rather than merely missing.
        .with_columns(pl.col("member").fill_null(False))
        .rename({"as_of": "ts"})
        .sort(["ticker", "ts"])
    )

    spine = (
        pl.DataFrame({"ts": dates.unique().sort()})
        .join(universe, how="cross")
        .sort(["ticker", "ts"])
    )

    with warnings.catch_warnings():
        # Both frames ARE sorted by (ticker, ts) above; Polars just cannot
        # verify that once `by` groups are used. Suppressed narrowly.
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        joined = spine.join_asof(snaps, on="ts", by="ticker", strategy="backward")

    return (
        joined
        .filter(pl.col("member"))          # null (pre-first-snapshot) drops too
        .select(["ts", "ticker"])
        .sort(["ts", "ticker"])
    )


def dropped_tickers(snapshots: pl.DataFrame, since: dt.date | None = None) -> list[str]:
    """Members at some point, absent from the latest snapshot.

    Precisely the names a backtest on the current constituent list cannot see.
    Feed them to the ingest to make the survivorship audit possible. Note this
    conflates index removals, delistings and ticker renames.
    """
    if snapshots.is_empty():
        return []
    window = snapshots if since is None else snapshots.filter(pl.col("as_of") >= since)
    if window.is_empty():
        return []
    current = members_as_of(snapshots, snapshots["as_of"].max())
    return sorted(set(window["ticker"].to_list()) - current)
