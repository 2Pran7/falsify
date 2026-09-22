"""Database access: daily_bars -> long-format Polars frames.

No methodology decisions here. Rows come back in the shape features/library.py
expects: one row per (ticker, ts), sorted by (ticker, ts).
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import psycopg

from falsify.config import settings
from falsify.data.pit_universe import SNAPSHOT_SCHEMA

BAR_COLUMNS =["ticker", "ts", "open", "high", "low", "close", "volume", "vwap", "n_trades"]

_SCHEMA = {
    "ticker": pl.Utf8,
    "ts": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "vwap": pl.Float64,
    "n_trades": pl.Int64,
}


def load_panel(
    tickers: list[str] | None = None,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    dsn: str | None = None,
) -> pl.DataFrame:
    """Read daily_bars into a long Polars panel.

    Args:
        tickers: restrict to these tickers. None = every ticker in the table.
        start, end: inclusive date bounds. None = unbounded.
        dsn: override the connection string (defaults to settings.db_dsn).

    Returns:
        Frame with columns BAR_COLUMNS, sorted by (ticker, ts).
    """
    where, params = [], []
    if tickers:
        where.append("ticker = ANY(%s)")
        params.append(list(tickers))
    if start:
        where.append("ts >= %s")
        params.append(start)
    if end:
        where.append("ts <= %s")
        params.append(end)

    sql = f"SELECT {', '.join(BAR_COLUMNS)} FROM daily_bars"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ticker, ts"

    with psycopg.connect(dsn or settings.db_dsn) as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return pl.DataFrame(schema=_SCHEMA)

    return pl.DataFrame(rows, schema=_SCHEMA, orient="row")


def load_prices(
    tickers: list[str] | None = None,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    dsn: str | None = None,
) -> pl.DataFrame:
    """(ticker, ts, close) only: the minimum the engine requires."""
    return load_panel(tickers, start, end, dsn).select(["ticker", "ts", "close"])


def load_snapshots(
    index_name: str = "SP500",
    dsn: str | None = None,
) -> pl.DataFrame:
    """Read `universe_snapshot` into the frame `pit_universe` consumes.

    Args:
        index_name: which index's snapshots to load.
        dsn: override the connection string.

    Returns:
        Frame (ticker, index_name, as_of) in `pit_universe.SNAPSHOT_SCHEMA`,
        sorted by (as_of, ticker). Empty frame of the right schema when the
        table holds nothing for this index.

    THE NUMBER OF DISTINCT `as_of` VALUES IS THE THING TO CHECK, and the caller
    is expected to. `run_ingest.py` writes exactly ONE snapshot, dated the day
    it ran. A membership panel built from a single snapshot never changes, so
    it reproduces today's constituent list for every historical date — with
    full coverage, no gap, and a survivorship audit that measures zero. It
    fails by looking healthy, which is why `agent.tools.fetch_data` refuses
    `point_in_time` on it rather than proceeding.
    """
    sql = """
        SELECT ticker, index_name, as_of FROM universe_snapshot
        WHERE index_name = %s ORDER BY as_of, ticker
    """
    with psycopg.connect(dsn or settings.db_dsn) as conn:
        rows = conn.execute(sql, (index_name,)).fetchall()
    if not rows:
        return pl.DataFrame(schema=SNAPSHOT_SCHEMA)
    return pl.DataFrame(rows, schema=SNAPSHOT_SCHEMA, orient="row")


def coverage(dsn: str | None = None) -> pl.DataFrame:
    """Row count and date span per ticker, for verifying ingest coverage."""
    sql = """
        SELECT ticker, count(*) AS n_rows, min(ts) AS first_ts, max(ts) AS last_ts
        FROM daily_bars GROUP BY ticker ORDER BY ticker
    """
    with psycopg.connect(dsn or settings.db_dsn) as conn:
        rows = conn.execute(sql).fetchall()
    schema = {"ticker": pl.Utf8, "n_rows": pl.Int64, "first_ts": pl.Date, "last_ts": pl.Date}
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row")
