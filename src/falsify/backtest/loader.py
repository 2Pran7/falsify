"""Database access: daily_bars -> long-format Polars frames.

Nothing here makes a methodology decision. It reads rows out of daily_bars and
hands back a long-format frame in exactly the shape features/library.py expects:
one row per (ticker, ts), sorted by (ticker, ts).
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import psycopg

from falsify.config import settings

BAR_COLUMNS = ["ticker", "ts", "open", "high", "low", "close", "volume", "vwap", "n_trades"]

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
    """Just (ticker, ts, close) — the minimum the engine needs."""
    return load_panel(tickers, start, end, dsn).select(["ticker", "ts", "close"])


def coverage(dsn: str | None = None) -> pl.DataFrame:
    """Sanity-check query: rows and date span per ticker. Use this to eyeball
    that the ingest covered the expected range."""
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
