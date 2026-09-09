"""Ingest pipeline: Polygon daily bars -> TimescaleDB.

Idempotent: completed (ticker, range) pairs are recorded in ingest_log and
skipped on re-run, so a crashed 500-ticker pull resumes where it stopped.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import psycopg

from falsify.config import settings
from falsify.data.polygon_client import PolygonClient

BAR_COLUMNS = ["ticker", "ts", "open", "high", "low", "close", "volume", "vwap", "n_trades"]


def bars_to_frame(ticker: str, raw: list[dict]) -> pl.DataFrame:
    """Polygon agg results -> typed Polars frame. 't' is ms since epoch UTC."""
    if not raw:
        return pl.DataFrame(schema={c: pl.Utf8 for c in BAR_COLUMNS})
    df = pl.DataFrame(raw)
    return (
        df.with_columns(
            pl.lit(ticker).alias("ticker"),
            pl.from_epoch(pl.col("t"), time_unit="ms").dt.date().alias("ts"),
        )
        .rename({"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
                 "vw": "vwap", "n": "n_trades"})
        .select([c for c in BAR_COLUMNS if c in
                 {"ticker", "ts", "open", "high", "low", "close", "volume", "vwap", "n_trades"}])
        .with_columns(pl.col("volume").cast(pl.Int64), pl.col("n_trades").cast(pl.Int64, strict=False))
    )


def already_ingested(conn: psycopg.Connection, ticker: str, from_date: dt.date, to_date: dt.date) -> bool:
    row = conn.execute(
        "SELECT 1 FROM ingest_log WHERE ticker=%s AND from_date=%s AND to_date=%s",
        (ticker, from_date, to_date),
    ).fetchone()
    return row is not None


def write_bars(conn: psycopg.Connection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    rows = df.rows()
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO daily_bars (ticker, ts, open, high, low, close, volume, vwap, n_trades)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (ticker, ts) DO NOTHING""",
            rows,
        )
    return len(rows)


def ingest_ticker(client: PolygonClient, conn: psycopg.Connection,
                  ticker: str, from_date: dt.date, to_date: dt.date) -> int:
    if already_ingested(conn, ticker, from_date, to_date):
        return 0
    raw = client.daily_bars(ticker, from_date.isoformat(), to_date.isoformat())
    df = bars_to_frame(ticker, raw)
    n = write_bars(conn, df)
    conn.execute(
        "INSERT INTO ingest_log (ticker, from_date, to_date, n_rows) VALUES (%s,%s,%s,%s) "
        "ON CONFLICT DO NOTHING",
        (ticker, from_date, to_date, n),
    )
    conn.commit()
    return n


def ingest_universe(tickers: list[str], years: int = 5) -> None:
    to_date = dt.date.today()
    from_date = to_date - dt.timedelta(days=365 * years)
    client = PolygonClient()
    with psycopg.connect(settings.db_dsn) as conn:
        for i, t in enumerate(tickers, 1):
            try:
                n = ingest_ticker(client, conn, t, from_date, to_date)
                print(f"[{i}/{len(tickers)}] {t}: {n} rows" + (" (skipped)" if n == 0 else ""))
            except Exception as e:  # noqa: BLE001 - one bad ticker must not kill the run
                print(f"[{i}/{len(tickers)}] {t}: FAILED — {e}")
    client.close()
