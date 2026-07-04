"""Run the full Module 1 ingest: fetch universe, snapshot it, pull 5yr bars.

Usage:  python scripts/run_ingest.py            # full S&P 500
        python scripts/run_ingest.py AAPL MSFT  # specific tickers (smoke test)
"""
import sys

sys.path.insert(0, "src")

import psycopg

from falsify.config import settings
from falsify.data.ingest import ingest_universe
from falsify.data.universe import fetch_sp500_tickers, snapshot_frame

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
        print(f"Smoke test mode: {tickers}")
    else:
        tickers = fetch_sp500_tickers()
        print(f"Fetched {len(tickers)} S&P 500 tickers")
        snap = snapshot_frame(tickers)
        with psycopg.connect(settings.db_dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO universe_snapshot (ticker, index_name, as_of) "
                    "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    snap.rows(),
                )
            conn.commit()
    ingest_universe(tickers, years=5)
    print("Done.")
