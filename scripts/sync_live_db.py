"""Copy the price data the agent needs to the hosted database behind the live demo.

Usage
-----
    python scripts/sync_live_db.py --target "postgresql://...neon.tech/neondb?sslmode=require"
    python scripts/sync_live_db.py --target "..." --dry-run

Copies daily_bars and universe_snapshot from your local database (DATABASE_URL)
to the target, replacing what the target holds. Re-run it after every ingest,
including the M6b backfill, or the live agent answers from stale prices.

WHAT IS NOT COPIED, on purpose: research_note and eval_result. The live server
never reads them, and live_run on the target is left alone, so re-syncing
never erases what visitors have run.

The target gets the schema WITHOUT TimescaleDB. The hosted database is plain
Postgres, and nothing the agent runs depends on a hypertable; it only needs the
tables and their primary keys. Rows are streamed with COPY, so this uses a
few megabytes of memory whatever the table size.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time

import psycopg

sys.path.insert(0, "src")
from falsify.config import settings  # noqa: E402
from falsify.live.store import SCHEMA_SQL as LIVE_SCHEMA  # noqa: E402

TABLES = ("daily_bars", "universe_snapshot")


def plain_schema() -> str:
    """db/schema.sql for the two tables, minus everything TimescaleDB."""
    # Comments first: they contain semicolons, and splitting on ';' before
    # removing them cut a comment in half and sent the rest as SQL.
    sql = re.sub(r"--[^\n]*", "", pathlib.Path("db/schema.sql").read_text(encoding="utf-8"))
    keep = []
    for stmt in sql.split(";"):
        s = stmt.strip()
        if not s or "timescaledb" in s or "create_hypertable" in s:
            continue
        if any(re.search(rf"\b{t}\b", s) for t in TABLES):
            keep.append(s + ";")
    return "\n".join(keep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="the hosted database URL")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.target.strip() == settings.db_dsn.strip():
        print("Refusing: --target is your local database. Pass the hosted URL.")
        return 2

    with psycopg.connect(settings.db_dsn) as src:
        counts = {t: src.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}
    for t, n in counts.items():
        print(f"local {t:<18} {n:>10,} rows")
    if any(n == 0 for n in counts.values()):
        print("Refusing: a source table is empty. Start Docker and check the ingest.")
        return 2
    if args.dry_run:
        print("--dry-run: nothing copied.")
        return 0

    t0 = time.time()
    with psycopg.connect(settings.db_dsn) as src, psycopg.connect(args.target) as dst:
        dst.execute(plain_schema())
        dst.execute(LIVE_SCHEMA)
        for t in TABLES:
            dst.execute(f"TRUNCATE {t}")
            with src.cursor().copy(f"COPY {t} TO STDOUT (FORMAT BINARY)") as out, \
                 dst.cursor().copy(f"COPY {t} FROM STDIN (FORMAT BINARY)") as inp:
                for chunk in out:
                    inp.write(chunk)
            print(f"copied {t}")
        dst.commit()
        remote = {t: dst.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}

    ok = remote == counts
    for t in TABLES:
        print(f"target {t:<17} {remote[t]:>10,} rows  {'ok' if remote[t] == counts[t] else 'MISMATCH'}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
