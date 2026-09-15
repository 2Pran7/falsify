"""Notes in Postgres. The whole of "durable" is this file.

Before Module 5 a research note existed in one place: the model's final turn,
printed to a terminal and mirrored into `.cache/runs/`, which is gitignored.
Module 7 has to serve a note without a Claude call and without a rerun, and it
cannot do that from a file nobody kept. Module 5 closes when a note survives a
process restart, not when the code is written.

THE SHAPE OF THE TABLE, and the one thing worth arguing about. `metrics`,
`statistics`, `provenance` and `run_meta` go in as JSONB rather than as forty
columns. The pipeline's output keys change (`prob_beats_best_of_n_trials` was
renamed after run 1 put it in the wrong column), and a rename should not need a
migration on a table holding six rows. What is NOT in JSONB is anything Module
7 filters on: `publishable` and `eval_key` are real columns.

`publishable` is DERIVED. It is stored so the web page can filter in SQL, and
it is recomputed on read from the record itself, so a row hand-edited to say
true does not come back publishable. The column is an index, never the source
of truth. There is a test for exactly that.

Unpublishable notes are stored. They are the eval suite's honest failures, and
the count of them is a number the page shows.
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from falsify.config import settings
from falsify.notes.schema import Note

TABLE = "research_note"

# Kept here as well as in db/schema.sql so a test, a fresh clone or a Module 6
# backfill can create the table without running the Docker init path. Both are
# CREATE ... IF NOT EXISTS, so applying either twice is a no-op and applying
# both is harmless. They must stay in step; there is a test that the columns
# this module writes all exist.
SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    note_id               UUID        PRIMARY KEY,
    eval_key              TEXT,
    created_at            TIMESTAMPTZ NOT NULL,
    schema_version        INTEGER     NOT NULL,
    hypothesis            TEXT        NOT NULL,
    prose                 TEXT        NOT NULL,
    publishable           BOOLEAN     NOT NULL,
    unpublishable_reasons JSONB       NOT NULL,
    backtests             JSONB       NOT NULL,
    provenance            JSONB       NOT NULL,
    run_meta              JSONB       NOT NULL,
    assumptions           JSONB       NOT NULL
);

-- Partial, so ad-hoc notes (eval_key IS NULL) can pile up while an eval case
-- has exactly one current row. Module 6 re-runs every anomaly and overwrites.
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_note_eval_key
    ON {TABLE} (eval_key) WHERE eval_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_research_note_publishable
    ON {TABLE} (publishable, created_at DESC);
"""

_COLUMNS = (
    "note_id, eval_key, created_at, schema_version, hypothesis, prose, "
    "publishable, unpublishable_reasons, backtests, provenance, run_meta, "
    "assumptions"
)


def _dsn(dsn: str | None) -> str:
    return dsn or settings.db_dsn


def apply_schema(dsn: str | None = None) -> None:
    """Create the table and its indexes. Safe to re-apply."""
    with psycopg.connect(_dsn(dsn)) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()


def save(note: Note, dsn: str | None = None) -> str:
    """Write a note, returning its id.

    A note carrying an `eval_key` REPLACES the existing row for that key, so
    Module 6 can re-run an anomaly without accumulating stale rows. A note
    without one always inserts.

    Unpublishable notes are written like any other. Refusing them here is how
    an eval suite quietly becomes a highlight reel.
    """
    data = note.to_dict()
    params = (
        note.note_id,
        note.eval_key,
        note.created_at,
        note.schema_version,
        note.hypothesis,
        note.prose,
        note.publishable,
        Jsonb(data["unpublishable_reasons"]),
        Jsonb(data["backtests"]),
        Jsonb(data["provenance"]),
        Jsonb(data["run"]),
        Jsonb(data["assumptions"]),
    )
    # ON CONFLICT infers the partial unique index by repeating its predicate.
    sql = f"""
        INSERT INTO {TABLE} ({_COLUMNS})
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (eval_key) WHERE eval_key IS NOT NULL
        DO UPDATE SET
            note_id               = EXCLUDED.note_id,
            created_at            = EXCLUDED.created_at,
            schema_version        = EXCLUDED.schema_version,
            hypothesis            = EXCLUDED.hypothesis,
            prose                 = EXCLUDED.prose,
            publishable           = EXCLUDED.publishable,
            unpublishable_reasons = EXCLUDED.unpublishable_reasons,
            backtests             = EXCLUDED.backtests,
            provenance            = EXCLUDED.provenance,
            run_meta              = EXCLUDED.run_meta,
            assumptions           = EXCLUDED.assumptions
        RETURNING note_id
    """
    with psycopg.connect(_dsn(dsn)) as conn:
        row = conn.execute(sql, params).fetchone()
        conn.commit()
    return str(row[0])


def _row_to_note(row: tuple) -> Note:
    """One row back into a Note.

    `publishable` and `unpublishable_reasons` are read past deliberately: they
    are recomputed by the Note from the provenance verdict and the run
    metadata. A row edited in psql to claim publishable does not come back
    publishable, which is the point of making it a property rather than a
    field.
    """
    (
        note_id,
        eval_key,
        created_at,
        schema_version,
        hypothesis,
        prose,
        _publishable,
        _reasons,
        backtests,
        provenance,
        run_meta,
        assumptions,
    ) = row
    return Note.from_dict(
        {
            "note_id": str(note_id),
            "eval_key": eval_key,
            "created_at": created_at,
            "schema_version": schema_version,
            "hypothesis": hypothesis,
            "prose": prose,
            "backtests": backtests,
            "provenance": provenance,
            "run": run_meta,
            "assumptions": assumptions,
        }
    )


def fetch(note_id: str, dsn: str | None = None) -> Note | None:
    """One note by id, or None."""
    sql = f"SELECT {_COLUMNS} FROM {TABLE} WHERE note_id = %s"
    with psycopg.connect(_dsn(dsn)) as conn:
        row = conn.execute(sql, (note_id,)).fetchone()
    return None if row is None else _row_to_note(row)


def fetch_by_eval_key(eval_key: str, dsn: str | None = None) -> Note | None:
    """The current note for one Module 6 anomaly, or None."""
    sql = f"SELECT {_COLUMNS} FROM {TABLE} WHERE eval_key = %s"
    with psycopg.connect(_dsn(dsn)) as conn:
        row = conn.execute(sql, (eval_key,)).fetchone()
    return None if row is None else _row_to_note(row)


def fetch_all(
    dsn: str | None = None,
    publishable_only: bool = False,
    limit: int | None = None,
) -> list[Note]:
    """Notes, newest first.

    `publishable_only` filters on the stored column for speed and then filters
    again on the recomputed property, so a stale or edited column can only ever
    hide a note, never publish one.
    """
    sql = f"SELECT {_COLUMNS} FROM {TABLE}"
    params: list[Any] = []
    if publishable_only:
        sql += " WHERE publishable"
    sql += " ORDER BY created_at DESC"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    with psycopg.connect(_dsn(dsn)) as conn:
        rows = conn.execute(sql, params).fetchall()
    notes = [_row_to_note(r) for r in rows]
    return [n for n in notes if n.publishable] if publishable_only else notes


def counts(dsn: str | None = None) -> dict[str, int]:
    """total, publishable and unpublishable.

    The last one is not a defect count to be minimised. A demo that shows six
    anomalies and says nothing about the runs that failed is a highlight reel;
    this is the number that stops it being one.
    """
    sql = f"""
        SELECT count(*), count(*) FILTER (WHERE publishable) FROM {TABLE}
    """
    with psycopg.connect(_dsn(dsn)) as conn:
        total, ok = conn.execute(sql).fetchone()
    return {
        "total": int(total),
        "publishable": int(ok),
        "unpublishable": int(total) - int(ok),
    }


def delete(note_id: str, dsn: str | None = None) -> bool:
    """Remove one note. Returns whether a row went.

    Present for test teardown and for dropping a run that should never have
    been recorded, not as routine housekeeping: a failed note is evidence.
    """
    with psycopg.connect(_dsn(dsn)) as conn:
        cur = conn.execute(f"DELETE FROM {TABLE} WHERE note_id = %s", (note_id,))
        conn.commit()
        return cur.rowcount > 0


def available(dsn: str | None = None) -> bool:
    """Whether the store is reachable, for a clean skip in the test suite."""
    try:
        with psycopg.connect(_dsn(dsn), connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


__all__ = [
    "SCHEMA_SQL",
    "TABLE",
    "apply_schema",
    "available",
    "counts",
    "delete",
    "fetch",
    "fetch_all",
    "fetch_by_eval_key",
    "save",
]
