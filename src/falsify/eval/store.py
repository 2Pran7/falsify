"""Eval verdicts in Postgres, keyed by (anomaly, universe).

THE KEY IS THE DESIGN DECISION. A verdict is not a property of an anomaly, it
is a property of an anomaly RUN ON A UNIVERSE. Momentum scores differently on
today's constituent list than on point-in-time membership -- that difference is
the project's headline finding -- so both rows must coexist. Keying on
`eval_key` alone would make the second run silently overwrite the first, and
the survivorship comparison would return a single row and look fine.

`registry_sha` and `scoring_rule_version` sit on every row. Together they say
"this prediction, judged by this rule". Neither alone would reveal that two
rows in the same table had been held to different standards, which is exactly
what happens when the registry is edited between runs.

`reasons` is stored, not just the verdict. Module 7 shows the reasons; a bare
count of passes is a number nobody can act on, and it is also how an eval suite
turns into a highlight reel.

---

AN IDEMPOTENT SCHEMA STATEMENT IS A NO-OP, AND A NO-OP IS INDISTINGUISHABLE
FROM A SUCCESS. This is the fifth lesson in the project's list and it was
learned here.

Every schema statement in this repo is `CREATE TABLE IF NOT EXISTS`, which
means that after the first `docker compose up`, ANY subsequent edit to the
schema is silently ignored on that database, forever. The injection audit
dropped `universe` from the primary key below; the string changed, nothing else
did, and every read and write kept succeeding while the code and the deployed
table said different things.

Two checks close it, and the second is the one that matters:

  1. `test_eval_store.py` asserts the PRIMARY KEY clause appears in BOTH this
     module's SCHEMA_SQL and `db/schema.sql` -- the pair that must stay in step.
  2. `primary_key_columns()` reads the key from `pg_index` ON THE LIVE SERVER,
     and a test asserts it is exactly ["eval_key", "universe"]. The invariant is
     checked against the database, not against the string that was supposed to
     create it.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from falsify.config import settings
from falsify.eval.score import Score

TABLE = "eval_result"

# Mirrored in db/schema.sql. Kept here too so a test or a fresh clone can
# create the table without the Docker init path. THEY MUST STAY IN STEP; see
# the module docstring for what happens when they do not.
SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    eval_key             TEXT        NOT NULL,
    universe             TEXT        NOT NULL,
    run_at               TIMESTAMPTZ NOT NULL,
    registry_sha         TEXT        NOT NULL,
    scoring_rule_version INTEGER     NOT NULL,
    verdict              TEXT        NOT NULL,
    reasons              JSONB       NOT NULL,
    direction            INTEGER     NOT NULL,
    realised_sharpe      DOUBLE PRECISION,
    oriented_sharpe      DOUBLE PRECISION,
    published_sharpe     DOUBLE PRECISION NOT NULL,
    p_value              DOUBLE PRECISION,
    p_value_adjusted     DOUBLE PRECISION,
    deflated_psr         DOUBLE PRECISION,
    n_invested_days      INTEGER     NOT NULL,
    history_days         INTEGER     NOT NULL,
    min_history_days     INTEGER     NOT NULL,
    n_trials             INTEGER     NOT NULL,
    caveat               TEXT        NOT NULL,
    detail               JSONB       NOT NULL,
    PRIMARY KEY (eval_key, universe)
);

CREATE INDEX IF NOT EXISTS idx_eval_result_verdict
    ON {TABLE} (verdict, universe);
"""

_COLUMNS = (
    "eval_key, universe, run_at, registry_sha, scoring_rule_version, verdict, "
    "reasons, direction, realised_sharpe, oriented_sharpe, published_sharpe, "
    "p_value, p_value_adjusted, deflated_psr, n_invested_days, history_days, "
    "min_history_days, n_trials, caveat, detail"
)


def _dsn(dsn: str | None) -> str:
    return dsn or settings.db_dsn


def apply_schema(dsn: str | None = None) -> None:
    """Create the table and its index. Safe to re-apply -- and see the docstring
    above for why "safe to re-apply" is exactly the hazard."""
    with psycopg.connect(_dsn(dsn)) as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()


def primary_key_columns(dsn: str | None = None) -> list[str]:
    """The table's ACTUAL primary-key columns, read from the live server.

    Returns:
        Column names in key order, e.g. ["eval_key", "universe"]. Empty list if
        the table has no primary key or does not exist.

    This is the check that `CREATE TABLE IF NOT EXISTS` defeats. Asserting on
    SCHEMA_SQL only proves what the code intended; this proves what the
    database did. Any invariant that can drift between the two belongs here.
    """
    sql = """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = to_regclass(%s) AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
    """
    with psycopg.connect(_dsn(dsn)) as conn:
        rows = conn.execute(sql, (TABLE,)).fetchall()
    return [r[0] for r in rows]


def save(
    score: Score,
    universe: str,
    registry_sha: str,
    run_at: dt.datetime | None = None,
    detail: dict[str, Any] | None = None,
    dsn: str | None = None,
) -> None:
    """Write one verdict, replacing the existing row for (eval_key, universe).

    Re-running the suite overwrites in place rather than accumulating, because
    the table is the CURRENT state of the evidence and Module 7 renders it
    directly. A verdict that has been superseded is not history worth keeping:
    it was computed from a shorter panel by an older rule, and both facts are
    recorded on the row that replaced it.
    """
    row = (
        score.key,
        universe,
        run_at or dt.datetime.now(dt.timezone.utc),
        registry_sha,
        score.scoring_rule_version,
        score.verdict,
        Jsonb(list(score.reasons)),
        score.direction,
        score.realised_sharpe,
        score.oriented_sharpe,
        score.published_sharpe,
        score.p_value,
        score.p_value_adjusted,
        score.deflated_psr,
        score.n_invested_days,
        score.history_days,
        score.min_history_days,
        score.n_trials,
        score.caveat,
        Jsonb(detail or score.to_dict()),
    )
    sql = f"""
        INSERT INTO {TABLE} ({_COLUMNS})
        VALUES ({', '.join(['%s'] * 20)})
        ON CONFLICT (eval_key, universe) DO UPDATE SET
            run_at               = EXCLUDED.run_at,
            registry_sha         = EXCLUDED.registry_sha,
            scoring_rule_version = EXCLUDED.scoring_rule_version,
            verdict              = EXCLUDED.verdict,
            reasons              = EXCLUDED.reasons,
            direction            = EXCLUDED.direction,
            realised_sharpe      = EXCLUDED.realised_sharpe,
            oriented_sharpe      = EXCLUDED.oriented_sharpe,
            published_sharpe     = EXCLUDED.published_sharpe,
            p_value              = EXCLUDED.p_value,
            p_value_adjusted     = EXCLUDED.p_value_adjusted,
            deflated_psr         = EXCLUDED.deflated_psr,
            n_invested_days      = EXCLUDED.n_invested_days,
            history_days         = EXCLUDED.history_days,
            min_history_days     = EXCLUDED.min_history_days,
            n_trials             = EXCLUDED.n_trials,
            caveat               = EXCLUDED.caveat,
            detail               = EXCLUDED.detail
    """
    with psycopg.connect(_dsn(dsn)) as conn:
        conn.execute(sql, row)
        conn.commit()


def save_suite(
    scores,
    universe: str,
    registry_sha: str,
    run_at: dt.datetime | None = None,
    dsn: str | None = None,
) -> int:
    """Write every verdict from one suite run. Returns the row count.

    Unpublishable, failing and untestable rows are written like any other. A
    suite that only stored its passes would be a highlight reel, and the count
    of honest failures is a number the demo shows.
    """
    run_at = run_at or dt.datetime.now(dt.timezone.utc)
    n = 0
    for s in scores:
        save(s, universe, registry_sha, run_at=run_at, dsn=dsn)
        n += 1
    return n


def fetch_all(universe: str | None = None, dsn: str | None = None) -> list[dict]:
    """Every stored verdict, newest run first, as plain dicts."""
    sql = f"SELECT {_COLUMNS} FROM {TABLE}"
    params: list[Any] = []
    if universe is not None:
        sql += " WHERE universe = %s"
        params.append(universe)
    sql += " ORDER BY run_at DESC, eval_key"
    names = [c.strip() for c in _COLUMNS.split(",")]
    with psycopg.connect(_dsn(dsn)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(zip(names, r)) for r in rows]


def fetch(eval_key: str, universe: str, dsn: str | None = None) -> dict | None:
    """One verdict, or None."""
    sql = f"SELECT {_COLUMNS} FROM {TABLE} WHERE eval_key = %s AND universe = %s"
    names = [c.strip() for c in _COLUMNS.split(",")]
    with psycopg.connect(_dsn(dsn)) as conn:
        row = conn.execute(sql, (eval_key, universe)).fetchone()
    return None if row is None else dict(zip(names, row))


def stale_rows(current_sha: str, dsn: str | None = None) -> list[dict]:
    """Rows whose `registry_sha` no longer matches the current registry.

    A stale row was scored against a prediction that has since been edited, so
    quoting it beside a fresh one puts two different standards in one table
    with nothing to distinguish them. `scripts/run_evals.py --check` exits
    non-zero when this is non-empty: a check that refuses to run is worth more
    than a check that warns.
    """
    return [r for r in fetch_all(dsn=dsn) if r["registry_sha"] != current_sha]


def counts(dsn: str | None = None) -> dict[str, int]:
    """Verdict counts across every stored row."""
    sql = f"SELECT verdict, count(*) FROM {TABLE} GROUP BY verdict"
    with psycopg.connect(_dsn(dsn)) as conn:
        rows = conn.execute(sql).fetchall()
    return {v: int(n) for v, n in rows}


def delete(eval_key: str, universe: str, dsn: str | None = None) -> bool:
    """Remove one verdict. Present for test teardown, not housekeeping."""
    with psycopg.connect(_dsn(dsn)) as conn:
        cur = conn.execute(
            f"DELETE FROM {TABLE} WHERE eval_key = %s AND universe = %s",
            (eval_key, universe),
        )
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
    "primary_key_columns",
    "save",
    "save_suite",
    "stale_rows",
]
