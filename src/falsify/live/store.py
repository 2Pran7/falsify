"""live_run: a visitor's run, kept apart from the published research notes.

Two implementations with one interface: Postgres for the server, memory for
tests. Deliberately NOT research_note: export_demo.py reads research_note, and
a stranger's hypothesis must never reach the public page by that route.

The ONE route from here to the public page is `published`, a flag only the
owner can set (POST /admin/runs/{id}/publish), and only on a run whose note
passed every publication check. export_demo.py reads `list_published` and
carries the note and the question, never the visitor id or the name.
"""
from __future__ import annotations

import datetime as dt
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from falsify.config import settings
from falsify.live.limits import RUN_RESERVE_USD, Today

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS live_run (
    run_id        UUID        PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL,
    finished_at   TIMESTAMPTZ,
    visitor       TEXT        NOT NULL,
    display_name  TEXT,
    hypothesis    TEXT        NOT NULL,
    status        TEXT        NOT NULL,
    note          JSONB,
    error         TEXT,
    cost_usd      DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_live_run_created ON live_run (created_at DESC);
ALTER TABLE live_run ADD COLUMN IF NOT EXISTS published BOOLEAN NOT NULL DEFAULT false;
"""

STATUSES = ("queued", "running", "done", "failed")
_COLS = (
    "run_id, created_at, finished_at, visitor, display_name, hypothesis, "
    "status, note, error, cost_usd, published"
)


class NotPublishable(Exception):
    """Only a finished run whose note passed every check may be published."""


def _check_publishable(row: dict | None) -> None:
    if row is None:
        raise NotPublishable("No such run.")
    note = row.get("note")
    if row["status"] != "done" or not isinstance(note, dict) or not note.get("publishable"):
        raise NotPublishable(
            "Only a finished run whose note passed every publication check can be published."
        )


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _midnight(now: dt.datetime) -> dt.datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass
class MemoryStore:
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def apply_schema(self) -> None:
        pass

    def create(self, visitor: str, name: str | None, hypothesis: str, now: dt.datetime | None = None) -> str:
        rid = str(uuid.uuid4())
        with self.lock:
            self.rows[rid] = {
                "run_id": rid, "created_at": now or _now(), "finished_at": None,
                "visitor": visitor, "display_name": name, "hypothesis": hypothesis,
                "status": "queued", "note": None, "error": None, "cost_usd": RUN_RESERVE_USD,
                "published": False,
            }
        return rid

    def set_status(self, rid: str, status: str) -> None:
        with self.lock:
            self.rows[rid]["status"] = status

    def finish(self, rid: str, note: dict | None, cost: float, error: str | None = None) -> None:
        with self.lock:
            r = self.rows[rid]
            r.update(status="failed" if error else "done", note=note, cost_usd=cost,
                     error=error, finished_at=_now())

    def get(self, rid: str) -> dict | None:
        with self.lock:
            r = self.rows.get(rid)
            return dict(r) if r else None

    def today(self, visitor: str, now: dt.datetime | None = None) -> Today:
        start = _midnight(now or _now())
        with self.lock:
            rs = [r for r in self.rows.values() if r["created_at"] >= start]
            return Today(
                spend_usd=sum(r["cost_usd"] for r in rs),
                runs=len(rs),
                runs_by_visitor=sum(1 for r in rs if r["visitor"] == visitor),
                queued=sum(1 for r in self.rows.values() if r["status"] in ("queued", "running")),
            )

    def list_all(self, limit: int = 200) -> list[dict]:
        with self.lock:
            rs = sorted(self.rows.values(), key=lambda r: r["created_at"], reverse=True)
            return [dict(r) for r in rs[:limit]]

    def set_published(self, rid: str, published: bool) -> None:
        with self.lock:
            row = self.rows.get(rid)
            if published:
                _check_publishable(row)
            elif row is None:
                raise NotPublishable("No such run.")
            row["published"] = bool(published)

    def list_published(self) -> list[dict]:
        with self.lock:
            rs = [r for r in self.rows.values() if r["published"]]
            return [dict(r) for r in sorted(rs, key=lambda r: r["created_at"])]

    def fail_unfinished(self, reason: str) -> int:
        n = 0
        with self.lock:
            for r in self.rows.values():
                if r["status"] in ("queued", "running"):
                    # The reservation is KEPT: a run cut off mid-way did spend money,
                    # and how much is unknown, so the worst case stays on the books.
                    r.update(status="failed", error=reason, finished_at=_now())
                    n += 1
        return n


class PgStore:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.db_dsn

    def _c(self):
        return psycopg.connect(self.dsn)

    def apply_schema(self) -> None:
        with self._c() as c:
            c.execute(SCHEMA_SQL)
            c.commit()

    def create(self, visitor: str, name: str | None, hypothesis: str, now: dt.datetime | None = None) -> str:
        rid = str(uuid.uuid4())
        with self._c() as c:
            c.execute(
                "INSERT INTO live_run (run_id, created_at, visitor, display_name, hypothesis, status, cost_usd) "
                "VALUES (%s, %s, %s, %s, %s, 'queued', %s)",
                (rid, now or _now(), visitor, name, hypothesis, RUN_RESERVE_USD),
            )
            c.commit()
        return rid

    def set_status(self, rid: str, status: str) -> None:
        with self._c() as c:
            c.execute("UPDATE live_run SET status=%s WHERE run_id=%s", (status, rid))
            c.commit()

    def finish(self, rid: str, note: dict | None, cost: float, error: str | None = None) -> None:
        with self._c() as c:
            c.execute(
                "UPDATE live_run SET status=%s, note=%s, cost_usd=%s, error=%s, finished_at=%s "
                "WHERE run_id=%s",
                ("failed" if error else "done", None if note is None else Jsonb(note),
                 cost, error, _now(), rid),
            )
            c.commit()

    def _rows(self, sql: str, params: tuple) -> list[dict]:
        names = [x.strip() for x in _COLS.split(",")]
        with self._c() as c:
            rows = c.execute(sql, params).fetchall()
        out = []
        for row in rows:
            d = dict(zip(names, row))
            d["run_id"] = str(d["run_id"])
            out.append(d)
        return out

    def get(self, rid: str) -> dict | None:
        try:
            uuid.UUID(rid)
        except ValueError:
            return None
        rs = self._rows(f"SELECT {_COLS} FROM live_run WHERE run_id=%s", (rid,))
        return rs[0] if rs else None

    def today(self, visitor: str, now: dt.datetime | None = None) -> Today:
        start = _midnight(now or _now())
        with self._c() as c:
            spend, runs, mine = c.execute(
                "SELECT coalesce(sum(cost_usd),0), count(*), count(*) FILTER (WHERE visitor=%s) "
                "FROM live_run WHERE created_at >= %s",
                (visitor, start),
            ).fetchone()
            (queued,) = c.execute(
                "SELECT count(*) FROM live_run WHERE status IN ('queued','running')"
            ).fetchone()
        return Today(float(spend), int(runs), int(mine), int(queued))

    def list_all(self, limit: int = 200) -> list[dict]:
        return self._rows(f"SELECT {_COLS} FROM live_run ORDER BY created_at DESC LIMIT %s", (limit,))

    def set_published(self, rid: str, published: bool) -> None:
        row = self.get(rid)
        if published:
            _check_publishable(row)
        elif row is None:
            raise NotPublishable("No such run.")
        with self._c() as c:
            c.execute("UPDATE live_run SET published=%s WHERE run_id=%s", (bool(published), rid))
            c.commit()

    def list_published(self) -> list[dict]:
        return self._rows(
            f"SELECT {_COLS} FROM live_run WHERE published ORDER BY created_at", ()
        )

    def fail_unfinished(self, reason: str) -> int:
        with self._c() as c:
            cur = c.execute(
                "UPDATE live_run SET status='failed', error=%s, finished_at=%s "
                "WHERE status IN ('queued','running')",
                (reason, _now()),
            )
            c.commit()
            return cur.rowcount
