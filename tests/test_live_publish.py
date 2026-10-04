"""Publishing a live run to the public notes: owner only, checked notes only,
and nothing about the visitor crosses.

The live table was built so a stranger's question could never reach the public
page by accident. Publishing opens one door on purpose, so these tests pin who
can open it, what may walk through, and what must stay behind.
"""
from __future__ import annotations

import datetime as dt
import os

import pytest

from falsify.demo import ExportError, build_snapshot, live_note_record
from falsify.live.store import MemoryStore, NotPublishable, PgStore
from tests.test_live import TOKEN, _app, _submit

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _done(c, app, **kw) -> str:
    rid = _submit(c, **kw).json()["run_id"]
    app.state.worker.process_one(rid)
    return rid


def test_only_the_owner_can_publish():
    app, store, c = _app()
    rid = _done(c, app)
    assert c.post(f"/admin/runs/{rid}/publish", json={"published": True}).status_code == 401
    r = c.post(f"/admin/runs/{rid}/publish", json={"published": True},
               headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert store.list_published() == []
    assert c.post(f"/admin/runs/{rid}/publish", json={"published": True}, headers=AUTH).status_code == 200
    assert [r["run_id"] for r in store.list_published()] == [rid]


def test_a_refused_or_unfinished_run_cannot_be_published():
    refused = lambda h: ({"hypothesis": h, "publishable": False, "prose": "x", "backtests": []}, 0.02)
    app, store, c = _app(runner=refused)
    rid = _done(c, app)
    assert c.post(f"/admin/runs/{rid}/publish", json={"published": True}, headers=AUTH).status_code == 409
    queued = _submit(c, ip="9.9.9.9").json()["run_id"]
    assert c.post(f"/admin/runs/{queued}/publish", json={"published": True}, headers=AUTH).status_code == 409
    assert store.list_published() == []


def test_unpublishing_takes_it_back_off():
    app, store, c = _app()
    rid = _done(c, app)
    c.post(f"/admin/runs/{rid}/publish", json={"published": True}, headers=AUTH)
    c.post(f"/admin/runs/{rid}/publish", json={"published": False}, headers=AUTH)
    assert store.list_published() == []


def test_the_owner_sees_the_published_flag():
    app, store, c = _app()
    rid = _done(c, app)
    c.post(f"/admin/runs/{rid}/publish", json={"published": True}, headers=AUTH)
    rows = c.get("/admin/runs", headers=AUTH).json()["runs"]
    assert rows[0]["published"] is True


def test_the_exported_note_carries_no_visitor_id_name_or_tool_errors():
    store = MemoryStore()
    rid = store.create("visitor-hash-abc", "Jane at HRT", "Do small caps outperform?")
    store.finish(rid, {"publishable": True, "prose": "ok", "backtests": [],
                       "tool_errors": [{"tool": "fetch_data", "error": "host db.internal"}],
                       "hypothesis": "ignored"}, 0.03)
    store.set_published(rid, True)
    rec = live_note_record(store.list_published()[0])
    text = str(rec)
    assert "visitor-hash-abc" not in text and "Jane" not in text and "db.internal" not in text
    assert rec["source"] == "live" and rec["note_id"] == rid
    assert rec["hypothesis"] == "Do small caps outperform?"


def test_export_refuses_a_row_that_was_not_approved():
    row = {"run_id": "r", "created_at": dt.datetime.now(dt.timezone.utc), "hypothesis": "h",
           "status": "done", "note": {"publishable": True}, "published": False}
    with pytest.raises(ExportError, match="not published"):
        live_note_record(row)
    row.update(published=True, note={"publishable": False})
    with pytest.raises(ExportError, match="publication check"):
        live_note_record(row)


def test_published_live_notes_join_the_snapshot_and_are_counted():
    from tests.test_demo_export import _all_rows as _rows

    store = MemoryStore()
    rid = store.create("v", None, "Do small caps outperform?")
    store.finish(rid, {"publishable": True, "prose": "ok", "backtests": []}, 0.03)
    store.set_published(rid, True)
    snap = build_snapshot(_rows(), [], live_rows=store.list_published())
    assert [n["note_id"] for n in snap["notes"]["items"]] == [rid]
    assert snap["notes"]["counts"]["publishable"] == 1


def test_memory_store_refuses_an_unknown_run():
    with pytest.raises(NotPublishable):
        MemoryStore().set_published("nope", True)


DSN = os.environ.get("DATABASE_URL")


def _pg_ok() -> bool:
    if not DSN:
        return False
    try:
        import psycopg

        with psycopg.connect(DSN, connect_timeout=3):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_ok(), reason="Postgres unreachable")
def test_pg_store_publish_round_trip():
    s = PgStore(DSN)
    s.apply_schema()
    vid = "pytest-" + os.urandom(4).hex()
    try:
        rid = s.create(vid, None, "Do small caps outperform?")
        with pytest.raises(NotPublishable):
            s.set_published(rid, True)
        s.finish(rid, {"publishable": True, "prose": "ok"}, 0.03)
        s.set_published(rid, True)
        assert rid in [r["run_id"] for r in s.list_published()]
        s.set_published(rid, False)
        assert rid not in [r["run_id"] for r in s.list_published()]
    finally:
        import psycopg

        with psycopg.connect(DSN) as c:
            c.execute("DELETE FROM live_run WHERE visitor=%s", (vid,))
            c.commit()
