"""Live runs on the public notes page: automatic, moderated, and anonymous.

A live run that finishes with a note passing every publication check goes on
the public notes page by itself, and stays until the owner hides it. These
tests pin the three things that make that safe: only checked notes appear,
only the owner can hide one, and nothing about the visitor crosses.
"""
from __future__ import annotations

import datetime as dt
import os

import pytest

from falsify.demo import ExportError, live_note_record, live_note_summary
from falsify.live.store import MemoryStore, PgStore, UnknownRun, is_public
from tests.test_live import TOKEN, _app, _submit

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _done(c, app, **kw) -> str:
    rid = _submit(c, **kw).json()["run_id"]
    app.state.worker.process_one(rid)
    return rid


def test_a_checked_run_appears_on_the_public_list_by_itself():
    app, store, c = _app()
    rid = _done(c, app, name="Jane at HRT")
    notes = c.get("/notes/live").json()["notes"]
    assert [n["note_id"] for n in notes] == [rid]
    assert notes[0]["source"] == "live" and notes[0]["created_at"]
    assert c.get(f"/notes/live/{rid}").json()["note_id"] == rid


def test_refused_and_unfinished_runs_never_appear():
    refused = lambda h: ({"hypothesis": h, "publishable": False, "prose": "x", "backtests": []}, 0.02)
    app, store, c = _app(runner=refused)
    rid = _done(c, app)
    queued = _submit(c, ip="9.9.9.9").json()["run_id"]
    assert c.get("/notes/live").json()["notes"] == []
    assert c.get(f"/notes/live/{rid}").status_code == 404
    assert c.get(f"/notes/live/{queued}").status_code == 404


def test_only_the_owner_can_hide_and_hiding_removes_it_everywhere():
    app, store, c = _app()
    rid = _done(c, app)
    assert c.post(f"/admin/runs/{rid}/hide", json={"hidden": True}).status_code == 401
    assert c.post(f"/admin/runs/{rid}/hide", json={"hidden": True},
                  headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert len(c.get("/notes/live").json()["notes"]) == 1
    assert c.post(f"/admin/runs/{rid}/hide", json={"hidden": True}, headers=AUTH).status_code == 200
    assert c.get("/notes/live").json()["notes"] == []
    assert c.get(f"/notes/live/{rid}").status_code == 404
    # The owner still sees it, flagged, and can put it back.
    row = c.get("/admin/runs", headers=AUTH).json()["runs"][0]
    assert row["hidden"] is True
    c.post(f"/admin/runs/{rid}/hide", json={"hidden": False}, headers=AUTH)
    assert len(c.get("/notes/live").json()["notes"]) == 1


def test_hiding_an_unknown_run_is_404():
    app, store, c = _app()
    assert c.post("/admin/runs/nope/hide", json={"hidden": True}, headers=AUTH).status_code == 404


def test_the_public_note_carries_no_visitor_id_name_or_tool_errors():
    store = MemoryStore()
    rid = store.create("visitor-hash-abc", "Jane at HRT", "Do small caps outperform?")
    store.finish(rid, {"publishable": True, "prose": "ok", "backtests": [],
                       "tool_errors": [{"tool": "fetch_data", "error": "host db.internal"}],
                       "hypothesis": "ignored"}, 0.03)
    row = store.list_public()[0]
    for rec in (live_note_record(row), live_note_summary(row)):
        text = str(rec)
        assert "visitor-hash-abc" not in text and "Jane" not in text and "db.internal" not in text
        assert rec["hypothesis"] == "Do small caps outperform?" and rec["source"] == "live"


def test_the_list_summary_leaves_the_charts_out():
    store = MemoryStore()
    rid = store.create("v", None, "q")
    store.finish(rid, {"publishable": True, "backtests": [
        {"variant": "mom_12_1, 10 buckets, long/short", "diagnostics": {"equity": [1] * 160}}]}, 0.03)
    s = live_note_summary(store.list_public()[0])
    assert s["backtests"] == [{"variant": "mom_12_1, 10 buckets, long/short"}]
    assert "equity" not in str(s)


def test_the_record_builders_refuse_a_row_that_is_not_public():
    base = {"run_id": "r", "created_at": dt.datetime.now(dt.timezone.utc), "hypothesis": "h",
            "status": "done", "note": {"publishable": True}, "hidden": False}
    assert is_public(base)
    for bad in ({"hidden": True}, {"status": "running"}, {"note": {"publishable": False}}, {"note": None}):
        row = {**base, **bad}
        with pytest.raises(ExportError, match="not public"):
            live_note_record(row)


def test_memory_store_refuses_an_unknown_run():
    with pytest.raises(UnknownRun):
        MemoryStore().set_hidden("nope", True)


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
def test_pg_store_public_list_and_hide_round_trip():
    s = PgStore(DSN)
    s.apply_schema()
    vid = "pytest-" + os.urandom(4).hex()
    try:
        rid = s.create(vid, None, "Do small caps outperform?")
        assert rid not in [r["run_id"] for r in s.list_public()]
        s.finish(rid, {"publishable": True, "prose": "ok"}, 0.03)
        assert rid in [r["run_id"] for r in s.list_public()]
        s.set_hidden(rid, True)
        assert rid not in [r["run_id"] for r in s.list_public()]
        refused = s.create(vid, None, "x")
        s.finish(refused, {"publishable": False}, 0.01)
        assert refused not in [r["run_id"] for r in s.list_public()]
    finally:
        import psycopg

        with psycopg.connect(DSN) as c:
            c.execute("DELETE FROM live_run WHERE visitor=%s", (vid,))
            c.commit()
