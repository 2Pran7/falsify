"""Live runs: limits, visibility and the worker. No API key, no network, no DB.

The runner is injected, so the real agent loop never runs here; what is tested
is everything the server adds around it: who may start a run, who may read one,
and what the owner sees.
"""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from falsify.live import limits as L
from falsify.live.app import create_app
from falsify.live.store import MemoryStore
from falsify.live.worker import Worker

TOKEN = "t" * 32
NOTE = {"hypothesis": "h", "publishable": True, "prose": "ok", "backtests": []}


def fake_runner(h: str):
    return dict(NOTE, hypothesis=h), 0.03


def boom(h: str):
    raise RuntimeError("model unavailable")


def _app(store=None, runner=fake_runner, lim=L.Limits(), start=False):
    store = store or MemoryStore()
    app = create_app(store, runner, TOKEN, "salt", ["https://x.test"], lim, start_worker=start)
    return app, store, TestClient(app)


def _submit(c, text="Do low-volatility stocks outperform?", ip="1.1.1.1", **kw):
    return c.post("/runs", json={"hypothesis": text, **kw}, headers={"x-forwarded-for": ip})


# --- limits (pure) --------------------------------------------------------


def test_spend_cap_refuses_before_run_cap():
    with pytest.raises(L.Refused, match="budget"):
        L.check(L.Limits(), L.Today(spend_usd=1.9, runs=0, runs_by_visitor=0, queued=0))


def test_reserve_covers_a_worst_case_run():
    worst = (L.RUN_MAX_TURNS * L.RUN_MAX_OUTPUT_TOKENS) / 1e6 * 10 + L.RUN_TOKEN_BUDGET / 1e6 * 2
    assert L.RUN_RESERVE_USD >= worst - 1e-9


def test_hypothesis_is_bounded_and_normalised():
    assert L.clean_hypothesis("  Do   small caps win?  ") == "Do small caps win?"
    with pytest.raises(L.Refused):
        L.clean_hypothesis("hi")
    with pytest.raises(L.Refused):
        L.clean_hypothesis("x" * 301)


def test_visitor_id_does_not_contain_the_ip():
    v = L.visitor_id("81.2.69.160", "salt")
    assert "81.2" not in v and len(v) == 12
    assert v == L.visitor_id("81.2.69.160", "salt")
    assert v != L.visitor_id("81.2.69.160", "other")


# --- submit and poll -------------------------------------------------------


def test_a_run_is_queued_then_done_and_readable_by_its_id():
    app, store, c = _app()
    r = _submit(c)
    assert r.status_code == 202
    rid = r.json()["run_id"]
    assert c.get(f"/runs/{rid}").json()["status"] == "queued"
    app.state.worker.process_one(rid)
    body = c.get(f"/runs/{rid}").json()
    assert body["status"] == "done"
    assert body["note"]["publishable"] is True


def test_the_visitor_view_carries_no_visitor_id_or_name():
    app, store, c = _app()
    rid = _submit(c, name="Jane at HRT").json()["run_id"]
    body = c.get(f"/runs/{rid}").json()
    assert "visitor" not in body and "display_name" not in body


def test_there_is_no_public_route_that_lists_runs():
    app, store, c = _app()
    _submit(c)
    assert c.get("/runs").status_code in (404, 405)
    assert c.get("/admin/runs").status_code == 401
    assert c.get("/admin/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_the_owner_sees_every_run_with_name_and_hashed_visitor():
    app, store, c = _app()
    _submit(c, name="Jane at HRT", ip="1.1.1.1")
    _submit(c, text="Do small caps outperform large caps?", ip="2.2.2.2")
    body = c.get("/admin/runs", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert len(body["runs"]) == 2
    names = {r["display_name"] for r in body["runs"]}
    assert "Jane at HRT" in names
    assert all("1.1.1.1" not in str(r) for r in body["runs"])


def test_an_unknown_run_id_is_404_not_500():
    app, store, c = _app()
    assert c.get("/runs/not-a-uuid").status_code == 404


# --- the caps ----------------------------------------------------------------


def test_per_visitor_cap():
    app, store, c = _app(lim=L.Limits(per_visitor_cap=2, max_queue=99))
    assert _submit(c).status_code == 202
    assert _submit(c).status_code == 202
    r = _submit(c)
    assert r.status_code == 429 and "your 2 runs" in r.json()["detail"]
    assert _submit(c, ip="9.9.9.9").status_code == 202


def test_queued_runs_reserve_their_worst_case_against_the_dollar_cap():
    """Five queued runs must not each read '$0 spent'. With a $1 cap and a
    $0.45 reservation, the third submission is refused before any run finishes."""
    app, store, c = _app(lim=L.Limits(daily_usd_cap=1.0, per_visitor_cap=99, max_queue=99))
    assert _submit(c).status_code == 202
    assert _submit(c).status_code == 202
    r = _submit(c)
    assert r.status_code == 429 and "budget" in r.json()["detail"]


def test_finishing_replaces_the_reservation_with_the_real_cost():
    app, store, c = _app(lim=L.Limits(daily_usd_cap=1.0, per_visitor_cap=99, max_queue=99))
    rid = _submit(c).json()["run_id"]
    app.state.worker.process_one(rid)
    assert store.today("x").spend_usd == pytest.approx(0.03)


def test_a_failed_run_keeps_its_reservation_and_hides_the_error_detail():
    app, store, c = _app(runner=boom)
    rid = _submit(c).json()["run_id"]
    app.state.worker.process_one(rid)
    row = store.get(rid)
    assert row["status"] == "failed" and row["cost_usd"] == L.RUN_RESERVE_USD
    body = c.get(f"/runs/{rid}").json()
    assert "model unavailable" not in str(body)


def test_yesterdays_runs_do_not_count_today():
    store = MemoryStore()
    y = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    for _ in range(5):
        store.create("v", None, "old question here", now=y)
    assert store.today("v").runs == 0


def test_a_restart_fails_unfinished_runs_instead_of_leaving_them_queued():
    store = MemoryStore()
    rid = store.create("v", None, "Do small caps outperform?")
    _app(store=store)
    assert store.get(rid)["status"] == "failed"


def test_a_short_admin_token_refuses_to_start():
    with pytest.raises(RuntimeError):
        create_app(MemoryStore(), fake_runner, "short", "s", [], start_worker=False)


def test_live_runs_never_touch_research_note():
    """The whole visibility design rests on this: export_demo reads
    research_note, so a visitor's run must not be written there."""
    import inspect

    import falsify.live.app as A
    import falsify.live.store as S
    import falsify.live.worker as W

    for mod in (A, S, W):
        src = inspect.getsource(mod)
        assert "notes.store" not in src and "note_store" not in src
        assert "INSERT INTO research_note" not in src


# --- the real runner, with the scripted fake model --------------------------


def test_agent_runner_produces_a_note_record_through_the_real_loop(monkeypatch):
    """Everything real except the model and the database: the loop, the tools,
    provenance, the Note and the record the page renders."""
    from falsify.agent import tools as T
    from falsify.live.worker import agent_runner
    from tests.test_agent_loop import FakeClient, _panel, response, text_block, tool_block

    monkeypatch.setattr(T, "load_panel", lambda *a, **k: _panel())
    script = [
        response([tool_block("fetch_data", {}, "a")], "tool_use"),
        response([tool_block("compute_feature", {"panel_handle": "panel_1", "feature": "mom_12_1"}, "b")], "tool_use"),
        response([tool_block("run_backtest", {"feature_handle": "feature_1"}, "c")], "tool_use"),
        response([tool_block("analyze_results", {"backtest_handle": "backtest_1"}, "d")], "tool_use"),
        response([text_block("The evidence is insufficient.")], "end_turn"),
    ]
    note, cost = agent_runner(lambda: FakeClient(script))("Do past winners keep winning?")
    assert note["hypothesis"] == "Do past winners keep winning?"
    assert note["run"]["stop_reason"] == "end_turn"
    assert note["run"]["tool_sequence"][-1] == "analyze_results"
    assert isinstance(note["publishable"], bool) and cost >= 0


# --- Postgres ------------------------------------------------------------------

import os  # noqa: E402

from falsify.live.store import PgStore  # noqa: E402

DSN = os.getenv("DATABASE_URL")


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
def test_pg_store_round_trip_and_today_counts():
    s = PgStore(DSN)
    s.apply_schema()
    vid = "pytest-" + os.urandom(4).hex()
    try:
        rid = s.create(vid, "Tester", "Do small caps outperform?")
        t = s.today(vid)
        assert t.runs_by_visitor == 1 and t.spend_usd >= L.RUN_RESERVE_USD
        s.set_status(rid, "running")
        s.finish(rid, {"k": 1}, 0.031)
        row = s.get(rid)
        assert row["status"] == "done" and row["note"] == {"k": 1}
        assert row["cost_usd"] == pytest.approx(0.031)
        assert any(r["run_id"] == rid for r in s.list_all())
        assert s.get("not-a-uuid") is None
    finally:
        import psycopg

        with psycopg.connect(DSN) as c:
            c.execute("DELETE FROM live_run WHERE visitor=%s", (vid,))
            c.commit()


def test_tool_errors_reach_the_owner_but_not_the_visitor():
    def runner(h):
        return dict(NOTE, tool_errors=[{"tool": "fetch_data", "error": "no rows matched"}]), 0.01
    app, store, c = _app(runner=runner)
    rid = _submit(c).json()["run_id"]
    app.state.worker.process_one(rid)
    assert "tool_errors" not in c.get(f"/runs/{rid}").json()["note"]
    admin = c.get("/admin/runs", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert admin["runs"][0]["note"]["tool_errors"][0]["error"] == "no rows matched"


def test_agent_runner_records_a_failing_tool_verbatim(monkeypatch):
    from falsify.agent import tools as T
    from falsify.live.worker import agent_runner
    from tests.test_agent_loop import FakeClient, response, tool_block
    import polars as pl

    monkeypatch.setattr(T, "load_panel", lambda *a, **k: pl.DataFrame(schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}))
    script = [response([tool_block("fetch_data", {}, "a")], "tool_use")]
    note, _ = agent_runner(lambda: FakeClient(script))("Do past winners keep winning?")
    assert note["run"]["stop_reason"] == "repeated_error"
    assert "no rows matched" in note["tool_errors"][0]["error"]
