"""The live API. Four routes, and the admin one needs a token.

    GET  /health            is the server up, and how many runs are left today
    POST /runs              submit a hypothesis; returns a run id
    GET  /runs/{run_id}     poll one run; the id is the only way to read it
    GET  /admin/runs        every run, newest first; Authorization: Bearer ADMIN_TOKEN

WHO SEES WHAT. A run id is a random UUID returned only to the visitor who
submitted it, so reading a run requires having submitted it. There is no route
that lists runs without the admin token. The owner sees everything, including
the optional name a visitor leaves and a hashed visitor id; the raw IP is
never stored.

Run locally:  uvicorn falsify.live.app:app --reload
"""
from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from falsify.live import limits as L
from falsify.live.store import MemoryStore, PgStore
from falsify.live.worker import Runner, Worker, agent_runner


class Submit(BaseModel):
    hypothesis: str
    name: str | None = None


def _public(row: dict) -> dict:
    """What a visitor may see about their own run. No visitor id, no name echo,
    and no raw tool errors: those can name tables and hosts, and are for the owner."""
    note = row["note"]
    if isinstance(note, dict):
        note = {k: v for k, v in note.items() if k != "tool_errors"}
    return {
        "run_id": row["run_id"],
        "status": row["status"],
        "hypothesis": row["hypothesis"],
        "created_at": row["created_at"].isoformat(),
        "note": note,
        "error": "The run failed on the server. Try again later." if row["error"] else None,
    }


def _admin(row: dict) -> dict:
    return {
        **{k: v for k, v in row.items() if k not in ("created_at", "finished_at")},
        "created_at": row["created_at"].isoformat(),
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
    }


def _ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def create_app(
    store: Any,
    runner: Runner,
    admin_token: str,
    salt: str,
    origins: list[str],
    lim: L.Limits = L.Limits(),
    start_worker: bool = True,
) -> FastAPI:
    if len(admin_token) < 16:
        raise RuntimeError("ADMIN_TOKEN must be at least 16 characters")
    app = FastAPI(title="falsify live", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    store.apply_schema()
    store.fail_unfinished("the server restarted before this run finished")
    worker = Worker(store, runner)
    if start_worker:
        worker.start()
    app.state.worker = worker
    app.state.store = store

    @app.get("/health")
    def health(request: Request) -> dict:
        today = store.today(L.visitor_id(_ip(request), salt))
        return {"ok": True, "runs_left_for_you_today": L.runs_left(lim, today)}

    @app.post("/runs", status_code=202)
    def submit(body: Submit, request: Request) -> dict:
        try:
            hyp = L.clean_hypothesis(body.hypothesis)
            vid = L.visitor_id(_ip(request), salt)
            # Check-then-insert under one lock, so two simultaneous requests
            # cannot both read "one run left" and both start.
            with _submit_lock:
                L.check(lim, store.today(vid))
                rid = store.create(vid, L.clean_name(body.name), hyp)
        except L.Refused as e:
            raise HTTPException(status_code=e.status, detail=str(e)) from None
        worker.submit(rid)
        return {"run_id": rid, "status": "queued"}

    @app.get("/runs/{run_id}")
    def poll(run_id: str) -> dict:
        row = store.get(run_id)
        if row is None:
            raise HTTPException(404, "No such run.")
        return _public(row)

    @app.get("/admin/runs")
    def admin(authorization: str = Header(default="")) -> dict:
        given = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), admin_token.encode()):
            raise HTTPException(401, "Not authorised.")
        rows = store.list_all()
        return {
            "runs": [_admin(r) for r in rows],
            "spent_today_usd": store.today("").spend_usd,
            "limits": lim.__dict__,
        }

    return app


import threading  # noqa: E402

_submit_lock = threading.Lock()


def _from_env() -> FastAPI:
    from anthropic import Anthropic

    origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    return create_app(
        store=PgStore(os.environ["DATABASE_URL"]),
        runner=agent_runner(Anthropic),
        admin_token=os.environ["ADMIN_TOKEN"],
        salt=os.environ["VISITOR_SALT"],
        origins=origins or ["http://localhost:3000"],
        lim=L.Limits(
            daily_usd_cap=float(os.environ.get("LIVE_DAILY_USD_CAP", "2.0")),
            daily_run_cap=int(os.environ.get("LIVE_DAILY_RUN_CAP", "25")),
            per_visitor_cap=int(os.environ.get("LIVE_PER_VISITOR_CAP", "3")),
        ),
    )


def __getattr__(name: str):  # lazy, so importing the module in tests needs no env
    if name == "app":
        return _from_env()
    raise AttributeError(name)


__all__ = ["MemoryStore", "PgStore", "create_app"]
