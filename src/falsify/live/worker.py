"""One live run at a time, off the request thread.

Serial on purpose. The server is a 512 MB free instance, and every run loads the
whole price panel; two at once is how it runs out of memory. A visitor whose
run is queued sees "queued" and waits a minute, which is the honest cost of a
free demo.
"""
from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable

from falsify.live.limits import RUN_MAX_OUTPUT_TOKENS, RUN_MAX_TURNS, RUN_TOKEN_BUDGET

# (hypothesis) -> (note_record dict, cost_usd). Injected so tests never call the API.
Runner = Callable[[str], "tuple[dict, float]"]


def agent_runner(client_factory: Callable[[], Any]) -> Runner:
    """The real runner: the same loop, provenance check and Note as the CLI."""

    def _run(hypothesis: str) -> tuple[dict, float]:
        from falsify.agent.loop import RunConfig, run
        from falsify.agent.provenance import check_run
        from falsify.agent.session import Session
        from falsify.demo import note_record
        from falsify.notes import from_run

        cfg = RunConfig(
            max_turns=RUN_MAX_TURNS,
            max_total_tokens=RUN_TOKEN_BUDGET,
            max_output_tokens=RUN_MAX_OUTPUT_TOKENS,
        )
        session = Session()
        result = run(hypothesis, client_factory(), session=session, config=cfg)
        note = from_run(result, hypothesis=hypothesis,
                        provenance_report=check_run(result), session=session)
        return note_record(note), float(result.cost_usd)

    return _run


class Worker:
    def __init__(self, store: Any, runner: Runner):
        self.store = store
        self.runner = runner
        self.q: "queue.Queue[str]" = queue.Queue()
        self._t = threading.Thread(target=self._loop, daemon=True, name="live-worker")
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._t.start()
            self._started = True

    def submit(self, run_id: str) -> None:
        self.q.put(run_id)

    def process_one(self, run_id: str) -> None:
        """Run one queued run to completion. Never raises: a failure is a row."""
        row = self.store.get(run_id)
        if row is None:
            return
        self.store.set_status(run_id, "running")
        try:
            note, cost = self.runner(row["hypothesis"])
            self.store.finish(run_id, note, cost)
        except Exception as exc:  # noqa: BLE001 - every failure must land on the row
            traceback.print_exc()
            # Keep the reservation: an exception after the first model call has
            # still spent money, and how much is unknown.
            self.store.finish(run_id, None, row["cost_usd"],
                              error=f"{type(exc).__name__}: {str(exc)[:300]}")

    def _loop(self) -> None:
        while True:
            rid = self.q.get()
            try:
                self.process_one(rid)
            finally:
                self.q.task_done()
