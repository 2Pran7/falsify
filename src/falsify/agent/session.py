"""The handle store: pipeline outputs live here, the model only gets handles.

A 548-ticker two-year panel is roughly 275,000 rows. Returning one to the model
would blow the context window, cost a fortune, and hand it the raw material to
do arithmetic on. So nothing large ever reaches the model. Every tool stores its
real output here and returns an opaque HANDLE plus a summary small enough to
read. The next tool takes the handle back.

That single decision does three jobs at once, which is why it comes first:

  1. Context. The model sees a few hundred bytes per step instead of megabytes.
  2. Cost. Input tokens are the bulk of the bill and this is what bounds them.
  3. The no-computation guarantee. The model cannot average a series it has
     never seen. "The LLM decides, never computes" is not enforced by asking it
     nicely; it is enforced by never putting the numbers in front of it.

The store is per-run and in memory. Nothing here is persisted: a research run is
a single conversation, and a handle from a previous run resolving in this one
would be a correctness bug, not a feature.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

Kind = Literal["panel", "feature", "backtest", "analysis"]
KINDS: tuple[Kind, ...] = ("panel", "feature", "backtest", "analysis")

# A summary is what the model reads. Capping it is not tidiness: without a cap,
# the easy way to write a tool is to dump the frame into the summary, and the
# handle indirection quietly stops protecting anything. The cap makes that
# mistake fail loudly at the point it is made.
MAX_SUMMARY_BYTES = 2_000


class SessionError(Exception):
    """Base for handle-store failures the tool layer turns into tool errors."""


class UnknownHandle(SessionError):
    """The model referenced a handle that does not exist in this run."""


class WrongKind(SessionError):
    """The handle exists but is the wrong sort of thing for this tool."""


class SummaryTooLarge(SessionError):
    """A tool tried to put a payload-sized object into a model-visible summary."""


@dataclass(frozen=True)
class Artifact:
    """One stored pipeline output.

    payload is the real object — a Polars frame, a BacktestResult — and never
    leaves the process. summary is the JSON-safe digest the model is allowed to
    see. Keeping both on one object means there is exactly one place where the
    boundary between them is decided.
    """

    handle: str
    kind: Kind
    payload: Any
    summary: dict
    created: dt.datetime = field(default_factory=dt.datetime.now)


class Session:
    """In-memory store for one research run.

    Handles are readable (`panel_1`, `backtest_2`) on purpose. An opaque UUID
    would be marginally harder to guess, but the transcript is the thing you
    will be reading when the agent misbehaves, and `backtest_2` in a tool call
    is legible where `a3f9c1...` is not. Guessing is not the threat model:
    `get` validates against the store, so a fabricated handle fails cleanly
    rather than resolving to the wrong object.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._counts: dict[str, int] = {k: 0 for k in KINDS}

    # -- writing -----------------------------------------------------------
    def put(self, kind: Kind, payload: Any, summary: dict) -> str:
        """Store a pipeline output, return the handle the model will see.

        Raises:
            ValueError: unknown kind.
            SummaryTooLarge: the summary is not JSON-safe or exceeds the cap.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")

        try:
            encoded = json.dumps(summary, default=str)
        except (TypeError, ValueError) as exc:
            raise SummaryTooLarge(f"summary for {kind} is not JSON-safe: {exc}") from exc
        if len(encoded) > MAX_SUMMARY_BYTES:
            raise SummaryTooLarge(
                f"summary for {kind} is {len(encoded)} bytes, over the "
                f"{MAX_SUMMARY_BYTES} cap. Summarise it; do not hand the model "
                "the payload."
            )

        self._counts[kind] += 1
        handle = f"{kind}_{self._counts[kind]}"
        self._artifacts[handle] = Artifact(handle, kind, payload, summary)
        return handle

    # -- reading -----------------------------------------------------------
    def get(self, handle: str, kind: Kind | None = None) -> Artifact:
        """Resolve a handle, optionally asserting its kind.

        Raises:
            UnknownHandle: no such handle in this run.
            WrongKind: it exists but is not the kind the caller needs.

        Both errors carry what IS available, because the message goes back to
        the model as a tool error and a model that can see the valid handles
        can correct itself in one turn instead of guessing for three.
        """
        art = self._artifacts.get(handle)
        if art is None:
            # Creation order, not sorted: this message is read next to the
            # transcript, where the handles appear in the order they were made.
            known = ", ".join(self._artifacts) or "none yet"
            raise UnknownHandle(f"no handle {handle!r} in this run. Available: {known}")
        if kind is not None and art.kind != kind:
            available = ", ".join(self.handles(kind)) or f"no {kind} handles yet"
            raise WrongKind(
                f"{handle!r} is a {art.kind}, not a {kind}. Available {kind}s: {available}"
            )
        return art

    def payload(self, handle: str, kind: Kind | None = None) -> Any:
        """The real object behind a handle. Never goes to the model."""
        return self.get(handle, kind).payload

    def summary(self, handle: str, kind: Kind | None = None) -> dict:
        """The model-visible digest behind a handle."""
        return self.get(handle, kind).summary

    def handles(self, kind: Kind | None = None) -> list[str]:
        """Handles in creation order, optionally of one kind."""
        return [h for h, a in self._artifacts.items() if kind is None or a.kind == kind]

    # -- the honesty counter ------------------------------------------------
    @property
    def n_backtests(self) -> int:
        """How many backtests this run has actually executed.

        This is the trial count for deflation, and it is tracked by the store
        rather than reported by the model on purpose. The honest N includes
        every variant tried and abandoned, which is exactly the number a model
        summarising its own work has an incentive to forget. Understating N is
        how a deflated Sharpe gets quietly re-inflated, so the number comes from
        the side of the boundary that cannot be talked out of it.
        """
        return self._counts["backtest"]

    def __len__(self) -> int:
        return len(self._artifacts)

    def __iter__(self) -> Iterator[Artifact]:
        return iter(self._artifacts.values())

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}={self._counts[k]}" for k in KINDS if self._counts[k])
        return f"Session({counts or 'empty'})"
