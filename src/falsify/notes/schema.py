"""The Note record, and the one rule that decides whether it can be published.

A research run produces two very different things, and the whole of this module
is the decision to keep them apart:

  * NUMBERS the pipeline computed. Sharpe, the two probabilities, MinTRL,
    invested days, the trial count they were deflated at. These are checkable
    by anyone with the repo and the data.
  * PROSE the model wrote about those numbers. The model's reading, which is
    the least reliable part of the run.

The tempting shortcut is to store the prose and call it the result. Run 1 of
this agent is the argument against: every number in its note was real and
traceable, and it still tabulated a probability in a column labelled "Deflated
Sharpe" and reasoned about the comparison. Storing that as THE artifact means
the demo serves an error with a provenance tick beside it.

So a Note stores `metrics` and `statistics` as structured data and the prose as
commentary alongside them. Module 7 renders the table from the data, never by
parsing the prose, and says on the page which is which. That is also the honest
answer to "how do I know this is not just an LLM writing plausible text?": the
table is the pipeline's, the commentary is the model's, and the reader is told.

THE PUBLISHABLE RULE

    A note is publishable only if provenance passed AND the run completed AND
    analyze_results was called.

Each of the three has already been violated by a real run in this project:

  * Provenance passed. Otherwise the note contains a number nobody can check.
  * The run completed (stop_reason == "end_turn"). A run cut short by the turn
    cap or the token budget has a partial answer that reads exactly like a
    finished one.
  * analyze_results was called. A conclusion drawn from a raw Sharpe is
    worthless however confident it sounds.

`publishable` is a COMPUTED PROPERTY, not a stored flag, so a caller cannot
assert it. It is also not a filter: an unpublishable note is still stored, and
stored as failed. Deleting failures is how an eval suite becomes a highlight
reel, and this project's entire pitch is that the failures are shown. The count
of unpublishable notes is itself an honest number to put on the page.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

# stop_reason values other than this one mean a guard in agent/loop.py fired.
COMPLETED_STOP_REASON = "end_turn"

# A conclusion that never passed through this tool rests on an unadjusted
# Sharpe ratio, whatever the prose claims.
REQUIRED_TOOL = "analyze_results"


class NoteError(Exception):
    """A Note was asked to hold something incoherent."""


@dataclass(frozen=True)
class ProvenanceVerdict:
    """What agent/provenance.py concluded about the prose.

    Stored with the note rather than recomputed on read. Recomputing needs the
    full transcript and the tool schemas as they were at the time; a schema
    edit six weeks later would silently change the verdict on an old note,
    which is the opposite of what a durable record is for.
    """

    ok: bool
    checked: int
    verified: int
    exempt: int = 0
    unverified: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.checked < 0 or self.verified < 0 or self.exempt < 0:
            raise NoteError("provenance counts cannot be negative")
        if self.verified > self.checked:
            raise NoteError(
                f"provenance verified={self.verified} exceeds checked={self.checked}"
            )
        # The two ways the verdict could lie about itself, both refused.
        if self.ok and self.unverified:
            raise NoteError(
                f"provenance claims ok with {len(self.unverified)} unverified literals"
            )
        if not self.ok and not self.unverified:
            raise NoteError("provenance claims failure with nothing unverified")

    @property
    def coverage(self) -> float:
        return 1.0 if self.checked == 0 else self.verified / self.checked

    @classmethod
    def from_report(cls, report: Any) -> ProvenanceVerdict:
        """Build from an agent.provenance.ProvenanceReport."""
        return cls(
            ok=bool(report.ok),
            checked=int(report.checked),
            verified=int(report.verified),
            exempt=int(report.exempt),
            unverified=tuple(str(u) for u in report.unverified),
        )


@dataclass(frozen=True)
class RunMetadata:
    """What the run was and what it cost.

    Cost and token counts are here because a result that cannot say what it
    spent is not reproducible in the way that matters for this project, and
    because the eval suite in Module 6 will want cost per anomaly without
    rerunning anything.
    """

    model: str
    stop_reason: str
    turns: int
    tool_sequence: tuple[str, ...] = ()
    n_tool_errors: int = 0
    n_backtests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not self.stop_reason:
            raise NoteError("stop_reason is required: it is how a cut-short run is told apart")
        if self.turns < 0:
            raise NoteError("turns cannot be negative")
        if self.cost_usd < 0:
            raise NoteError("cost_usd cannot be negative")

    @property
    def completed(self) -> bool:
        return self.stop_reason == COMPLETED_STOP_REASON

    @property
    def called_required_tool(self) -> bool:
        return REQUIRED_TOOL in self.tool_sequence

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True)
class BacktestRecord:
    """One backtest, as data.

    `metrics` is the engine's summary over the invested window. `statistics` is
    what analyze_results returned AT THE FINAL TRIAL COUNT, which is not
    necessarily what the model was shown: an agent that analyses its first
    backtest and then runs two more was given a row deflated for one trial when
    the honest number turned out to be three. Recomputing here means the stored
    record can never be more flattering than the run deserves.

    `analysis_error` is set, and `statistics` left empty, when the recompute
    failed. An empty statistics dict is why such a note is unpublishable: there
    is no deflated figure to show.
    """

    handle: str
    feature: str
    n_buckets: int
    long_short: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    analysis_error: str | None = None

    def __post_init__(self) -> None:
        if not self.handle:
            raise NoteError("a backtest record needs its handle")
        if self.statistics and self.analysis_error:
            raise NoteError(
                f"{self.handle}: has both statistics and an analysis_error; "
                "one of them is wrong"
            )

    @property
    def variant(self) -> str:
        """How this backtest is named in a table. Built, never stored."""
        side = "long/short" if self.long_short else "long-only"
        return f"{self.feature}, {self.n_buckets} buckets, {side}"

    @property
    def analysed(self) -> bool:
        return bool(self.statistics)

    @property
    def sharpe(self) -> float | None:
        v = self.metrics.get("sharpe")
        return None if v is None else float(v)

    @property
    def deflated_probability(self) -> float | None:
        """P(this beats the best of n_trials worthless strategies).

        Named for what it IS. The field it reads is called
        prob_beats_best_of_n_trials for the same reason: a key called
        "deflated_sharpe" invites a reader to put a number in [0,1] beside an
        actual Sharpe in one column, which is exactly what run 1 did.
        """
        v = self.statistics.get("prob_beats_best_of_n_trials")
        return None if v is None else float(v)


@dataclass(frozen=True)
class Note:
    """One research run, durable.

    Construct with `from_run`. Building one by hand is legitimate for tests and
    for backfilling a historical run, but the fields must agree with each other
    or __post_init__ refuses them.
    """

    hypothesis: str
    prose: str
    backtests: tuple[BacktestRecord, ...]
    provenance: ProvenanceVerdict
    run: RunMetadata
    assumptions: dict[str, Any] = field(default_factory=dict)
    note_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: dt.datetime = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )
    # Set by Module 6 so a re-run of the same anomaly overwrites rather than
    # accumulating. None for an ad-hoc run, which always inserts.
    eval_key: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.hypothesis.strip():
            raise NoteError(
                "the hypothesis is required, verbatim: a note whose question "
                "has been paraphrased cannot be audited against its run"
            )
        if not isinstance(self.backtests, tuple):
            raise NoteError("backtests must be a tuple so the record stays immutable")
        if self.created_at.tzinfo is None:
            raise NoteError(
                "created_at must be timezone-aware; a naive timestamp is "
                "ambiguous the moment it crosses a machine"
            )

    # -- the rule ----------------------------------------------------------
    @property
    def unpublishable_reasons(self) -> tuple[str, ...]:
        """Why this note cannot be published, in the order worth reading.

        Returning reasons rather than a bare False is the point: Module 7
        displays the count of unpublishable notes, and a count with no reasons
        attached is a number nobody can act on.
        """
        reasons: list[str] = []
        if not self.provenance.ok:
            n = len(self.provenance.unverified)
            reasons.append(
                f"provenance failed: {n} numeral{'' if n == 1 else 's'} in the "
                f"prose appear{'s' if n == 1 else ''} in no tool result"
            )
        if not self.run.completed:
            reasons.append(
                f"the run did not complete: a guard fired ({self.run.stop_reason}), "
                "so the answer is partial"
            )
        if not self.run.called_required_tool:
            reasons.append(
                f"{REQUIRED_TOOL} was never called: any conclusion rests on an "
                "unadjusted Sharpe ratio"
            )
        if self.backtests and not any(b.analysed for b in self.backtests):
            reasons.append(
                "no backtest carries statistics: there is no deflated figure to show"
            )
        return tuple(reasons)

    @property
    def publishable(self) -> bool:
        """Computed, never stored as an assertion, so it cannot be forged."""
        return not self.unpublishable_reasons

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe. `publishable` is included for readers, not for reloading.

        from_dict ignores it and recomputes, so a hand-edited row claiming to
        be publishable does not become publishable.
        """
        return {
            "schema_version": self.schema_version,
            "note_id": self.note_id,
            "eval_key": self.eval_key,
            "created_at": self.created_at.isoformat(),
            "hypothesis": self.hypothesis,
            "prose": self.prose,
            "backtests": [dataclasses.asdict(b) for b in self.backtests],
            "provenance": dataclasses.asdict(self.provenance),
            "run": dataclasses.asdict(self.run),
            "assumptions": dict(self.assumptions),
            "publishable": self.publishable,
            "unpublishable_reasons": list(self.unpublishable_reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Note:
        """Rebuild from to_dict output. The round trip must be exact."""
        prov = dict(data["provenance"])
        prov["unverified"] = tuple(prov.get("unverified", ()))
        meta = dict(data["run"])
        meta["tool_sequence"] = tuple(meta.get("tool_sequence", ()))
        created = data["created_at"]
        return cls(
            hypothesis=data["hypothesis"],
            prose=data["prose"],
            backtests=tuple(BacktestRecord(**b) for b in data.get("backtests", [])),
            provenance=ProvenanceVerdict(**prov),
            run=RunMetadata(**meta),
            assumptions=dict(data.get("assumptions", {})),
            note_id=data["note_id"],
            created_at=(
                created
                if isinstance(created, dt.datetime)
                else dt.datetime.fromisoformat(created)
            ),
            eval_key=data.get("eval_key"),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


def from_run(
    result: Any,
    hypothesis: str,
    provenance_report: Any,
    session: Any = None,
    eval_key: str | None = None,
    extra_assumptions: dict[str, Any] | None = None,
) -> Note:
    """Build a Note from a finished run.

    Args:
        result: an agent.loop.RunResult.
        hypothesis: the question as it was asked, verbatim.
        provenance_report: what agent.provenance.check_run returned for it.
        session: the run's Session. Defaults to `result.session`.
        eval_key: Module 6's anomaly slug, when this run is an eval case.
        extra_assumptions: anything else in force that a future reader would
            need to interpret the numbers.

    Returns:
        Note. Check `publishable` before quoting anything in it.

    THE STATISTICS ARE RECOMPUTED HERE, at `session.n_backtests`, rather than
    read out of the transcript. An assumption that is not stored with the
    result becomes invisible the moment the result is quoted, and the trial
    count is the assumption that moves every deflated figure in the table.
    """
    from falsify.agent.tools import TRIAL_VARIANCE, ToolError, analyze_results

    session = session if session is not None else getattr(result, "session", None)
    if session is None:
        raise NoteError("from_run needs the run's Session to rebuild the statistics")

    n_trials = session.n_backtests
    records: list[BacktestRecord] = []
    for handle in session.handles("backtest"):
        summary = dict(session.summary(handle))
        stats: dict[str, Any] = {}
        error: str | None = None
        try:
            stats = analyze_results(session, handle, n_trials=n_trials)
        except ToolError as exc:
            error = str(exc)
        records.append(
            BacktestRecord(
                handle=handle,
                feature=str(summary.get("feature", "")),
                n_buckets=int(summary.get("n_buckets", 0)),
                long_short=bool(summary.get("long_short", False)),
                metrics=summary,
                statistics=stats,
                analysis_error=error,
            )
        )

    calls = list(getattr(result, "tool_calls", []))
    meta = RunMetadata(
        model=str(getattr(result, "model", "")) or _model_of(result),
        stop_reason=str(result.stop_reason),
        turns=int(result.turns),
        # Successful calls only: a tool that errored produced nothing, and the
        # publishable rule asks whether analyze_results RAN, not whether the
        # model tried it.
        tool_sequence=tuple(c["name"] for c in calls if c.get("ok")),
        n_tool_errors=sum(1 for c in calls if not c.get("ok")),
        n_backtests=n_trials,
        input_tokens=int(getattr(result, "input_tokens", 0)),
        output_tokens=int(getattr(result, "output_tokens", 0)),
        cache_read_tokens=int(getattr(result, "cache_read_tokens", 0)),
        cache_write_tokens=int(getattr(result, "cache_write_tokens", 0)),
        cost_usd=round(float(getattr(result, "cost_usd", 0.0)), 6),
    )

    assumptions: dict[str, Any] = {
        "trial_variance": TRIAL_VARIANCE,
        "trial_variance_is_assumed": True,
        "n_trials_for_deflation": n_trials,
        "prices_are_split_adjusted_only": True,
        "returns_are_price_returns_not_total_returns": True,
        "benchmark_tool_available": False,
        "quality_gate_is_point_in_time": False,
        # THE ONE THAT MATTERS MOST, and the reason it is a stored field rather
        # than a README line. `fetch_data` has no universe argument: it loads
        # every ingested ticker, which is the CURRENT S&P 500 membership.
        # Companies that were dropped or delisted are absent, so every return
        # in this note is overstated by survivorship bias.
        #
        # The project measures that bias — 20.18% against 10.02% on a common
        # window, Sharpe 0.68 against 0.45 — but `stats/survivorship.py` is
        # reachable only from `scripts/run_survivorship.py`, never from the
        # agent. A note quoted without this line looks like a point-in-time
        # result and is not one.
        "universe": "all ingested tickers (CURRENT constituents)",
        "universe_is_point_in_time": False,
        "survivorship_bias_present": True,
    }
    assumptions.update(extra_assumptions or {})

    return Note(
        hypothesis=hypothesis,
        prose=str(getattr(result, "answer", "") or ""),
        backtests=tuple(records),
        provenance=ProvenanceVerdict.from_report(provenance_report),
        run=meta,
        assumptions=assumptions,
        eval_key=eval_key,
    )


def _model_of(result: Any) -> str:
    """The model name, which RunResult does not carry as a field."""
    from falsify.agent.loop import MODEL

    return MODEL
