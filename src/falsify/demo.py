"""Module 7: freeze the evidence into one file the public page can serve.

The demo is a static page. It makes no Claude call, holds no database
connection and runs no backtest: every number on it was produced earlier by the
pipeline, stored in Postgres, and frozen here into `web/data/demo.json`. A page
that could compute would be a page that could compute something different from
what the repo says it computed.

`build_snapshot` is pure, so it is tested without a database. It REFUSES,
rather than warns, in every case where the file it would write could mislead:

  stale registry   a row scored against a prediction that has since been
                   edited. The same rule as `run_evals.py --check`, applied
                   again at the last moment before the evidence leaves the repo.
  mixed rules      rows held to different scoring-rule versions in one table.
  missing rows     fewer than six anomalies times two universes. A table with a
                   hole in it is read as a table with nothing to hide there.
  unknown verdict  anything outside the four outcomes.

Unpublishable notes and failing verdicts are exported like any other. The page
shows them with their reasons; dropping them here is exactly how an eval suite
turns into a highlight reel, and it would happen in the one file nobody reads.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Iterable

from falsify.eval.registry import ANOMALIES, Anomaly, registry_digest
from falsify.eval.score import (
    DEFLATION_THRESHOLD,
    EMBARRASSMENT_MULTIPLE,
    FDR_ALPHA,
    SCORING_RULE_VERSION,
    VERDICTS,
)
from falsify.notes.schema import Note

SNAPSHOT_VERSION = 1
UNIVERSES: tuple[str, ...] = ("current", "point_in_time")

# Fields copied from a stored eval row to the page. `detail` is deliberately
# not copied whole: it duplicates these, and anything on the page should be
# something the page renders.
_ROW_FIELDS = (
    "verdict",
    "reasons",
    "realised_sharpe",
    "oriented_sharpe",
    "published_sharpe",
    "p_value",
    "p_value_adjusted",
    "deflated_psr",
    "n_invested_days",
    "history_days",
    "min_history_days",
    "n_trials",
    "run_at",
)


class ExportError(Exception):
    """The snapshot would mislead, so it is not written."""


def _clean(v: Any) -> Any:
    """JSON-safe. NaN and infinity become null rather than invalid JSON, and a
    timestamp becomes ISO-8601 with its offset."""
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _check_rows(
    rows: list[dict], anomalies: tuple[Anomaly, ...], current_sha: str, allow_partial: bool
) -> None:
    stale = sorted({(r["eval_key"], r["universe"]) for r in rows if r["registry_sha"] != current_sha})
    if stale:
        raise ExportError(
            f"{len(stale)} stored verdict(s) were scored against an edited registry "
            f"(current sha {current_sha[:12]}): {stale[:4]}. Re-run "
            "`python scripts/run_evals.py --compare --store` before exporting."
        )
    versions = {r["scoring_rule_version"] for r in rows}
    if versions and versions != {SCORING_RULE_VERSION}:
        raise ExportError(
            f"stored verdicts carry scoring-rule versions {sorted(versions)}; this code is "
            f"version {SCORING_RULE_VERSION}. One table, one standard: re-run and store."
        )
    bad = sorted({r["verdict"] for r in rows} - set(VERDICTS))
    if bad:
        raise ExportError(f"unknown verdict(s) {bad}; the four outcomes are {VERDICTS}")

    have = {(r["eval_key"], r["universe"]) for r in rows}
    want = {(a.key, u) for a in anomalies for u in UNIVERSES}
    unknown = sorted(have - want)
    if unknown:
        raise ExportError(
            f"stored verdicts for anomalies or universes not in the registry: {unknown}"
        )
    missing = sorted(want - have)
    if missing and not allow_partial:
        raise ExportError(
            f"{len(missing)} of {len(want)} verdicts are missing: {missing[:4]}. Run "
            "`python scripts/run_evals.py --compare --store`, or pass --allow-partial "
            "and the page will say which are absent."
        )


def _tally(rows: Iterable[dict]) -> dict[str, int]:
    """Every verdict, including the zeroes. A missing key reads as 'not
    applicable' when it means 'none this time'."""
    out = {v: 0 for v in VERDICTS}
    for r in rows:
        out[r["verdict"]] += 1
    return out


def _note_record(n: Note) -> dict[str, Any]:
    """One note for the page. `publishable` is the recomputed property, never a
    stored column, so an edited row cannot publish itself on the way out."""
    d = n.to_dict()
    return _clean(
        {
            "note_id": n.note_id,
            "eval_key": n.eval_key,
            "created_at": n.created_at,
            "hypothesis": n.hypothesis,
            "prose": n.prose,
            "publishable": n.publishable,
            "unpublishable_reasons": list(n.unpublishable_reasons),
            "backtests": [
                {
                    "variant": b.variant,
                    "metrics": b.metrics,
                    "statistics": b.statistics,
                    "analysis_error": b.analysis_error,
                }
                for b in n.backtests
            ],
            "provenance": d["provenance"],
            "run": {
                "model": n.run.model,
                "stop_reason": n.run.stop_reason,
                "turns": n.run.turns,
                "tool_sequence": list(n.run.tool_sequence),
                "n_backtests": n.run.n_backtests,
                "total_tokens": n.run.total_tokens,
                "cost_usd": n.run.cost_usd,
            },
            "assumptions": n.assumptions,
        }
    )


def _check_survivorship(s: dict | None) -> None:
    if s is None:
        return
    for k in ("gate", "window", "metrics", "bound"):
        if k not in s:
            raise ExportError(f"survivorship file has no {k!r}; regenerate it with --save")
    for k in ("sharpe_current", "sharpe_pit", "total_return_current", "total_return_pit"):
        if k not in s["metrics"]:
            raise ExportError(f"survivorship metrics missing {k!r}")
    if s.get("forced_past_coverage_gate"):
        raise ExportError(
            "the survivorship figure was produced with --force past an incomplete-coverage "
            "gate. It under-measures the bias and the page would quote it as the headline."
        )


def build_snapshot(
    eval_rows: list[dict],
    notes: list[Note],
    *,
    survivorship: dict | None = None,
    code_commit: str | None = None,
    trial_variance: float | None = None,
    anomalies: tuple[Anomaly, ...] = ANOMALIES,
    allow_partial: bool = False,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Everything the page renders, as one JSON-safe dict.

    Args:
        eval_rows: `eval.store.fetch_all()` output.
        notes: `notes.store.fetch_all()` output, publishable or not.
        survivorship: `run_survivorship.py --save` output, or None.
        code_commit: the git commit the stored rows are claimed to come from.
        trial_variance: the assumption behind every deflated figure.
        anomalies: the registry; the default is the frozen suite.
        allow_partial: export with verdicts missing, and say which.
        now: injected for tests.

    Raises:
        ExportError: any condition in the module docstring.
    """
    sha = registry_digest(anomalies)
    _check_rows(eval_rows, anomalies, sha, allow_partial)
    _check_survivorship(survivorship)

    by_key = {(r["eval_key"], r["universe"]): r for r in eval_rows}
    table = []
    for a in anomalies:
        results = {}
        for u in UNIVERSES:
            r = by_key.get((a.key, u))
            results[u] = None if r is None else {f: r.get(f) for f in _ROW_FIELDS}
        cur, pit = results["current"], results["point_in_time"]
        gap = (
            None
            if not cur or not pit or cur["oriented_sharpe"] is None or pit["oriented_sharpe"] is None
            else cur["oriented_sharpe"] - pit["oriented_sharpe"]
        )
        table.append(
            {
                "key": a.key,
                "feature": a.feature,
                "direction": a.direction,
                "published_sharpe": a.published_sharpe,
                "min_history_days": a.min_history_days,
                "citation": a.citation,
                "hypothesis": a.hypothesis,
                "sharpe_source": a.sharpe_source,
                "caveat": a.caveat,
                "results": results,
                "survivorship_sharpe_gap": gap,
            }
        )

    rendered_notes = sorted(
        (_note_record(n) for n in notes), key=lambda x: x["created_at"], reverse=True
    )
    n_pub = sum(1 for n in rendered_notes if n["publishable"])
    run_dates = [r["run_at"] for r in eval_rows if r.get("run_at") is not None]

    return _clean(
        {
            "snapshot_version": SNAPSHOT_VERSION,
            "generated_at": (now or dt.datetime.now(dt.timezone.utc)).replace(microsecond=0),
            "code_commit": code_commit,
            "registry_sha": sha,
            "scoring": {
                "rule_version": SCORING_RULE_VERSION,
                "deflation_threshold": DEFLATION_THRESHOLD,
                "fdr_alpha": FDR_ALPHA,
                "embarrassment_multiple": EMBARRASSMENT_MULTIPLE,
                "trial_variance_assumed": trial_variance,
            },
            "evals": {
                "universes": list(UNIVERSES),
                "last_run_at": max(run_dates) if run_dates else None,
                "tally": {
                    u: _tally(r for r in eval_rows if r["universe"] == u) for u in UNIVERSES
                },
                "missing": sorted(
                    f"{a.key}/{u}" for a in anomalies for u in UNIVERSES
                    if (a.key, u) not in by_key
                ),
                "anomalies": table,
            },
            "notes": {
                "counts": {
                    "total": len(rendered_notes),
                    "publishable": n_pub,
                    "unpublishable": len(rendered_notes) - n_pub,
                },
                "items": rendered_notes,
            },
            "survivorship": survivorship,
        }
    )


# Public name for the live API, which renders a visitor's note the same way.
note_record = _note_record

__all__ = ["ExportError", "SNAPSHOT_VERSION", "UNIVERSES", "build_snapshot", "note_record"]
