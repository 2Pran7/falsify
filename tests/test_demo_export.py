"""Module 7: the frozen snapshot the public page serves.

No database. `build_snapshot` is pure, so every refusal is tested on
hand-built rows. The refusals are the point of the module: the page cannot
recompute anything, so whatever this function lets through is what the world
sees.
"""
from __future__ import annotations

import datetime as dt
import json
import math

import pytest

from falsify.demo import ExportError, build_snapshot
from falsify.eval.registry import ANOMALIES, registry_digest
from falsify.eval.score import SCORING_RULE_VERSION
from tests.test_notes_schema import a_note, bad_provenance

SHA = registry_digest()
T0 = dt.datetime(2026, 9, 22, 21, 0, tzinfo=dt.timezone.utc)


def _row(key: str, universe: str, **over) -> dict:
    base = {
        "eval_key": key,
        "universe": universe,
        "run_at": T0,
        "registry_sha": SHA,
        "scoring_rule_version": SCORING_RULE_VERSION,
        "verdict": "fail",
        "reasons": ["the oriented spread is negative"],
        "direction": 1,
        "realised_sharpe": -0.4,
        "oriented_sharpe": -0.4,
        "published_sharpe": 0.5,
        "p_value": 0.8,
        "p_value_adjusted": 0.9,
        "deflated_psr": 0.1,
        "n_invested_days": 250,
        "history_days": 500,
        "min_history_days": 252,
        "n_trials": 5,
        "caveat": "",
        "detail": {},
    }
    base.update(over)
    return base


def _all_rows(**over) -> list[dict]:
    return [_row(a.key, u, **over) for a in ANOMALIES for u in ("current", "point_in_time")]


# --- the shape -------------------------------------------------------------


def test_twelve_rows_become_six_anomalies_with_two_results_each():
    snap = build_snapshot(_all_rows(), [])
    table = snap["evals"]["anomalies"]
    assert [a["key"] for a in table] == [a.key for a in ANOMALIES]
    assert all(set(a["results"]) == {"current", "point_in_time"} for a in table)
    assert snap["evals"]["missing"] == []


def test_reasons_travel_with_every_verdict():
    """The page renders the reasons, not the tally. A row without them is a
    number nobody can act on."""
    snap = build_snapshot(_all_rows(), [])
    for a in snap["evals"]["anomalies"]:
        for r in a["results"].values():
            assert r["reasons"] == ["the oriented spread is negative"]


def test_the_tally_lists_every_verdict_including_zeroes():
    snap = build_snapshot(_all_rows(), [])
    assert snap["evals"]["tally"]["current"] == {
        "pass": 0, "partial": 0, "fail": 6, "insufficient_data": 0,
    }


def test_the_registry_prose_is_carried_so_the_page_needs_no_python():
    snap = build_snapshot(_all_rows(), [])
    mom = snap["evals"]["anomalies"][0]
    assert mom["citation"].startswith("Jegadeesh")
    assert mom["caveat"] and mom["hypothesis"]
    assert snap["registry_sha"] == SHA


def test_the_survivorship_gap_per_anomaly_is_current_minus_pit():
    rows = _all_rows()
    for r in rows:
        if r["eval_key"] == "momentum_12_1":
            r["oriented_sharpe"] = 0.648 if r["universe"] == "current" else 0.461
    snap = build_snapshot(rows, [])
    assert snap["evals"]["anomalies"][0]["survivorship_sharpe_gap"] == pytest.approx(0.187)


def test_the_output_is_strict_json():
    """NaN is valid Python and invalid JSON; a browser's JSON.parse rejects it
    and the whole page goes blank."""
    rows = _all_rows()
    rows[0]["p_value"] = float("nan")
    rows[1]["deflated_psr"] = float("inf")
    snap = build_snapshot(rows, [a_note()], now=T0)
    text = json.dumps(snap, allow_nan=False)
    back = json.loads(text)
    assert back["evals"]["anomalies"][0]["results"]["current"]["p_value"] is None
    assert back["generated_at"] == "2026-09-22T21:00:00+00:00"


# --- the refusals ----------------------------------------------------------


def test_refuses_a_row_scored_against_an_edited_registry():
    rows = _all_rows()
    rows[3]["registry_sha"] = "0" * 64
    with pytest.raises(ExportError, match="edited registry"):
        build_snapshot(rows, [])


def test_refuses_mixed_scoring_rule_versions():
    rows = _all_rows()
    rows[0]["scoring_rule_version"] = SCORING_RULE_VERSION + 1
    with pytest.raises(ExportError, match="one standard"):
        build_snapshot(rows, [])


def test_refuses_a_table_with_a_hole_in_it():
    rows = _all_rows()[:-1]
    with pytest.raises(ExportError, match="missing"):
        build_snapshot(rows, [])


def test_allow_partial_exports_the_hole_and_names_it():
    rows = _all_rows()[:-1]
    snap = build_snapshot(rows, [], allow_partial=True)
    last = ANOMALIES[-1].key
    assert snap["evals"]["missing"] == [f"{last}/point_in_time"]
    assert snap["evals"]["anomalies"][-1]["results"]["point_in_time"] is None


def test_refuses_an_unknown_verdict():
    rows = _all_rows()
    rows[0]["verdict"] = "promising"
    with pytest.raises(ExportError, match="unknown verdict"):
        build_snapshot(rows, [])


def test_refuses_a_row_for_an_anomaly_not_in_the_registry():
    rows = _all_rows() + [_row("lottery_max", "current")]
    with pytest.raises(ExportError, match="not in the registry"):
        build_snapshot(rows, [])


# --- notes -----------------------------------------------------------------


def test_unpublishable_notes_are_exported_with_their_reasons():
    good = a_note()
    bad = a_note(provenance=bad_provenance())
    snap = build_snapshot(_all_rows(), [good, bad])
    assert snap["notes"]["counts"] == {"total": 2, "publishable": 1, "unpublishable": 1}
    failed = [n for n in snap["notes"]["items"] if not n["publishable"]]
    assert failed[0]["unpublishable_reasons"][0].startswith("provenance failed")


def test_publishable_is_recomputed_not_read():
    """A Note whose provenance failed is unpublishable however it was stored;
    the export reads the property, never a column."""
    bad = a_note(provenance=bad_provenance())
    snap = build_snapshot(_all_rows(), [bad])
    assert snap["notes"]["items"][0]["publishable"] is False


def test_note_backtests_carry_the_deflated_figure_under_its_real_name():
    snap = build_snapshot(_all_rows(), [a_note()])
    bt = snap["notes"]["items"][0]["backtests"][0]
    assert bt["statistics"]["prob_beats_best_of_n_trials"] == pytest.approx(0.4218)
    assert "deflated_sharpe" not in bt["statistics"]


# --- survivorship ----------------------------------------------------------


def _surv(**over) -> dict:
    base = {
        "generated_at": "2026-09-22T21:00:00+00:00",
        "window": {"first": "2025-06-02", "last": "2026-05-01", "invested_days": 239},
        "gate": "point_in_time",
        "n_tickers_gated": 3,
        "forced_past_coverage_gate": False,
        "bound": "lower",
        "metrics": {
            "sharpe_current": 0.68, "sharpe_pit": 0.45, "sharpe_gap": 0.23,
            "total_return_current": 0.2018, "total_return_pit": 0.1002,
        },
    }
    base.update(over)
    return base


def test_the_survivorship_figure_is_carried_through():
    snap = build_snapshot(_all_rows(), [], survivorship=_surv())
    assert snap["survivorship"]["metrics"]["sharpe_pit"] == 0.45


def test_refuses_a_survivorship_figure_forced_past_its_coverage_gate():
    """--force prints a number the audit itself calls an under-measurement.
    Printing it in a terminal is fine; quoting it as the headline is not."""
    with pytest.raises(ExportError, match="--force"):
        build_snapshot(_all_rows(), [], survivorship=_surv(forced_past_coverage_gate=True))


def test_refuses_a_survivorship_file_missing_its_metrics():
    s = _surv()
    del s["metrics"]["sharpe_pit"]
    with pytest.raises(ExportError, match="sharpe_pit"):
        build_snapshot(_all_rows(), [], survivorship=s)


def test_no_nan_anywhere_in_a_realistic_snapshot():
    snap = build_snapshot(_all_rows(), [a_note()], survivorship=_surv())

    def walk(x):
        if isinstance(x, float):
            assert not math.isnan(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(snap)
