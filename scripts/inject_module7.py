"""Injection audit for Module 7: the splice gate and the export.

Usage
-----
    python scripts/inject_module7.py

Same machinery as inject_module6.py, reused rather than copied: the clean-tree
guard, byte-exact reverts, and line-ending-adapted patterns all come from
there, because each of those was a bug paid for once already.

Every refusal `falsify.demo` makes is injected away here, one at a time. The
page cannot recompute anything, so a refusal that has never been seen to fire
is a refusal the public figures are relying on without evidence.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("inject_module6", HERE / "inject_module6.py")
m6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m6)

Q = "src/falsify/data/quality.py"
TOOLS = "src/falsify/agent/tools.py"
DEMO = "src/falsify/demo.py"
TQ = "tests/test_quality.py"
TT = "tests/test_agent_tools.py"
TD = "tests/test_demo_export.py"

INJECTIONS = [
    # --- the point-in-time gate -------------------------------------------
    ("gate cuts the genuine last close before a splice",
     Q,
     '    keep = pl.col("last_kept").is_null() | (pl.col("ts") <= pl.col("last_kept"))',
     '    keep = pl.col("last_kept").is_null() | (pl.col("ts") < pl.col("last_kept"))',
     TQ),

    ("gate cuts at the LAST defect, keeping a spliced stretch",
     Q,
     '        .group_by("ticker", maintain_order=True)\n        .first()',
     '        .group_by("ticker", maintain_order=True)\n        .last()',
     TQ),

    ("gate ignores coverage gaps",
     Q,
     "            coverage_gaps(panel, max_gap_days).select(",
     "            coverage_gaps(panel, max_gap_days).head(0).select(",
     TQ),

    ("fetch_data computes the gate and ignores it",
     TOOLS,
     "    frame, gated = truncate_suspect_tickers(frame)",
     "    _, gated = truncate_suspect_tickers(frame)",
     TT),

    ("fetch_data uses the whole-sample (lookahead) gate",
     TOOLS,
     "    frame, gated = truncate_suspect_tickers(frame)",
     "    from falsify.data.quality import drop_suspect_tickers as _d\n"
     "    frame, gated = _d(frame)\n"
     "    gated = gated.with_columns(pl.lit(0).alias(\"rows_dropped\"))",
     TT),

    ("fetch_data ignores a ragged panel end",
     TOOLS,
     "    frame, tail = trim_ragged_end(frame)",
     "    _, tail = trim_ragged_end(frame)",
     TT),

    ("ragged-end trim keeps every thin trailing date",
     Q,
     "    floor = float(counts[\"n_tickers\"].median()) * min_coverage",
     "    floor = 0.0",
     TQ),

    ("the analysis reminder is never sent",
     "src/falsify/agent/loop.py",
     "            and result.analysis_nudges < config.max_analysis_nudges",
     "            and False",
     "tests/test_agent_loop.py"),

    ("the analysis reminder repeats until the model complies",
     "src/falsify/agent/loop.py",
     "            and result.analysis_nudges < config.max_analysis_nudges",
     "            and result.analysis_nudges < 5",
     "tests/test_agent_loop.py"),

    ("the gate verdict is inverted",
     "src/falsify/agent/tools.py",
     '        "clears_confirmation_gate": bool(dsr >= CONFIRMATION_GATE),',
     '        "clears_confirmation_gate": bool(dsr < CONFIRMATION_GATE),',
     "tests/test_agent_tools.py"),

    # --- the export's refusals --------------------------------------------
    ("export accepts rows scored against an edited registry",
     DEMO, "    if stale:", "    if False:", TD),

    ("export accepts mixed scoring-rule versions",
     DEMO,
     "    if versions and versions != {SCORING_RULE_VERSION}:",
     "    if False:",
     TD),

    ("export ships a table with a hole in it",
     DEMO, "    if missing and not allow_partial:", "    if False:", TD),

    ("export reads publishable instead of recomputing it",
     DEMO, '            "publishable": n.publishable,', '            "publishable": True,', TD),

    ("export drops unpublishable notes (highlight reel)",
     DEMO,
     "(_note_record(n) for n in notes)",
     "(_note_record(n) for n in notes if n.publishable)",
     TD),

    ("NaN reaches the JSON and blanks the page",
     DEMO,
     "        return None if math.isnan(v) or math.isinf(v) else v",
     "        return v",
     TD),

    ("export quotes a survivorship figure produced with --force",
     DEMO, '    if s.get("forced_past_coverage_gate"):', "    if False:", TD),
]

if __name__ == "__main__":
    m6.INJECTIONS = INJECTIONS
    sys.exit(m6.main())
