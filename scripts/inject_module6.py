"""Injection audit for Module 6: eighteen bugs, deliberately introduced.

Usage
-----
    python scripts/inject_module6.py

A test that has never failed has not been shown to work. For each claim the
Module 6 suites make, this introduces the exact bug that claim is about,
confirms the file really changed, runs the tests, and reverts.

Reports CAUGHT only when the suite goes RED. An injection that leaves it green
is a test that proves nothing, and on the first pass one of these did exactly
that -- see the note on `untested anomalies pad the multiplicity denominator`
below.

TWO THINGS THIS REFUSES TO DO, both because it edits source files in place:

  It refuses to run on a DIRTY working tree. Every injection is reverted in a
  `finally`, so an ordinary Ctrl-C is safe -- but a hard kill, a power cut or a
  crashed interpreter can leave a mutated file behind. Starting from a clean
  tree means `git checkout -- .` always recovers, and it also means the diff
  you see afterwards is proof nothing was left behind. Starting from a dirty
  one would mix your own uncommitted work into that recovery.

  It refuses to run outside a git repository, for the same reason: without
  version control there is no recovery path at all.

SIXTH LESSON, learned here: an injection that passes may only mean the test and
the bug happen to agree on the case you chose. Injecting is not enough. Check
that the injected bug actually changes behaviour on the input the test uses.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
import os
DSN = os.getenv("DATABASE_URL", "postgresql://falsify:falsify_dev@localhost:5432/falsify")

# (label, file, old, new, test target)
INJECTIONS = [
    # --- score.py: the four rules ---------------------------------------
    ("scorer ignores `direction`",
     "src/falsify/eval/score.py",
     "    return realised * direction",
     "    return realised",
     "tests/test_eval_score.py"),

    ("deflation gate always satisfied",
     "src/falsify/eval/score.py",
     "    if measurement.deflated_psr < deflation_threshold:",
     "    if False:",
     "tests/test_eval_score.py"),

    ("a missing deflation counts as a pass",
     "src/falsify/eval/score.py",
     "    if measurement.deflated_psr is None:\n        return Score(\n            verdict=\"fail\",",
     "    if measurement.deflated_psr is None and False:\n        return Score(\n            verdict=\"fail\",",
     "tests/test_eval_score.py"),

    ("no multiple-testing correction across the suite",
     "src/falsify/eval/score.py",
     "        _reject, adj = benjamini_hochberg(pvals, alpha=fdr_alpha)",
     "        adj = list(pvals)",
     "tests/test_eval_score.py"),

    ("untested anomalies pad the multiplicity denominator",
     "src/falsify/eval/score.py",
     "    testable = [a for a in anomalies if _testable(a)]",
     "    testable = [a for a in anomalies if by_key[a.key].p_value is not None]",
     "tests/test_eval_score.py"),

    ("insufficient data reported as a failure",
     "src/falsify/eval/score.py",
     "            verdict=\"insufficient_data\",\n            reasons=(\n                f\"needs {anomaly.min_history_days} trading days",
     "            verdict=\"fail\",\n            reasons=(\n                f\"needs {anomaly.min_history_days} trading days",
     "tests/test_eval_score.py"),

    ("an implausibly large result is not downgraded",
     "src/falsify/eval/score.py",
     "    if ratio is not None and ratio > embarrassment_multiple:",
     "    if False:",
     "tests/test_eval_score.py"),

    # --- registry.py: the pre-registration -------------------------------
    ("registry digest ignores `direction`",
     "src/falsify/eval/registry.py",
     '    "direction",\n    "published_sharpe",',
     '    "published_sharpe",',
     "tests/test_eval_registry.py"),

    ("registry digest depends on tuple order",
     "src/falsify/eval/registry.py",
     "        for a in sorted(anomalies, key=lambda x: x.key)\n    ]\n    encoded = json.dumps(records, sort_keys=True",
     "        for a in anomalies\n    ]\n    encoded = json.dumps(records, sort_keys=True",
     "tests/test_eval_registry.py"),

    # --- tools.py: the universe ------------------------------------------
    ("universe mask filters PRICES instead of the signal",
     "src/falsify/agent/tools.py",
     "        membership = membership_panel(snapshots, frame[\"ts\"].unique())",
     "        membership = membership_panel(snapshots, frame[\"ts\"].unique())\n        frame = frame.join(membership, on=[\"ts\", \"ticker\"], how=\"inner\")",
     "tests/test_tools_universe.py"),

    ("universe mask runs AFTER ranking instead of before",
     "src/falsify/agent/tools.py",
     "        signal = restrict_to_members(signal, panel.membership)",
     "        pass  # mask moved downstream of the sort",
     "tests/test_tools_universe.py"),

    ("`point_in_time` accepts a single snapshot date",
     "src/falsify/agent/tools.py",
     "        if n_snapshots < 2:",
     "        if n_snapshots < 1:",
     "tests/test_tools_universe.py"),

    ("a feature's column guessed from its menu key",
     "src/falsify/agent/tools.py",
     '    return _FEATURE_COLUMNS.get(feature, feature)',
     '    return feature',
     "tests/test_tools_universe.py"),

    ("idio vol falls back to total volatility without SPY",
     "src/falsify/features/library.py",
     "    if mkt.is_empty():\n        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))",
     "    if mkt.is_empty():\n        return add_rolling_vol(frame, window).rename({f'vol_{window}d': name})",
     "tests/test_tools_universe.py"),

    # --- runner.py: the trial count ---------------------------------------
    ("fresh Session per anomaly, so N=1 every time",
     "src/falsify/eval/runner.py",
     "            fh = T.compute_feature(session, panel_handle, a.feature)[\"handle\"]",
     "            session = Session(); panel_handle = fetch(session, tickers=tickers, start=start, end=end, universe=universe)[\"handle\"]\n            fh = T.compute_feature(session, panel_handle, a.feature)[\"handle\"]",
     "tests/test_eval_runner.py"),

    ("results analysed as they finish, not after the last backtest",
     "src/falsify/eval/runner.py",
     "        except T.ToolError as exc:\n            errors[a.key] = str(exc)\n\n    # --- pass two",
     "            T.analyze_results(session, backtests[a.key][\"handle\"])\n        except T.ToolError as exc:\n            errors[a.key] = str(exc)\n\n    # --- pass two",
     "tests/test_eval_runner.py"),

    ("measured trial variance left annualised",
     "src/falsify/eval/runner.py",
     "    return sum((s - mean) ** 2 for s in sharpes) / (len(sharpes) - 1)",
     "    return sum((s - mean) ** 2 for s in sharpes) / (len(sharpes) - 1) * 252",
     "tests/test_eval_runner.py"),

    # --- tools.py: the statistic that is allowed to be undefined ----------
    ("an undefined MinTRL kills the whole analysis",
     "src/falsify/agent/tools.py",
     "    except ValueError as exc:\n        trl = None",
     "    except ValueError as exc:\n        raise ToolError(f'stats failed: {exc}') from exc\n        trl = None",
     "tests/test_tools_universe.py"),

    # --- store.py: the schema ---------------------------------------------
    ("eval rows keyed on `eval_key` alone",
     "src/falsify/eval/store.py",
     "    PRIMARY KEY (eval_key, universe)",
     "    PRIMARY KEY (eval_key)",
     "tests/test_eval_store.py"),
]


def _as(text: str, newline: bytes) -> bytes:
    """Encode a pattern using the line ending the target file actually uses."""
    return text.encode().replace(b"\r\n", b"\n").replace(b"\n", newline)


def guard() -> str | None:
    """Refuse to run unless the working tree is clean. Returns an error, or None.

    A check that refuses to run is worth more than a check that warns -- the
    project's first lesson, applied to the script that validates the others.
    """
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=ROOT, capture_output=True, text=True,
        )
    except FileNotFoundError:
        return "git is not on PATH. This script edits source files in place and "\
               "will not do that without a recovery path."
    if inside.returncode != 0:
        return f"{ROOT} is not a git repository. This script edits source files "\
               "in place and will not do that without a recovery path."

    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        return (
            "the working tree has uncommitted changes:\n\n"
            + "\n".join("    " + line for line in dirty.splitlines()[:20])
            + "\n\nEvery injection is reverted in a `finally`, so an ordinary Ctrl-C is\n"
            "safe. A hard kill is not, and recovering from one means `git checkout -- .`,\n"
            "which would also discard the work above. Commit or stash first."
        )
    return None


def run(target: str) -> bool:
    """True when the target suite is GREEN."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "DATABASE_URL": DSN},
    )
    return r.returncode == 0


def main() -> int:
    problem = guard()
    if problem:
        print(f"REFUSING TO RUN: {problem}", file=sys.stderr)
        return 2

    print(f"{'#':>3}  {'injected bug':58s} result")
    print("-" * 78)
    missed = []
    for i, (label, rel, old, new, target) in enumerate(INJECTIONS, 1):
        path = ROOT / rel
        # BINARY IN, BINARY OUT, AND THE PATTERN ADAPTED TO THE FILE.
        #
        # Two line-ending bugs were paid for here, one after the other, and they
        # pull in opposite directions.
        #
        # First: `read_text`/`write_text` open in TEXT mode, which on Windows
        # writes \n back out as \r\n. The content reverted perfectly and every
        # line ending changed, so a clean tree came back dirty in six files and
        # the final guard cried wolf about its own edit. Hence read_bytes and
        # write_bytes: the bytes in must be the bytes out.
        #
        # Second, and only visible once the first was fixed: a byte comparison
        # is line-ending SENSITIVE. The patterns below are written with \n, and
        # a checked-out file on Windows holds \r\n, so every MULTI-LINE pattern
        # silently stopped matching and six injections reported PATTERN NOT
        # FOUND. Single-line patterns kept working, which is exactly the sort of
        # partial failure that looks like a bad pattern rather than a bad
        # matcher.
        #
        # So the pattern is translated into whatever the file actually uses,
        # with the other convention as a fallback for a mixed file. A
        # PATTERN NOT FOUND now means the source really has changed.
        original = path.read_bytes()
        newline = b"\r\n" if b"\r\n" in original else b"\n"
        old_b, new_b = _as(old, newline), _as(new, newline)
        if old_b not in original:
            other = b"\n" if newline == b"\r\n" else b"\r\n"
            old_b, new_b = _as(old, other), _as(new, other)
        if old_b not in original:
            print(f"{i:>3}  {label:58s} PATTERN NOT FOUND")
            missed.append(label)
            continue
        patched = original.replace(old_b, new_b, 1)
        assert patched != original, "edit was a no-op"
        path.write_bytes(patched)
        try:
            green = run(target)
        finally:
            path.write_bytes(original)
        if green:
            print(f"{i:>3}  {label:58s} *** MISSED ***")
            missed.append(label)
        else:
            print(f"{i:>3}  {label:58s} caught")

    print("-" * 78)
    print(f"{len(INJECTIONS) - len(missed)}/{len(INJECTIONS)} caught")
    if missed:
        print("\nMISSED:")
        for mlabel in missed:
            print(f"  - {mlabel}")

    # Proof that nothing was left behind. If this is not empty, a revert failed
    # and the tree needs `git checkout -- .` before anything else happens.
    left = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if left:
        print("\nWARNING: the tree is no longer clean, so a revert did not complete:",
              file=sys.stderr)
        print(left, file=sys.stderr)
        print("Run: git checkout -- .", file=sys.stderr)
        return 2
    print("tree clean: every injection was reverted")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
