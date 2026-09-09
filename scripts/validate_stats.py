"""Validate the statistics layer by simulation rather than by unit test.

Unit tests prove the code matches the tests. They cannot prove the formulas
were transcribed from the papers correctly, because the tests were written from
the same transcription. These checks close that gap: each one re-derives the
answer empirically, without using the formula under test.

    python scripts/validate_stats.py

Runs in about 20 seconds. Prints a table; exits non-zero if any check fails.
"""
from __future__ import annotations

import math
import random
import sys
from statistics import NormalDist

sys.path.insert(0, "src")

import polars as pl

from falsify.stats.deflated import (
    expected_max_sharpe,
    probabilistic_sharpe,
    sharpe_stats,
)
from falsify.stats.multipletest import benjamini_hochberg, holm_bonferroni

N = NormalDist()
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {detail}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. expected_max_sharpe against brute-force simulation
# ---------------------------------------------------------------------------

def check_expected_max(trials: int = 60_000) -> None:
    """Run the experiment the formula approximates: draw N worthless Sharpes,
    take the best, average. Uses no part of the closed form."""
    print("\n1. expected_max_sharpe vs Monte Carlo")
    random.seed(0)
    variance, sd = 0.0025, math.sqrt(0.0025)
    for n_trials in (2, 6, 20, 100):
        total = sum(
            max(random.gauss(0.0, sd) for _ in range(n_trials)) for _ in range(trials)
        )
        empirical = total / trials
        formula = expected_max_sharpe(n_trials, variance)
        # The closed form is asymptotic, so agreement is approximate by design.
        check(
            f"N={n_trials}",
            abs(empirical - formula) < 0.005,
            f"simulated {empirical:.6f}  formula {formula:.6f}",
        )


# ---------------------------------------------------------------------------
# 2. PSR calibration under the null
# ---------------------------------------------------------------------------

def check_psr_calibration(runs: int = 4000, n_days: int = 500) -> None:
    """A correctly built PSR is uniform on [0, 1] when the true Sharpe is zero,
    so exactly 5% of runs should exceed 0.95.

    This is the check that catches a mistranscribed formula. A flipped skew
    sign, excess instead of non-excess kurtosis, or sqrt(n) instead of
    sqrt(n - 1) all break calibration visibly.
    """
    print("\n2. PSR calibration under the null (true Sharpe = 0)")
    random.seed(1)
    psrs = []
    for _ in range(runs):
        r = pl.Series([random.gauss(0.0, 0.01) for _ in range(n_days)])
        s = sharpe_stats(r)
        psrs.append(probabilistic_sharpe(s["sr"], s["skew"], s["kurt"], n_days))
    for threshold in (0.50, 0.90, 0.95, 0.99):
        observed = sum(p > threshold for p in psrs) / runs
        expected = 1 - threshold
        check(
            f"P(PSR > {threshold:.2f})",
            abs(observed - expected) < max(0.02, 0.5 * expected),
            f"observed {observed:6.2%}  expected {expected:6.2%}",
        )


def check_psr_power(runs: int = 2000, n_days: int = 500) -> None:
    """Calibration alone is satisfied by a function that always returns noise.
    This confirms PSR still detects a genuine edge."""
    print("\n3. PSR power (true per-period Sharpe = 0.10)")
    random.seed(2)
    hits = 0
    for _ in range(runs):
        r = pl.Series([random.gauss(0.10 * 0.01, 0.01) for _ in range(n_days)])
        s = sharpe_stats(r)
        if probabilistic_sharpe(s["sr"], s["skew"], s["kurt"], n_days) > 0.95:
            hits += 1
    check("detected at 95%", hits / runs > 0.5, f"{hits / runs:.1%} of runs")


# ---------------------------------------------------------------------------
# 4. BH and Holm against brute-force implementations of the definitions
# ---------------------------------------------------------------------------

def _bh_bruteforce(pvalues: list[float], alpha: float) -> list[bool]:
    """Literal step-up: find the largest passing rank, reject everything below."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    k = 0
    for rank in range(1, m + 1):
        if pvalues[order[rank - 1]] <= rank / m * alpha:
            k = rank
    reject = [False] * m
    for rank in range(1, k + 1):
        reject[order[rank - 1]] = True
    return reject


def _holm_bruteforce(pvalues: list[float], alpha: float) -> list[bool]:
    """Literal step-down: stop at the first failure."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    for rank in range(1, m + 1):
        if pvalues[order[rank - 1]] <= alpha / (m - rank + 1):
            reject[order[rank - 1]] = True
        else:
            break
    return reject


def check_corrections(cases: int = 20_000) -> None:
    """The module derives `reject` from adjusted p-values via running min/max.
    These brute-force versions follow the textbook definitions directly, so
    agreement across random inputs means both readings of the procedure match.
    """
    print("\n4. BH and Holm vs brute-force from the definitions")
    random.seed(11)
    bad_bh = bad_holm = 0
    for _ in range(cases):
        m = random.randint(1, 15)
        raw = [round(random.random() ** 3, 12) for _ in range(m)]
        alpha = random.choice([0.01, 0.05, 0.1, 0.2])
        if benjamini_hochberg(raw, alpha)[0] != _bh_bruteforce(raw, alpha):
            bad_bh += 1
        if holm_bonferroni(raw, alpha)[0] != _holm_bruteforce(raw, alpha):
            bad_holm += 1
    check("Benjamini-Hochberg", bad_bh == 0, f"{cases - bad_bh}/{cases} agree")
    check("Holm-Bonferroni", bad_holm == 0, f"{cases - bad_holm}/{cases} agree")


def check_error_control(runs: int = 1000, m: int = 20, alpha: float = 0.05) -> None:
    """Simulate whole experiments where every hypothesis is null and count how
    often anything is falsely declared. Holm controls the family-wise error
    rate, so its figure must land near alpha."""
    print(f"\n5. Error control by simulation ({m} null hypotheses, alpha={alpha})")
    random.seed(7)
    any_bh = any_holm = 0
    for _ in range(runs):
        raw = [1 - N.cdf(random.gauss(0.0, 1.0)) for _ in range(m)]
        any_bh += any(benjamini_hochberg(raw, alpha)[0])
        any_holm += any(holm_bonferroni(raw, alpha)[0])
    check(
        "Holm FWER near alpha",
        any_holm / runs < alpha * 1.6,
        f"{any_holm / runs:.1%} of experiments had a false positive",
    )
    check(
        "BH no stricter than Holm",
        any_bh >= any_holm - runs * 0.01,
        f"{any_bh / runs:.1%} of experiments had a false positive",
    )


def main() -> None:
    print("Validating the statistics layer by simulation.")
    check_expected_max()
    check_psr_calibration()
    check_psr_power()
    check_corrections()
    check_error_control()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
