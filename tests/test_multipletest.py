"""Tests for Benjamini-Hochberg and Holm-Bonferroni correction.

Worked examples are computed by hand in the docstrings. The cross-procedure
relationships (Holm subset of BH, monotone adjusted p-values, reject agrees with
adjusted <= alpha) catch step-up/step-down errors a single example would not.
"""
from __future__ import annotations

from statistics import NormalDist

import pytest

from falsify.stats.multipletest import (
    benjamini_hochberg,
    holm_bonferroni,
    sharpe_pvalue,
)

N = NormalDist()


# ==========================================================================
# Benjamini-Hochberg
# ==========================================================================

def test_bh_worked_example_all_rejected():
    """p = [0.01, 0.02, 0.03, 0.04, 0.05], m = 5, alpha = 0.05.

    Thresholds k/m*alpha = [0.01, 0.02, 0.03, 0.04, 0.05].
    Every p_(k) meets its own threshold, so the largest k is 5: reject all.

    Adjusted: (m/j)*p_(j) = [0.05, 0.05, 0.05, 0.05, 0.05], already monotone.

    Bonferroni at 0.05/5 = 0.01 would reject only the first. That gap in power
    is why BH is the screening standard.
    """
    reject, adj = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05)
    assert reject == [True] * 5
    assert adj == pytest.approx([0.05] * 5)


def test_bh_worked_example_one_rejected():
    """p = [0.001, 0.5, 0.6, 0.7, 0.8], alpha = 0.05.

    p_(1) = 0.001 <= 0.01  ok
    p_(2) = 0.5   <= 0.02  fails, and so does everything after.
    Largest k = 1: reject only the first.

    Adjusted, q_(i) = min over j >= i of (5/j)*p_(j), capped at 1:
        raw (5/j)*p_(j) = [0.005, 1.25, 1.0, 0.875, 0.8]
        running minimum from the right = [0.005, 0.8, 0.8, 0.8, 0.8]
    The last four are 0.8, not 1.0: the running minimum pulls large middle
    values down, and only a value still above 1 after scaling is capped. Wrong
    here is harmless at alpha = 0.05 but wrong in the reported table.
    """
    reject, adj = benjamini_hochberg([0.001, 0.5, 0.6, 0.7, 0.8], alpha=0.05)
    assert reject == [True, False, False, False, False]
    assert adj == pytest.approx([0.005, 0.8, 0.8, 0.8, 0.8])


def test_bh_step_up_rejects_above_its_own_threshold():
    """Commonly implemented wrong. p = [0.001, 0.049], m = 2, alpha = 0.05.

      p_(1) = 0.001 <= (1/2)*0.05 = 0.025  ok
      p_(2) = 0.049 <= (2/2)*0.05 = 0.05   ok  -> largest k = 2
    Both rejected. Comparing each p independently against k/m*alpha would keep
    only the first. BH steps UP: largest passing k, then reject everything below.
    """
    reject, _ = benjamini_hochberg([0.001, 0.049], alpha=0.05)
    assert reject == [True, True]


def test_bh_middle_failure_does_not_stop_the_procedure():
    """p = [0.001, 0.04, 0.045], m = 3, alpha = 0.05.
      k=1: 0.001 <= 0.01667  ok
      k=2: 0.04  <= 0.03333  FAILS
      k=3: 0.045 <= 0.05     ok  -> largest k = 3, all three rejected,
    including the one that failed its own threshold. Stopping at the first
    failure is Holm's rule, not BH's.
    """
    reject, _ = benjamini_hochberg([0.001, 0.04, 0.045], alpha=0.05)
    assert reject == [True, True, True]


def test_bh_rejects_nothing_when_nothing_is_significant():
    reject, adj = benjamini_hochberg([0.4, 0.6, 0.9, 1.0], alpha=0.05)
    assert reject == [False] * 4
    assert all(a <= 1.0 for a in adj)


def test_bh_preserves_input_order():
    """Results zip back onto anomaly names. Sorted output mislabels silently:
    the strongest anomaly takes the first name in the list, not its own."""
    reject, adj = benjamini_hochberg([0.8, 0.001, 0.5], alpha=0.05)
    assert reject == [False, True, False]
    assert adj[1] < adj[0]
    assert adj[1] < adj[2]


def test_bh_adjusted_pvalues_are_monotone():
    """A larger raw p can never yield a smaller adjusted one: what the running
    minimum from the top enforces."""
    raw = [0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9]
    _, adj = benjamini_hochberg(raw)
    pairs = sorted(zip(raw, adj))
    assert [a for _, a in pairs] == sorted(a for _, a in pairs)


def test_bh_reject_agrees_with_adjusted_against_alpha():
    """Two derivations of one decision must not disagree."""
    raw = [0.0001, 0.01, 0.02, 0.03, 0.049, 0.2, 0.6]
    for alpha in (0.01, 0.05, 0.1, 0.2):
        reject, adj = benjamini_hochberg(raw, alpha=alpha)
        assert reject == [a <= alpha + 1e-15 for a in adj]


def test_bh_adjusted_never_exceeds_one():
    _, adj = benjamini_hochberg([0.9, 0.95, 0.99, 1.0])
    assert all(0.0 <= a <= 1.0 for a in adj)


def test_bh_single_hypothesis_is_the_identity():
    reject, adj = benjamini_hochberg([0.03], alpha=0.05)
    assert reject == [True]
    assert adj == pytest.approx([0.03])


def test_bh_rejects_invalid_input():
    with pytest.raises(ValueError):
        benjamini_hochberg([])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, 1.5])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, -0.1])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5], alpha=0.0)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5], alpha=1.0)


# ==========================================================================
# Holm-Bonferroni
# ==========================================================================

def test_holm_worked_example():
    """p = [0.01, 0.02, 0.03, 0.04, 0.05], m = 5, alpha = 0.05.

    Step down: compare p_(i) against alpha / (m - i + 1).
      i=1: 0.01 <= 0.05/5 = 0.01   ok
      i=2: 0.02 <= 0.05/4 = 0.0125 FAILS -> stop, reject nothing further.
    Holm rejects one where BH rejected all five: same data, two error rates,
    two answers. Reporting only the favourable one is what this pair of
    functions exists to prevent.

    Adjusted: (m-j+1)*p_(j) = [0.05, 0.08, 0.09, 0.08, 0.05], then the running
    maximum from the left gives [0.05, 0.08, 0.09, 0.09, 0.09].
    """
    reject, adj = holm_bonferroni([0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05)
    assert reject == [True, False, False, False, False]
    assert adj == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])


def test_holm_stops_at_the_first_failure():
    """p = [0.001, 0.9, 0.0001]. Sorted: 0.0001 (<= 0.05/3), 0.001 (<= 0.025),
    0.9 fails. Both small ones reject. Rank order, not raw value, decides: a
    small p sorted after the stop would be retained."""
    reject, _ = holm_bonferroni([0.001, 0.9, 0.0001], alpha=0.05)
    assert reject == [True, False, True]


def test_holm_is_at_least_as_strict_as_bh():
    """Anything Holm rejects, BH must also reject: FWER control implies FDR
    control. A violation means one procedure is implemented backwards."""
    import random

    random.seed(3)
    for _ in range(200):
        m = random.randint(1, 12)
        raw = [random.random() ** 3 for _ in range(m)]
        alpha = random.choice([0.01, 0.05, 0.1])
        h, _ = holm_bonferroni(raw, alpha)
        b, _ = benjamini_hochberg(raw, alpha)
        assert all((not hi) or bi for hi, bi in zip(h, b))


def test_holm_beats_plain_bonferroni():
    """Holm rejects a superset of Bonferroni at the same alpha, which is why
    plain Bonferroni has no remaining use.

    p = [0.004, 0.02], alpha = 0.05. Bonferroni at 0.025 takes the first only.
    Holm: 0.004 <= 0.025 ok, then 0.02 <= 0.05 ok -> both.
    """
    reject, _ = holm_bonferroni([0.004, 0.02], alpha=0.05)
    assert reject == [True, True]


def test_holm_adjusted_pvalues_are_monotone():
    raw = [0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9]
    _, adj = holm_bonferroni(raw)
    pairs = sorted(zip(raw, adj))
    assert [a for _, a in pairs] == sorted(a for _, a in pairs)


def test_holm_reject_agrees_with_adjusted_against_alpha():
    raw = [0.0001, 0.01, 0.02, 0.03, 0.049, 0.2, 0.6]
    for alpha in (0.01, 0.05, 0.1):
        reject, adj = holm_bonferroni(raw, alpha=alpha)
        assert reject == [a <= alpha + 1e-15 for a in adj]


def test_holm_preserves_input_order():
    reject, adj = holm_bonferroni([0.9, 0.0001, 0.5], alpha=0.05)
    assert reject == [False, True, False]


def test_holm_rejects_invalid_input():
    with pytest.raises(ValueError):
        holm_bonferroni([])
    with pytest.raises(ValueError):
        holm_bonferroni([0.5, 2.0])


# ==========================================================================
# sharpe_pvalue — the bridge from metrics.py
# ==========================================================================

def test_sharpe_pvalue_golden_value_hand_derived():
    """sr_annual = 1.0, n = 252, 252 periods/year, one-sided.

    per-period SR = 1.0 / sqrt(252) = 0.06299...
    t             = 0.06299 * sqrt(252) = 1.0
    p             = 1 - Phi(1.0) = 0.158655...

    De-annualisation and sqrt(n) cancel at n = one year: Sharpe 1.0 over one
    year is a one-sigma result, not significant.
    """
    assert sharpe_pvalue(1.0, n=252) == pytest.approx(1 - N.cdf(1.0), abs=1e-12)
    assert sharpe_pvalue(1.0, n=252) == pytest.approx(0.158655, abs=1e-6)


def test_sharpe_pvalue_zero_sharpe_is_one_half_one_sided():
    assert sharpe_pvalue(0.0, n=1000) == pytest.approx(0.5)


def test_sharpe_pvalue_two_sided_is_double_when_positive():
    one = sharpe_pvalue(1.0, n=252, alternative="greater")
    two = sharpe_pvalue(1.0, n=252, alternative="two-sided")
    assert two == pytest.approx(2 * one, rel=1e-12)


def test_sharpe_pvalue_two_sided_is_symmetric():
    """A Sharpe of -1.0 is as unusual as +1.0 under the two-sided null."""
    assert sharpe_pvalue(-1.0, n=500, alternative="two-sided") == pytest.approx(
        sharpe_pvalue(1.0, n=500, alternative="two-sided"), rel=1e-12
    )


def test_sharpe_pvalue_one_sided_penalises_losses():
    """One-sided, a negative Sharpe gives p above 0.5. Losing money consistently
    is not a discovery."""
    assert sharpe_pvalue(-1.0, n=500, alternative="greater") > 0.5


def test_sharpe_pvalue_falls_with_sample_length():
    vals = [sharpe_pvalue(0.5, n=k) for k in (100, 500, 2000, 10000)]
    assert vals == sorted(vals, reverse=True)


def test_sharpe_pvalue_matches_the_momentum_run_scale():
    """A 0.71 annualised Sharpe over 233 days: not significant one-sided at 5%.

    t = (0.71/sqrt(252)) * sqrt(233) = 0.0447 * 15.264 = 0.6828, p = 0.247.
    The naive test and PSR agree here: the verdict does not depend on the
    instrument.
    """
    p = sharpe_pvalue(0.71, n=233)
    assert p == pytest.approx(0.2474, abs=5e-4)
    assert p > 0.05


def test_sharpe_pvalue_rejects_bad_input():
    with pytest.raises(ValueError):
        sharpe_pvalue(1.0, n=1)
    with pytest.raises(ValueError):
        sharpe_pvalue(1.0, n=252, alternative="less-than-maybe")


# ==========================================================================
# The two procedures against a realistic eval-suite result
# ==========================================================================

def test_six_anomaly_suite_end_to_end():
    """The M6 shape end to end: six anomalies, 5 years daily, one genuinely
    strong and the rest near noise. BH keeps the strong one, Holm agrees,
    neither certifies the marginal ones."""
    n = 1260
    sharpes = {
        "momentum": 1.9,
        "value": 0.5,
        "low_vol": 0.9,
        "pead": 0.2,
        "size": -0.3,
        "profitability": 0.6,
    }
    names = list(sharpes)
    raw = [sharpe_pvalue(sharpes[k], n=n) for k in names]

    bh_reject, bh_adj = benjamini_hochberg(raw, alpha=0.05)
    holm_reject, _ = holm_bonferroni(raw, alpha=0.05)

    assert bh_reject[names.index("momentum")] is True
    assert holm_reject[names.index("momentum")] is True
    assert bh_reject[names.index("size")] is False
    assert bh_reject[names.index("pead")] is False
    # Correction bites.
    assert bh_adj[names.index("value")] > raw[names.index("value")]
    assert all((not h) or b for h, b in zip(holm_reject, bh_reject))
