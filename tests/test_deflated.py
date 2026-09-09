"""Tests for the Probabilistic and Deflated Sharpe Ratio.

Every expectation is hand-derived from the published formula (derivation in the
docstring) or is an analytic identity. Nothing is copied from an implementation,
so these can fail one rather than merely describe it.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import polars as pl
import pytest

from falsify.stats.deflated import (
    EULER_GAMMA,
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe,
    sharpe_stats,
)

N = NormalDist()


def series_with_sharpe(ann_sharpe: float, n: int, sd: float = 0.01, seed: int = 1) -> pl.Series:
    """A return series whose ANNUALISED Sharpe is exactly `ann_sharpe`.

    Sampling and hoping is not good enough here: at n = 233 a seed can deliver
    2.2 when 0.7 was intended, so the test would turn on the seed. Standardise
    the draws and impose the moment.
    """
    import random

    random.seed(seed)
    x = [random.gauss(0.0, 1.0) for _ in range(n)]
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    target = ann_sharpe / math.sqrt(252)          # per-period
    return pl.Series([sd * (target + (v - m) / s) for v in x])


# Variance of per-period Sharpes across the six eval-suite anomalies. 0.0025 is
# a spread of about 0.8 in ANNUALISED Sharpe, the right order of magnitude.
# Placeholder until measured at M6; any output using it before then says so.
SUITE_TRIAL_VARIANCE = 0.0025


# ==========================================================================
# probabilistic_sharpe
# ==========================================================================

def test_psr_golden_value_hand_derived():
    """PSR(sr=0.1, skew=0, kurt=3, n=101, benchmark=0).

    numerator   = (0.1 - 0) * sqrt(101 - 1) = 0.1 * 10          = 1.0
    denominator = sqrt(1 - 0*0.1 + ((3 - 1)/4) * 0.1**2)
                = sqrt(1 + 0.5 * 0.01) = sqrt(1.005)            = 1.00249688...
    z           = 1.0 / 1.00249688                              = 0.99750934...
    PSR         = Phi(0.99750934)                               = 0.84074133...
    """
    got = probabilistic_sharpe(sr=0.1, skew=0.0, kurt=3.0, n=101)
    assert got == pytest.approx(0.8407413278, abs=1e-9)


def test_psr_at_the_benchmark_is_one_half():
    """Phi(0) = 0.5 at the benchmark, whatever the length or higher moments.
    Failing this means the numerator is wrong."""
    for skew, kurt, n in [(0.0, 3.0, 50), (-1.5, 9.0, 5000), (2.0, 20.0, 10)]:
        assert probabilistic_sharpe(0.2, skew, kurt, n, benchmark=0.2) == pytest.approx(0.5)


def test_psr_normal_case_reduces_to_the_familiar_form():
    """Under normality (skew 0, kurt 3) the denominator collapses to
    sqrt(1 + SR**2 / 2), the form quoted in most textbooks."""
    sr, n = 0.08, 500
    expected = N.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe(sr, 0.0, 3.0, n) == pytest.approx(expected, abs=1e-12)


def test_psr_increases_with_sample_length():
    """The same Sharpe observed for longer is stronger evidence: the reason PSR
    exists rather than reporting the Sharpe alone."""
    vals = [probabilistic_sharpe(0.05, 0.0, 3.0, n) for n in (50, 200, 1000, 5000)]
    assert vals == sorted(vals)
    assert vals[0] < vals[-1]


def test_psr_increases_with_sharpe():
    vals = [probabilistic_sharpe(sr, 0.0, 3.0, 500) for sr in (-0.05, 0.0, 0.05, 0.2)]
    assert vals == sorted(vals)


def test_negative_skew_is_penalised():
    """Selling tail risk must score LOWER than symmetric returns at equal Sharpe.

    -g3*SR grows when g3 is negative, widening the standard error. This term
    distinguishes PSR from a t-stat; sign it backwards, commonly done, and the
    metric rewards exactly the return profile you should distrust.
    """
    neg = probabilistic_sharpe(0.1, skew=-2.0, kurt=3.0, n=101)
    sym = probabilistic_sharpe(0.1, skew=0.0, kurt=3.0, n=101)
    pos = probabilistic_sharpe(0.1, skew=+2.0, kurt=3.0, n=101)
    assert neg < sym < pos


def test_fat_tails_are_penalised():
    """Higher kurtosis at fixed skew lowers PSR: the sample may not yet have
    realised the tail the mean is compensation for."""
    thin = probabilistic_sharpe(0.1, 0.0, kurt=3.0, n=101)
    fat = probabilistic_sharpe(0.1, 0.0, kurt=30.0, n=101)
    assert fat < thin


def test_psr_is_a_probability():
    for sr in (-2.0, -0.1, 0.0, 0.1, 2.0):
        p = probabilistic_sharpe(sr, 0.5, 5.0, 300)
        assert 0.0 <= p <= 1.0


def test_psr_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        probabilistic_sharpe(0.1, 0.0, 3.0, n=1)
    with pytest.raises(ValueError):
        probabilistic_sharpe(0.1, 0.0, 3.0, n=0)


def test_psr_rejects_inconsistent_moments():
    """Moments driving the quantity under the root to zero or below are not a
    distribution; nan or a complex number would propagate silently."""
    with pytest.raises(ValueError):
        # 1 - g3*SR goes deeply negative, kurtosis term cannot rescue it.
        probabilistic_sharpe(sr=2.0, skew=10.0, kurt=1.0, n=100)


# ==========================================================================
# expected_max_sharpe
# ==========================================================================

def test_expected_max_golden_value_hand_derived():
    """E[max SR] for N=10 trials with Sharpe variance 0.01.

    sqrt(V) = 0.1, gamma = 0.5772156649...
    term1 = (1 - gamma) * Phi^-1(1 - 1/10)      = 0.4227843351 * Phi^-1(0.9)
    term2 = gamma       * Phi^-1(1 - 1/(10*e))  = 0.5772156649 * Phi^-1(0.96321...)
    E     = 0.1 * (term1 + term2)
    """
    expected = 0.1 * (
        (1 - EULER_GAMMA) * N.inv_cdf(1 - 1 / 10)
        + EULER_GAMMA * N.inv_cdf(1 - 1 / (10 * math.e))
    )
    assert expected_max_sharpe(10, 0.01) == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(0.157460, abs=1e-5)


def test_single_trial_needs_no_deflation():
    """No selection to correct for, and Phi^-1(0) diverges anyway. The answer is
    0, not an error and not infinity."""
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(1, 100.0) == 0.0


def test_expected_max_grows_with_trial_count():
    """The more strategies tried, the higher the bar. This monotonicity is the
    economic content of the deflation."""
    vals = [expected_max_sharpe(n, 0.01) for n in (2, 5, 10, 50, 100, 1000)]
    assert vals == sorted(vals)


def test_expected_max_scales_with_the_square_root_of_variance():
    """sqrt(V) factors out of the bracket, so quadrupling the variance must
    exactly double the benchmark."""
    a = expected_max_sharpe(20, 0.01)
    b = expected_max_sharpe(20, 0.04)
    assert b == pytest.approx(2.0 * a, rel=1e-12)


def test_zero_variance_means_no_bar():
    """No dispersion to select from means selection bought nothing."""
    assert expected_max_sharpe(100, 0.0) == pytest.approx(0.0)


def test_expected_max_rejects_bad_inputs():
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 0.01)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -0.01)


# ==========================================================================
# sharpe_stats
# ==========================================================================

def test_sharpe_stats_is_not_annualised():
    """THE trap in this module: a per-period Sharpe of 0.1 must report 0.1, not
    0.1*sqrt(252) = 1.59. An annualised Sharpe fed into PSR turns a marginal
    result into a certainty and still looks plausible, so nothing else catches
    it."""
    # 100 obs alternating 0.001 +/- 0.01: mean 0.001, std(ddof=1) =
    # 0.01 * sqrt(100/99), so sr = 0.001 / (0.01 * sqrt(100/99)) = 0.0994987...
    # Not exactly 0.1 is the point: it pins ddof=1, so population std fails here.
    vals = [0.001 + 0.01, 0.001 - 0.01] * 50
    s = sharpe_stats(pl.Series(vals))
    assert s["sr"] == pytest.approx(0.001 / (0.01 * math.sqrt(100 / 99)), rel=1e-12)
    assert s["sr"] == pytest.approx(0.0994987, abs=1e-6)
    assert abs(s["sr"]) < 1.0                     # annualised would be 1.58


def test_sharpe_stats_matches_metrics_sharpe_de_annualised():
    """metrics.sharpe is annualised, sharpe_stats is not. Disagreement beyond the
    sqrt(252) factor means two different Sharpe ratios in the project, and the
    note would quote whichever was convenient."""
    from falsify.backtest.metrics import sharpe as ann_sharpe

    r = pl.Series([0.004, -0.002, 0.011, -0.008, 0.003, 0.006, -0.001] * 30)
    assert sharpe_stats(r)["sr"] == pytest.approx(ann_sharpe(r) / math.sqrt(252), rel=1e-12)


def test_sharpe_stats_kurtosis_is_non_excess():
    """A normal-ish sample reports kurt near 3, not near 0. Excess kurtosis
    substituted here shifts the PSR denominator by ((3-1)/4 - (0-1)/4) * SR**2
    and biases every result in the note."""
    import random

    random.seed(20260909)
    r = pl.Series([random.gauss(0.0005, 0.01) for _ in range(20000)])
    s = sharpe_stats(r)
    assert s["kurt"] == pytest.approx(3.0, abs=0.2)
    assert s["skew"] == pytest.approx(0.0, abs=0.1)


def test_sharpe_stats_detects_negative_skew():
    """Many small gains, one large loss: skew must come back clearly negative."""
    r = pl.Series([0.002] * 99 + [-0.15])
    assert sharpe_stats(r)["skew"] < -5.0


def test_sharpe_stats_reports_n():
    r = pl.Series([0.01, -0.01, 0.02, -0.02, 0.005])
    assert sharpe_stats(r)["n"] == 5


def test_sharpe_stats_rejects_undefined_input():
    with pytest.raises(ValueError):
        sharpe_stats(pl.Series([0.01, 0.02]))            # too few for moment 4
    with pytest.raises(ValueError):
        sharpe_stats(pl.Series([0.01] * 100))            # zero volatility


# ==========================================================================
# deflated_sharpe — the composition
# ==========================================================================

@pytest.fixture
def modest_returns() -> pl.Series:
    """A series with a real but unremarkable per-period Sharpe."""
    return series_with_sharpe(0.9, n=1200, seed=7)


def test_deflation_can_only_lower_confidence(modest_returns):
    """DSR <= PSR(0) above one trial: raising the benchmark cannot raise the
    probability of clearing it."""
    s = sharpe_stats(modest_returns)
    psr0 = probabilistic_sharpe(s["sr"], s["skew"], s["kurt"], s["n"], benchmark=0.0)
    for n_trials in (2, 6, 20, 500):
        dsr = deflated_sharpe(modest_returns, n_trials=n_trials, trial_variance=0.01)
        assert dsr <= psr0 + 1e-12


def test_more_trials_lower_the_deflated_sharpe(modest_returns):
    vals = [
        deflated_sharpe(modest_returns, n_trials=k, trial_variance=0.01)
        for k in (2, 5, 25, 200)
    ]
    assert vals == sorted(vals, reverse=True)


def test_one_trial_deflated_equals_undeflated(modest_returns):
    s = sharpe_stats(modest_returns)
    psr0 = probabilistic_sharpe(s["sr"], s["skew"], s["kurt"], s["n"], benchmark=0.0)
    assert deflated_sharpe(modest_returns, 1, 0.01) == pytest.approx(psr0, abs=1e-12)


def test_a_short_impressive_backtest_does_not_survive_deflation():
    """The result this module exists to produce. A ~0.7 annualised Sharpe over
    233 days is the momentum run already in the repo: it looks like a finding,
    but against the benchmark set by six trials it must not clear one-in-two."""
    r = series_with_sharpe(0.7, n=233)
    s = sharpe_stats(r)

    # Undeflated it is already unconvincing: ~75% chance the true Sharpe is
    # merely positive.
    psr0 = probabilistic_sharpe(s["sr"], s["skew"], s["kurt"], 233)
    assert psr0 == pytest.approx(0.748, abs=0.01)

    # Deflated for having tried six anomalies, it does not clear a coin flip.
    dsr = deflated_sharpe(r, n_trials=6, trial_variance=SUITE_TRIAL_VARIANCE)
    assert dsr < 0.5


def test_a_long_strong_backtest_does_survive_deflation():
    """Control for the previous test: without it, always returning 0.0 passes."""
    r = series_with_sharpe(2.0, n=4000)
    assert deflated_sharpe(r, n_trials=6, trial_variance=SUITE_TRIAL_VARIANCE) > 0.95


# ==========================================================================
# min_track_record_length
# ==========================================================================

def test_min_trl_golden_value_hand_derived():
    """MinTRL(sr=0.1, skew=0, kurt=3, benchmark=0, confidence=0.95).

    = 1 + (1 - 0*0.1 + ((3-1)/4)*0.01) * (Phi^-1(0.95) / 0.1)**2
    = 1 + 1.005 * (1.6448536270 / 0.1)**2
    = 1 + 1.005 * 270.5543...                                   = 272.907...
    """
    expected = 1 + 1.005 * (N.inv_cdf(0.95) / 0.1) ** 2
    assert min_track_record_length(0.1, 0.0, 3.0) == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(272.907, abs=0.01)


def test_min_trl_is_consistent_with_psr():
    """At exactly MinTRL observations PSR equals the confidence level. The two
    functions invert one another, so this cannot pass unless both are right."""
    sr, skew, kurt, conf = 0.06, -0.4, 6.0, 0.95
    n = min_track_record_length(sr, skew, kurt, benchmark=0.0, confidence=conf)
    assert probabilistic_sharpe(sr, skew, kurt, int(round(n))) == pytest.approx(conf, abs=1e-3)


def test_min_trl_grows_as_the_edge_shrinks():
    """Quadratic: halving the Sharpe roughly quadruples the required sample."""
    a = min_track_record_length(0.10, 0.0, 3.0)
    b = min_track_record_length(0.05, 0.0, 3.0)
    assert b > 3.5 * a


def test_min_trl_undefined_below_the_benchmark():
    """No amount of data proves a strategy beats a benchmark it does not beat."""
    with pytest.raises(ValueError):
        min_track_record_length(0.05, 0.0, 3.0, benchmark=0.05)
    with pytest.raises(ValueError):
        min_track_record_length(0.02, 0.0, 3.0, benchmark=0.10)


def test_min_trl_rejects_bad_confidence():
    for c in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            min_track_record_length(0.1, 0.0, 3.0, confidence=c)
