"""Probabilistic and Deflated Sharpe Ratio.

A Sharpe ratio is an estimate from a finite sample, and two forces bias it
upward: short samples are noisy, and reporting the best of N strategies reports
the maximum of N draws. Non-normal returns compound both, since a negatively
skewed, fat-tailed series earns its mean by selling disaster insurance.

    PSR   probability the TRUE Sharpe beats a benchmark, given this sample's
          length and shape.
    DSR   the same question, with the benchmark set to what the best of N
          worthless trials would print.

Conventions, pinned because every result depends on them:
  - Sharpe is PER-PERIOD, never annualised. Passing an annualised Sharpe
    inflates PSR enormously; it is the standard error in this literature.
  - Kurtosis is NON-EXCESS: a normal distribution scores 3, not 0.
  - skew/kurt use the moment estimators (/n); `sr` uses std with ddof=1 to
    match metrics.sharpe.

Bailey & Lopez de Prado (2012, 2014). No SciPy: stdlib NormalDist only.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import polars as pl

TRADING_DAYS = 252

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe approximation.
EULER_GAMMA = 0.5772156649015329


def sharpe_stats(returns: pl.Series, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    """The four inputs PSR needs, from a daily return series.

    Args:
        returns: daily simple returns, chronological.

    Returns:
        sr    per-period Sharpe, mean / std(ddof=1). NOT annualised.
        skew  moment skewness, m3 / m2**1.5.
        kurt  moment kurtosis, m4 / m2**2, NON-EXCESS (normal = 3).
        n     observation count.

    Raises:
        ValueError: fewer than 4 observations, or zero volatility. Both leave
            the higher moments undefined, and a plausible-looking number would
            be worse than failing.
    """
    n = len(returns)
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}")

    r = returns.cast(pl.Float64)
    sd = r.std(ddof=1)                 # ddof=1 to match metrics.sharpe
    if sd is None or sd == 0.0:
        raise ValueError("zero volatility: Sharpe and higher moments undefined")

    mean = float(r.mean())
    deviations = [float(v) - mean for v in r.to_list()]

    # Moment estimators (/n), as in the papers — deliberately a different
    # convention from the ddof=1 above. Tests pin both.
    m2 = sum(d**2 for d in deviations) / n
    m3 = sum(d**3 for d in deviations) / n
    m4 = sum(d**4 for d in deviations) / n
    if m2 == 0.0:
        raise ValueError("zero variance")

    # Computed explicitly rather than via Polars .skew()/.kurtosis(), whose
    # defaults differ: .kurtosis() returns EXCESS kurtosis (normal = 0).
    return {
        "sr": mean / float(sd),
        "skew": m3 / m2**1.5,
        "kurt": m4 / m2**2,
        "n": float(n),
    }


def probabilistic_sharpe(
    sr: float,
    skew: float,
    kurt: float,
    n: int,
    benchmark: float = 0.0,
) -> float:
    """Probability that the TRUE per-period Sharpe exceeds `benchmark`.

        PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n - 1)
                        / sqrt(1 - g3*SR_hat + ((g4 - 1)/4) * SR_hat**2) )

    Args:
        sr:        estimated PER-PERIOD Sharpe (not annualised).
        skew:      moment skewness g3.
        kurt:      moment kurtosis g4, non-excess (normal = 3).
        n:         number of observations.
        benchmark: SR*, the per-period Sharpe to beat. 0.0 asks the weakest
                   interesting question: is this strategy better than nothing?

    Returns:
        A probability in [0, 1].

    The denominator carries the intuition: sqrt(n - 1) means confidence grows
    with the square root of sample length; `- g3*SR` penalises negative skew,
    whose mean is compensation for a tail the sample may not have realised
    yet; and the g4 term penalises fat tails. Under normality (g3=0, g4=3) the
    denominator collapses to sqrt(1 + SR**2 / 2).

    Raises:
        ValueError: n < 2, or a non-positive quantity under the root, meaning
            the supplied moments are mutually inconsistent.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}")

    under_root = 1 - skew * sr + ((kurt - 1) / 4) * sr**2
    if under_root <= 0.0:
        raise ValueError(f"inconsistent moments: variance term {under_root} <= 0")

    numerator = (sr - benchmark) * math.sqrt(n - 1)
    denominator = math.sqrt(under_root)
    return NormalDist().cdf(numerator / denominator)


def expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """The per-period Sharpe the BEST of `n_trials` worthless strategies would print.

        E[max SR] ~= sqrt(V) * ( (1 - gamma) * Phi^-1(1 - 1/N)
                                 + gamma * Phi^-1(1 - 1/(N*e)) )

    where gamma is Euler-Mascheroni and V is the variance of the estimated
    Sharpes ACROSS the trials.

    Args:
        n_trials:       configurations actually tried, not the count reported.
            Six anomalies at two rebalancing frequencies is twelve, and
            abandoned variants count. Understating N silently re-inflates the
            deflated Sharpe.
        trial_variance: variance of the per-period Sharpes across those trials.
            Measured from the six anomalies at M6; an assumption before that,
            and must be labelled as one in any output.

    Returns:
        Per-period Sharpe benchmark. 0.0 for n_trials == 1: no selection to
        correct for, and Phi^-1(0) diverges anyway.

    Raises:
        ValueError: n_trials < 1 or trial_variance < 0.

    Assumes the trials' true Sharpes are zero and their estimates independent
    draws from N(0, V). Real trials are correlated (six anomalies on one
    universe share market beta), so the true expected maximum is smaller and
    this benchmark is conservative.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if trial_variance < 0:
        raise ValueError(f"trial_variance must be >= 0, got {trial_variance}")

    # Must precede the maths: Phi^-1(1 - 1/1) = Phi^-1(0) diverges, and with one
    # trial there is no selection to correct for anyway.
    if n_trials == 1:
        return 0.0

    # sqrt(V) scales the whole bracket, so dispersion across trials sets how
    # much selection buys; the bracket is how far out the max of N draws lands.
    s = math.sqrt(trial_variance)
    return s * (
        (1 - EULER_GAMMA) * NormalDist().inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * NormalDist().inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(
    returns: pl.Series,
    n_trials: int,
    trial_variance: float,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """PSR measured against the expected best-of-N-trials Sharpe.

    Args:
        returns:        daily simple returns of the SELECTED strategy.
        n_trials:       see `expected_max_sharpe`.
        trial_variance: see `expected_max_sharpe`.

    Returns:
        Probability in [0, 1] that this strategy's true Sharpe beats what the
        best of `n_trials` coin flips would have produced.

    A DSR below 0.5 does not mean the strategy loses money. It means the
    evidence does not survive the fact that several strategies were tried:
    a statement about evidence, not about the sign of the edge.
    """
    s = sharpe_stats(returns, periods_per_year)
    benchmark = expected_max_sharpe(n_trials, trial_variance)
    return probabilistic_sharpe(
        s["sr"], s["skew"], s["kurt"], int(s["n"]), benchmark=benchmark
    )


def min_track_record_length(
    sr: float,
    skew: float,
    kurt: float,
    benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed before PSR(benchmark) would reach `confidence`.

        MinTRL = 1 + (1 - g3*SR + ((g4 - 1)/4) * SR**2) * (Phi^-1(conf) / (SR - SR*))**2

    Returns:
        A number of PERIODS (days, at daily input). Exceeding any realistic
        sample is itself the finding.

    Raises:
        ValueError: sr <= benchmark (the answer is infinite), or confidence
            outside (0, 1).

    The most quotable statistic here: "this result needs N years before its
    Sharpe is distinguishable from zero" beats any p-value.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if sr <= benchmark:
        raise ValueError(
            f"sr {sr} does not exceed benchmark {benchmark}; MinTRL is infinite"
        )

    under_root = 1 - skew * sr + ((kurt - 1) / 4) * sr**2
    if under_root <= 0.0:
        raise ValueError(f"inconsistent moments: variance term {under_root} <= 0")

    # PSR solved for n. The edge enters squared, so halving it roughly
    # quadruples the sample needed to prove it.
    return 1.0 + under_root * (NormalDist().inv_cdf(confidence) / (sr - benchmark)) ** 2
