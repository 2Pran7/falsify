"""Multiple-testing correction.

Test six anomalies at the 5% level and, even if all six are worthless, there is
about a 26% chance one comes back "significant". The eval suite tests six
hypotheses by design, so this is the price of running a suite at all.

    FWER  P(at least one false positive). Holm-Bonferroni. The standard when a
          single false positive is expensive, i.e. you are about to allocate.
    FDR   E[fraction of rejections that are false]. Benjamini-Hochberg. More
          powerful, and the right standard for a research screen.

falsify reports both. A result that survives BH but not Holm is a lead, not a
finding, and the gap between them is informative rather than embarrassing.

Distinct from deflated.py, which corrects a related but different thing: DSR
asks whether ONE selected strategy survives the fact that N were tried; BH asks
which of the N to declare discoveries at a controlled error rate.
"""
from __future__ import annotations

import math
from statistics import NormalDist

TRADING_DAYS = 252


def _check(pvalues: list[float], alpha: float) -> None:
    """Shared input validation for both procedures."""
    if not pvalues:
        raise ValueError("no p-values supplied")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    for p in pvalues:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value out of range: {p}")


def benjamini_hochberg(
    pvalues: list[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg step-up. Controls the False Discovery Rate.

    Sort ascending, find the largest k with p_(k) <= (k / m) * alpha, and
    reject 1..k — INCLUDING hypotheses whose own p-value exceeds their
    individual threshold. That step-up behaviour is the part implementations
    get wrong: rank among the others decides, not the value alone.

    Args:
        pvalues: p-values in the caller's order.
        alpha:   target FDR.

    Returns:
        (reject, adjusted), both in INPUT ORDER so results zip back onto the
        anomaly names. adjusted[i] = min over j >= i of (m / j) * p_(j),
        capped at 1; the running minimum from the top enforces monotonicity.

    Raises:
        ValueError: empty input, p outside [0, 1], or alpha outside (0, 1).
    """
    _check(pvalues, alpha)
    m = len(pvalues)

    # Positions of the p-values in ascending order. Sorting the INDICES rather
    # than the values is what lets the answer be mapped back to input order.
    order = sorted(range(m), key=lambda i: pvalues[i])

    # Walk from the largest p-value down, keeping a running minimum. That is
    # what enforces monotonicity: an adjusted p-value can never fall as the raw
    # one rises.
    adj_sorted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        p = pvalues[order[rank - 1]]
        running = min(running, m / rank * p)
        adj_sorted[rank - 1] = min(running, 1.0)

    adjusted = [0.0] * m
    for rank, idx in enumerate(order, 1):
        adjusted[idx] = adj_sorted[rank - 1]

    # Deriving `reject` from the adjusted values rather than re-running the
    # step-up search gives the step-up behaviour for free and guarantees the
    # two agree.
    return [a <= alpha + 1e-15 for a in adjusted], adjusted


def holm_bonferroni(
    pvalues: list[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni step-down. Controls the Family-Wise Error Rate.

    Sort ascending and walk up, comparing p_(i) against alpha / (m - i + 1).
    Stop at the first failure, however small the later p-values are.

    Args:
        pvalues: p-values in the caller's order.
        alpha:   target FWER.

    Returns:
        (reject, adjusted) in input order, where adjusted[i] = max over j <= i
        of (m - j + 1) * p_(j), capped at 1 — a running maximum from the
        bottom, mirroring BH's running minimum from the top.

    Raises:
        ValueError: as `benjamini_hochberg`.

    Uniformly more powerful than plain Bonferroni at the same error rate, so
    Bonferroni has no remaining use. Strictly more conservative than BH:
    anything Holm rejects, BH also rejects, and the tests pin that.
    """
    _check(pvalues, alpha)
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])

    # Mirror of BH: walk UP from the smallest p-value keeping a running MAXIMUM.
    # Once a p-value fails, every larger one inherits that failure, which is the
    # step-DOWN rule.
    adj_sorted = [0.0] * m
    running = 0.0
    for rank in range(1, m + 1):
        p = pvalues[order[rank - 1]]
        running = max(running, (m - rank + 1) * p)
        adj_sorted[rank - 1] = min(running, 1.0)

    adjusted = [0.0] * m
    for rank, idx in enumerate(order, 1):
        adjusted[idx] = adj_sorted[rank - 1]

    return [a <= alpha + 1e-15 for a in adjusted], adjusted


def sharpe_pvalue(
    sr_annual: float,
    n: int,
    periods_per_year: int = TRADING_DAYS,
    alternative: str = "greater",
) -> float:
    """p-value for the null that the true Sharpe is zero.

        t = (sr_annual / sqrt(periods_per_year)) * sqrt(n)

    and p is read off the standard normal.

    Args:
        sr_annual:   ANNUALISED Sharpe, straight from `metrics.sharpe`. The one
                     function here taking an annualised input, since it is the
                     bridge from metrics.py; it de-annualises internally.
        n:           observations behind that Sharpe.
        alternative: "greater" (default — losing money is not a discovery) or
                     "two-sided".

    Returns:
        p-value in [0, 1].

    Raises:
        ValueError: n < 2, or an unrecognised `alternative`.

    DELIBERATE LIMITATION: this is the naive test. It assumes normal,
    independent returns and ignores selection bias and the higher moments.
    `deflated.probabilistic_sharpe` is the better instrument for a single
    strategy. This exists only to give BH and Holm comparable p-values across
    the six anomalies, and must never be the headline number.
    """
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}")
    if alternative not in ("greater", "two-sided"):
        raise ValueError(f"unknown alternative: {alternative}")

    # De-annualise, then scale by sample length. At exactly one year the two
    # cancel, so a Sharpe of 1.0 over 252 days is a one-sigma result.
    t = (sr_annual / math.sqrt(periods_per_year)) * math.sqrt(n)

    if alternative == "greater":
        return 1.0 - NormalDist().cdf(t)
    return 2.0 * (1.0 - NormalDist().cdf(abs(t)))
