"""Run the six anomalies through the SAME tools the agent uses, and score them.

No Claude call happens here, and that is a deliberate architectural choice
rather than a cost saving. The eval suite's question is "does this PIPELINE
rediscover published effects", not "does the model drive it well". Putting a
language model in the loop would mix the two, and a failing row would be
unattributable: a bad verdict could mean the anomaly is absent from the data or
that the model chose eleven buckets.

What it does NOT skip is the tool boundary. Every step goes through
`agent.tools.dispatch`-level functions, so the eval suite exercises exactly the
code path a real run takes: the same closed feature menu, the same universe
mask applied at the same point, the same invested-window metrics. An eval suite
that reached past the boundary into the engine would be testing code the agent
does not run.

TWO DECISIONS CARRY THE STATISTICS, and both are places where the obvious
implementation quietly reports a better number.

  ONE SESSION FOR THE WHOLE SUITE. `Session.n_backtests` is the trial count for
  deflation, and it is per-Session. A fresh Session per anomaly would make
  every anomaly the only trial that ever ran, N=1, no deflation at all, six
  times over -- and every row would still look like a deflated result. The
  suite shares one Session precisely so that each anomaly is penalised for the
  other five having been tried.

  ANALYSIS AFTER THE LAST BACKTEST, NEVER AS THEY FINISH. The trial count is
  taken WHEN `analyze_results` IS CALLED. Analysing anomaly one immediately
  after running it deflates it against N=1, anomaly two against N=2, and so on:
  six rows held to six different standards, ordered by nothing more meaningful
  than the order of the registry tuple. So every backtest runs first, and only
  then is every one analysed, at the final count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from falsify.agent import tools as T
from falsify.agent.session import Session
from falsify.eval.registry import ANOMALIES, Anomaly, registry_digest
from falsify.eval.score import Measurement, SuiteScore, score_suite
from falsify.stats.multipletest import sharpe_pvalue

DEFAULT_N_BUCKETS = 10
DEFAULT_COST_BPS = 10.0


@dataclass
class SuiteRun:
    """Everything one pass over the suite produced.

    `score` is the verdicts. `measurements` is what they were computed from.
    `backtests` and `analyses` are the raw tool outputs, kept so a row in the
    results table can be traced back to the call that produced it without
    re-running anything.
    """

    universe: str
    registry_sha: str
    score: SuiteScore
    measurements: list[Measurement]
    panel: dict[str, Any]
    backtests: dict[str, dict] = field(default_factory=dict)
    analyses: dict[str, dict] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    n_trials: int = 0
    trial_variance_assumed: float = T.TRIAL_VARIANCE
    trial_variance_measured: float | None = None


def _history_days(panel_summary: dict) -> int:
    """Trading days in the panel: the denominator every min_history_days meets.

    Panel-wide rather than per-ticker. A per-ticker minimum would be the
    stricter reading, but a single newly-listed name would then make the whole
    anomaly untestable, and the feature already returns nulls for names with
    too little of their own history. This is the length of the window the
    suite had to work with.
    """
    return int(panel_summary.get("n_trading_days", 0))


def run_suite(
    universe: str = "current",
    anomalies: tuple[Anomaly, ...] = ANOMALIES,
    n_buckets: int = DEFAULT_N_BUCKETS,
    cost_bps: float = DEFAULT_COST_BPS,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    fetch: Callable[..., dict] | None = None,
) -> SuiteRun:
    """Run every anomaly on one universe and score the suite.

    Args:
        universe: "current" or "point_in_time", passed straight to fetch_data.
        anomalies: the pre-registered suite.
        n_buckets: cross-sectional buckets, the same for every anomaly. Held
            fixed on purpose: varying it per anomaly would be a free parameter
            chosen after seeing the data, and six anomalies at three bucket
            counts is eighteen trials, not six.
        cost_bps: one-way transaction cost, likewise fixed across the suite.
        fetch: injection point for tests. Defaults to `tools.fetch_data`.

    Returns:
        A SuiteRun.

    Raises:
        T.ToolError: only from the initial fetch. A per-anomaly failure is
            recorded as an untestable measurement with the tool's own message
            attached, because one anomaly that cannot run must not cost the
            other five their verdicts.
    """
    session = Session()
    fetch = fetch or T.fetch_data

    panel = fetch(session, tickers=tickers, start=start, end=end, universe=universe)
    panel_handle = panel["handle"]
    n_trading_days = len(session.payload(panel_handle, "panel").frame["ts"].unique())
    panel = {**panel, "n_trading_days": n_trading_days}

    backtests: dict[str, dict] = {}
    errors: dict[str, str] = {}

    # --- pass one: every backtest, before any analysis -------------------
    for a in anomalies:
        try:
            fh = T.compute_feature(session, panel_handle, a.feature)["handle"]
            backtests[a.key] = T.run_backtest(
                session,
                fh,
                n_buckets=n_buckets,
                long_short=True,      # the spread IS the test of a ranking
                cost_bps=cost_bps,
            )
        except T.ToolError as exc:
            errors[a.key] = str(exc)

    # --- pass two: analyse, now that the trial count is final ------------
    n_trials = session.n_backtests
    analyses: dict[str, dict] = {}
    for key, bt in backtests.items():
        try:
            analyses[key] = T.analyze_results(session, bt["handle"])
        except T.ToolError as exc:
            errors[key] = str(exc)

    measurements = [
        _measure(a, backtests.get(a.key), analyses.get(a.key), errors.get(a.key),
                 n_trading_days, n_trials)
        for a in anomalies
    ]

    return SuiteRun(
        universe=universe,
        registry_sha=registry_digest(anomalies),
        score=score_suite(measurements, anomalies),
        measurements=measurements,
        panel=panel,
        backtests=backtests,
        analyses=analyses,
        errors=errors,
        n_trials=n_trials,
        trial_variance_measured=measured_trial_variance(analyses),
    )


def _measure(
    anomaly: Anomaly,
    backtest: dict | None,
    analysis: dict | None,
    error: str | None,
    n_trading_days: int,
    n_trials: int,
) -> Measurement:
    """Turn raw tool output into the input the scorer takes. No judgement here.

    The one piece of arithmetic is the p-value, and it is computed on the
    ORIENTED Sharpe. `sharpe_pvalue` is one-sided "greater" by default, so
    passing the raw spread would score a direction=-1 anomaly as insignificant
    exactly when its effect was strongest.
    """
    if backtest is None:
        return Measurement(
            key=anomaly.key,
            history_days=n_trading_days,
            n_trials=n_trials,
            tested=False,
            note=error or "no backtest was produced",
        )

    realised = backtest.get("sharpe")
    n_invested = int(backtest.get("n_invested_days", 0))
    oriented = None if realised is None else realised * anomaly.direction

    p = None
    if oriented is not None and n_invested >= 2:
        try:
            p = sharpe_pvalue(oriented, n_invested, alternative="greater")
        except ValueError:
            p = None

    deflated = None if analysis is None else analysis.get("prob_beats_best_of_n_trials")

    return Measurement(
        key=anomaly.key,
        realised_sharpe=realised,
        p_value=p,
        deflated_psr=deflated,
        history_days=n_trading_days,
        n_invested_days=n_invested,
        n_trials=n_trials,
        tested=realised is not None and realised == realised,  # NaN is not tested
        note=error or "",
    )


def measured_trial_variance(analyses: dict[str, dict]) -> float | None:
    """Variance of the PER-PERIOD Sharpes across the suite's own trials.

    `deflated.expected_max_sharpe` takes the variance of the estimated Sharpes
    ACROSS trials, and `tools.TRIAL_VARIANCE` is currently an assumption of
    0.0009 carried since Module 4. This measures the real thing from the six
    anomalies so the two can be printed side by side.

    IT IS REPORTED, NOT SUBSTITUTED. Replacing a labelled assumption with an
    unlabelled estimate from six points is not an improvement; six is far too
    few for a variance, and the substitution would be invisible in every number
    downstream of it. Revisit after the backfill, when the six run on five
    years instead of two.

    Per-period, not annualised, because that is the unit
    `expected_max_sharpe` consumes -- an annualised variance is 252 times too
    large and would inflate the benchmark enormously.
    """
    sharpes = [
        a["sharpe_per_period"]
        for a in analyses.values()
        if a.get("sharpe_per_period") is not None
    ]
    if len(sharpes) < 2:
        return None
    mean = sum(sharpes) / len(sharpes)
    return sum((s - mean) ** 2 for s in sharpes) / (len(sharpes) - 1)


def compare_universes(
    anomalies: tuple[Anomaly, ...] = ANOMALIES,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the whole suite on BOTH universes and report the gap per anomaly.

    Returns:
        {"current": SuiteRun, "point_in_time": SuiteRun, "gaps": [...]}.

    This is the Module 3 survivorship measurement generalised from one strategy
    to six, and it is a question the project could not previously ask: whether
    the roughly-half-the-return result holds across every anomaly, or whether
    some are barely touched by it. A survivorship gap that appears on momentum
    and on nothing else is a different finding from one that appears on all
    six, and only the second supports a general claim.
    """
    runs = {
        u: run_suite(universe=u, anomalies=anomalies, **kwargs)
        for u in ("current", "point_in_time")
    }
    gaps = []
    for a in anomalies:
        cur = runs["current"].score[a.key]
        pit = runs["point_in_time"].score[a.key]
        gap = (
            None
            if cur.oriented_sharpe is None or pit.oriented_sharpe is None
            else cur.oriented_sharpe - pit.oriented_sharpe
        )
        gaps.append(
            {
                "key": a.key,
                "current_sharpe": cur.oriented_sharpe,
                "pit_sharpe": pit.oriented_sharpe,
                "sharpe_gap": gap,
                "current_verdict": cur.verdict,
                "pit_verdict": pit.verdict,
            }
        )
    return {**runs, "gaps": gaps}


__all__ = [
    "SuiteRun",
    "compare_universes",
    "measured_trial_variance",
    "run_suite",
]
