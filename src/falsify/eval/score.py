"""The scoring rule: did the pipeline rediscover the published effect, or not?

    Sign, then significance, then multiplicity.
    Magnitude is reported, never gated -- except downward.

This is the file the project's premise rests on. "How did you decide an anomaly
had been rediscovered?" is the question every claim in the README reduces to,
and the answer has to be four rules in code rather than a paragraph of
judgement, or the eval suite is a narrative.

THE FOUR RULES, in the order they are applied and in the order they matter.

1. SIGN. Did the spread run the way the paper predicted? `run_backtest` always
   goes long the top bucket and short the bottom, so its Sharpe is the sign of
   the top-minus-bottom spread, and FOUR of the six anomalies predict that
   spread to be NEGATIVE -- the majority of the suite. The registry fixes the
   direction in advance; the scorer multiplies by it. Everything downstream uses the ORIENTED Sharpe,
   including the p-value: testing the raw spread one-sided would score a
   direction=-1 anomaly as a failure precisely when it worked.

2. DEFLATION. Does it survive the expected best-of-N-trials benchmark, at
   PSR >= 0.95, for the number of strategies the SUITE actually tried? Six
   anomalies is six trials whether or not all six are reported.
   A MISSING DEFLATION STATISTIC IS NOT A SATISFIED CONDITION. This is the
   Module 5 lesson applied again: silence is not consent. An anomaly whose
   series was too short for the statistics fails the gate; it does not skip it.

3. MULTIPLICITY. Does it survive Benjamini-Hochberg across the suite? A p-value
   of 0.04 means something different as one of six than it does alone.
   Anomalies that could not be tested are EXCLUDED from the correction rather
   than counted as nulls: padding the denominator with untested hypotheses
   would make the survivors look better, which is a way of profiting from a
   short sample.

4. THE EMBARRASSMENT CHECK. A spread more than `EMBARRASSMENT_MULTIPLE` times
   the published reference DOWNGRADES a pass to partial, and says so. On a
   sample this short a huge number is likelier a defect than a discovery, and
   an eval suite that cannot be embarrassed by its own best result is not
   measuring anything. It only ever moves a verdict DOWNWARD -- it can never
   turn a fail into a pass.

MAGNITUDE IS NOT A GATE. Matching a published Sharpe across a different sample,
universe, horizon and construction is not achievable, and an eval that demanded
it would fail everything, including the effects that are real. The realised and
published numbers are both reported and their ratio is printed; only the
3x direction is acted on.

FOUR OUTCOMES, NOT THREE, and the separation is the honest part.
`insufficient_data` is not `fail`. An anomaly that needs three years of history
on a two-year panel has not been refuted -- it has not been tested. Collapsing
the two would let a short sample manufacture rejections, and this suite runs on
a short sample.

EVERY VERDICT CARRIES REASONS, including a pass, where the reasons are what was
checked and survived. Same rule as the Module 5 `publishable` property: a count
with nothing attached is a number nobody can act on, and the demo shows the
reasons rather than the tally.

`SCORING_RULE_VERSION` is stored beside `registry_sha` on every row. Together
they say "this prediction, judged by this rule". Neither alone would reveal
that two rows in the same table had been held to different standards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from falsify.eval.registry import ANOMALIES, Anomaly
from falsify.stats.multipletest import benjamini_hochberg

# Bump when any of the four rules changes. Stored on every verdict.
SCORING_RULE_VERSION = 1

Verdict = Literal["pass", "partial", "fail", "insufficient_data"]
VERDICTS: tuple[Verdict, ...] = ("pass", "partial", "fail", "insufficient_data")

# PSR against the best-of-N-trials benchmark must reach this to clear rule 2.
DEFLATION_THRESHOLD = 0.95

# Target false discovery rate for rule 3.
FDR_ALPHA = 0.05

# Rule 4. A realised spread this many times the published reference downgrades
# a pass. Three is a judgement call and is deliberately loose: the point is to
# catch a result that is absurd, not to second-guess one that is merely good.
EMBARRASSMENT_MULTIPLE = 3.0


@dataclass(frozen=True)
class Measurement:
    """What the runner measured for one anomaly. Nothing here is a judgement.

    realised_sharpe:  ANNUALISED Sharpe of the long/short spread, SIGNED AS
                      MEASURED. Kept unoriented so a reader can reconcile it
                      against the raw backtest output.
    p_value:          one-sided p-value on the ORIENTED Sharpe. None when the
                      series was too short to compute one.
    deflated_psr:     `prob_beats_best_of_n_trials` from `analyze_results`, the
                      probability the true Sharpe beats the best of n_trials
                      worthless strategies. None when unavailable -- which is a
                      failed gate, not a skipped one.
    history_days:     trading days of panel history available, compared against
                      the anomaly's `min_history_days`.
    n_invested_days:  days the strategy actually held a position.
    tested:           False when the feature produced no values at all, e.g.
                      `rev_36_12` on a two-year panel.
    note:             why it could not be tested, when it could not.
    """

    key: str
    realised_sharpe: float | None = None
    p_value: float | None = None
    deflated_psr: float | None = None
    history_days: int = 0
    n_invested_days: int = 0
    n_trials: int = 0
    tested: bool = True
    note: str = ""


@dataclass(frozen=True)
class Score:
    """One scored anomaly: the verdict, the numbers behind it, and the reasons."""

    key: str
    verdict: Verdict
    reasons: tuple[str, ...]
    direction: int
    realised_sharpe: float | None
    oriented_sharpe: float | None
    published_sharpe: float
    sharpe_ratio_to_published: float | None
    p_value: float | None
    p_value_adjusted: float | None
    deflated_psr: float | None
    n_invested_days: int
    history_days: int
    min_history_days: int
    n_trials: int
    scoring_rule_version: int = SCORING_RULE_VERSION
    caveat: str = ""

    @property
    def counted_in_multiplicity(self) -> bool:
        """Whether this anomaly entered the BH denominator."""
        return self.verdict != "insufficient_data"

    def to_dict(self) -> dict:
        d = {
            "key": self.key,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "direction": self.direction,
            "realised_sharpe": self.realised_sharpe,
            "oriented_sharpe": self.oriented_sharpe,
            "published_sharpe": self.published_sharpe,
            "sharpe_ratio_to_published": self.sharpe_ratio_to_published,
            "p_value": self.p_value,
            "p_value_adjusted": self.p_value_adjusted,
            "deflated_psr": self.deflated_psr,
            "n_invested_days": self.n_invested_days,
            "history_days": self.history_days,
            "min_history_days": self.min_history_days,
            "n_trials": self.n_trials,
            "scoring_rule_version": self.scoring_rule_version,
            "caveat": self.caveat,
        }
        return d


@dataclass(frozen=True)
class SuiteScore:
    """Every anomaly's verdict plus the suite-level context that produced them."""

    scores: tuple[Score, ...]
    n_tested: int
    n_untestable: int
    fdr_alpha: float
    deflation_threshold: float
    embarrassment_multiple: float
    scoring_rule_version: int = SCORING_RULE_VERSION
    _by_key: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_key", {s.key: s for s in self.scores})

    def __getitem__(self, key: str) -> Score:
        return self._by_key[key]

    def __iter__(self):
        return iter(self.scores)

    def __len__(self) -> int:
        return len(self.scores)

    def tally(self) -> dict[str, int]:
        """Counts per verdict. Every outcome appears, including the zeroes.

        A verdict missing from a tally reads as "not applicable" when it means
        "none this time", and the difference matters most for
        `insufficient_data`, whose absence would hide that the suite ran on
        enough history.
        """
        out = {v: 0 for v in VERDICTS}
        for s in self.scores:
            out[s.verdict] += 1
        return out


def _oriented(realised: float, direction: int) -> float:
    """The spread re-signed so that positive always means "as predicted"."""
    return realised * direction


def score_anomaly(
    measurement: Measurement,
    anomaly: Anomaly,
    p_value_adjusted: float | None,
    deflation_threshold: float = DEFLATION_THRESHOLD,
    fdr_alpha: float = FDR_ALPHA,
    embarrassment_multiple: float = EMBARRASSMENT_MULTIPLE,
) -> Score:
    """Apply the four rules to one anomaly.

    Args:
        measurement: what the runner observed.
        anomaly: the pre-registered prediction.
        p_value_adjusted: the BH-adjusted p-value from `score_suite`, or None
            for an untestable anomaly that never entered the correction.

    Returns:
        A Score. Never raises on a bad measurement: an anomaly that could not
        be measured is `insufficient_data` with the reason attached, because a
        suite that throws on its weakest case reports nothing about the other
        five.

    The rules are applied in order and the first failure is terminal, EXCEPT
    rule 4, which is evaluated only after the first three have all passed and
    can only move the verdict down.
    """
    common = {
        "key": anomaly.key,
        "direction": anomaly.direction,
        "realised_sharpe": measurement.realised_sharpe,
        "published_sharpe": anomaly.published_sharpe,
        "p_value": measurement.p_value,
        "p_value_adjusted": p_value_adjusted,
        "deflated_psr": measurement.deflated_psr,
        "n_invested_days": measurement.n_invested_days,
        "history_days": measurement.history_days,
        "min_history_days": anomaly.min_history_days,
        "n_trials": measurement.n_trials,
        "caveat": anomaly.caveat,
    }

    # --- rule 0: is this testable at all? --------------------------------
    # Deliberately NOT a failure. An anomaly needing three years of history on
    # a two-year panel has not been refuted; it has not been tested, and
    # collapsing the two lets a short sample manufacture rejections.
    if not measurement.tested or measurement.realised_sharpe is None:
        reason = measurement.note or "the feature produced no values on this panel"
        return Score(
            verdict="insufficient_data",
            reasons=(f"not tested: {reason}",),
            oriented_sharpe=None,
            sharpe_ratio_to_published=None,
            **common,
        )
    if measurement.history_days < anomaly.min_history_days:
        return Score(
            verdict="insufficient_data",
            reasons=(
                f"needs {anomaly.min_history_days} trading days of history per ticker; "
                f"the panel has {measurement.history_days}",
            ),
            oriented_sharpe=None,
            sharpe_ratio_to_published=None,
            **common,
        )
    if measurement.n_invested_days < 1:
        return Score(
            verdict="insufficient_data",
            reasons=("the strategy never held a position on this panel",),
            oriented_sharpe=None,
            sharpe_ratio_to_published=None,
            **common,
        )

    oriented = _oriented(measurement.realised_sharpe, anomaly.direction)
    ratio = oriented / anomaly.published_sharpe if anomaly.published_sharpe else None
    common = {**common, "oriented_sharpe": oriented, "sharpe_ratio_to_published": ratio}

    direction_word = "top bucket" if anomaly.direction == 1 else "bottom bucket"
    reasons: list[str] = []

    # --- rule 1: sign ----------------------------------------------------
    if oriented <= 0:
        return Score(
            verdict="fail",
            reasons=(
                f"sign: the paper predicts the {direction_word} wins, and the oriented "
                f"spread was {oriented:+.3f}. The effect ran the wrong way on this "
                "sample.",
            ),
            **common,
        )
    reasons.append(
        f"sign: oriented spread {oriented:+.3f} runs as predicted "
        f"({direction_word} wins)"
    )

    # --- rule 2: deflation -----------------------------------------------
    # SILENCE IS NOT CONSENT. A missing statistic fails the gate.
    if measurement.deflated_psr is None:
        return Score(
            verdict="fail",
            reasons=(
                *reasons,
                "deflation: no deflated Sharpe was computed for this anomaly, so the "
                "gate is unmet. A missing statistic is not a satisfied condition.",
            ),
            **common,
        )
    if measurement.deflated_psr < deflation_threshold:
        return Score(
            verdict="fail",
            reasons=(
                *reasons,
                f"deflation: probability of beating the best of {measurement.n_trials} "
                f"trials is {measurement.deflated_psr:.3f}, below the "
                f"{deflation_threshold:.2f} gate. The evidence does not survive the "
                "number of strategies this suite tried.",
            ),
            **common,
        )
    reasons.append(
        f"deflation: beats the best-of-{measurement.n_trials} benchmark with "
        f"probability {measurement.deflated_psr:.3f} >= {deflation_threshold:.2f}"
    )

    # --- rule 3: multiplicity --------------------------------------------
    if p_value_adjusted is None:
        return Score(
            verdict="fail",
            reasons=(
                *reasons,
                "multiplicity: no p-value was available, so the Benjamini-Hochberg "
                "correction could not be applied. Untested is not passed.",
            ),
            **common,
        )
    if p_value_adjusted > fdr_alpha:
        return Score(
            verdict="fail",
            reasons=(
                *reasons,
                f"multiplicity: BH-adjusted p-value {p_value_adjusted:.4f} exceeds "
                f"alpha {fdr_alpha:.2f} across the tested anomalies. Significant alone "
                "is not significant as one of several.",
            ),
            **common,
        )
    reasons.append(
        f"multiplicity: survives Benjamini-Hochberg at alpha {fdr_alpha:.2f} "
        f"(adjusted p = {p_value_adjusted:.4f})"
    )

    # --- rule 4: the embarrassment check ---------------------------------
    # Only ever downward. A result cannot be promoted by being large.
    if ratio is not None and ratio > embarrassment_multiple:
        return Score(
            verdict="partial",
            reasons=(
                *reasons,
                f"DOWNGRADED: the oriented spread is {ratio:.1f}x the published "
                f"reference of {anomaly.published_sharpe:.2f}, above the "
                f"{embarrassment_multiple:.0f}x plausibility ceiling. On a sample this "
                "short a result this large is likelier a defect than a discovery, so "
                "it is reported as partial pending a longer history.",
            ),
            **common,
        )

    reasons.append(
        f"magnitude: {oriented:.2f} against a published reference of "
        f"{anomaly.published_sharpe:.2f} ({ratio:.1f}x), reported not gated"
    )
    return Score(verdict="pass", reasons=tuple(reasons), **common)


def score_suite(
    measurements: list[Measurement],
    anomalies: tuple[Anomaly, ...] = ANOMALIES,
    fdr_alpha: float = FDR_ALPHA,
    deflation_threshold: float = DEFLATION_THRESHOLD,
    embarrassment_multiple: float = EMBARRASSMENT_MULTIPLE,
) -> SuiteScore:
    """Score every anomaly, with the multiplicity correction taken across the suite.

    Args:
        measurements: one per anomaly. Order is irrelevant; matched by key.
        anomalies: the pre-registered predictions.

    Returns:
        A SuiteScore whose `scores` are in registry order.

    Raises:
        KeyError: a registered anomaly has no measurement. Scoring a partial
            suite as though it were the whole one would under-count the
            trials and under-correct the p-values, which is the failure this
            whole module exists to prevent.

    THE ONE SUITE-LEVEL DECISION: only TESTABLE anomalies enter the BH
    correction. An anomaly with no p-value cannot be a null hypothesis that
    failed to reject -- it is a hypothesis that was never examined. Counting it
    would inflate `m` in the (k/m)*alpha threshold and make every survivor look
    stronger for the sole reason that something else could not be run.
    """
    by_key = {m.key: m for m in measurements}
    missing = [a.key for a in anomalies if a.key not in by_key]
    if missing:
        raise KeyError(
            f"no measurement for {', '.join(missing)}. Score the whole suite or none "
            "of it: a partial suite under-corrects every p-value in it."
        )

    # Testability is decided here, with the same predicates score_anomaly uses,
    # so the BH denominator and the verdicts cannot disagree.
    def _testable(a: Anomaly) -> bool:
        m = by_key[a.key]
        return (
            m.tested
            and m.realised_sharpe is not None
            and m.p_value is not None
            and m.history_days >= a.min_history_days
            and m.n_invested_days >= 1
        )

    testable = [a for a in anomalies if _testable(a)]
    adjusted: dict[str, float] = {}
    if testable:
        pvals = [by_key[a.key].p_value for a in testable]
        _reject, adj = benjamini_hochberg(pvals, alpha=fdr_alpha)
        adjusted = {a.key: adj[i] for i, a in enumerate(testable)}

    scores = tuple(
        score_anomaly(
            by_key[a.key],
            a,
            adjusted.get(a.key),
            deflation_threshold=deflation_threshold,
            fdr_alpha=fdr_alpha,
            embarrassment_multiple=embarrassment_multiple,
        )
        for a in anomalies
    )

    n_untestable = sum(1 for s in scores if s.verdict == "insufficient_data")
    return SuiteScore(
        scores=scores,
        n_tested=len(scores) - n_untestable,
        n_untestable=n_untestable,
        fdr_alpha=fdr_alpha,
        deflation_threshold=deflation_threshold,
        embarrassment_multiple=embarrassment_multiple,
    )


__all__ = [
    "DEFLATION_THRESHOLD",
    "EMBARRASSMENT_MULTIPLE",
    "FDR_ALPHA",
    "SCORING_RULE_VERSION",
    "VERDICTS",
    "Measurement",
    "Score",
    "SuiteScore",
    "Verdict",
    "score_anomaly",
    "score_suite",
]
