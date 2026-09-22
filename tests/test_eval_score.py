"""The scoring rule. This is the file the project's premise rests on.

Grouped by the four rules. The ones that discriminate hardest, and the ones an
interviewer would go for:

    test_a_negative_direction_anomaly_passes_on_a_negative_spread
    test_a_missing_deflation_statistic_fails_rather_than_skips
    test_untestable_anomalies_do_not_pad_the_multiplicity_denominator
    test_insufficient_data_is_not_a_failure
    test_the_embarrassment_check_only_moves_a_verdict_downward

Every expectation here is derived from the rule, never read off an
implementation. Where a number is asserted it is an analytic identity: BH with
one hypothesis leaves the p-value unchanged; BH with m hypotheses at rank m
multiplies by 1.
"""
from __future__ import annotations

import pytest

from falsify.eval import registry as R
from falsify.eval.score import (
    DEFLATION_THRESHOLD,
    EMBARRASSMENT_MULTIPLE,
    FDR_ALPHA,
    SCORING_RULE_VERSION,
    VERDICTS,
    Measurement,
    score_anomaly,
    score_suite,
)

PASSING = dict(
    realised_sharpe=0.8,
    p_value=0.001,
    deflated_psr=0.99,
    history_days=800,
    n_invested_days=250,
    n_trials=6,
)


def m(key: str, **over) -> Measurement:
    return Measurement(key=key, **{**PASSING, **over})


def one(anomaly_key: str, adj: float | None = 0.001, **over):
    """Score a single anomaly against a supplied adjusted p-value."""
    a = R.get(anomaly_key)
    return score_anomaly(m(anomaly_key, **over), a, adj)


# --- rule 1: sign -----------------------------------------------------------


def test_a_positive_direction_anomaly_passes_on_a_positive_spread():
    s = one("momentum_12_1", realised_sharpe=0.8)
    assert s.verdict == "pass"
    assert s.oriented_sharpe == pytest.approx(0.8)


def test_a_negative_direction_anomaly_passes_on_a_negative_spread():
    """THE TEST DIRECTION EXISTS FOR.

    low_volatility predicts the BOTTOM bucket wins, so a raw spread of -0.8 is
    the effect working. A scorer that ignored direction would call this a
    failure precisely when the anomaly was strongest.
    """
    s = one("low_volatility", realised_sharpe=-0.8)
    assert s.verdict == "pass"
    assert s.oriented_sharpe == pytest.approx(0.8)
    assert s.realised_sharpe == pytest.approx(-0.8), "stored signed as measured"


def test_a_negative_direction_anomaly_fails_on_a_positive_spread():
    s = one("low_volatility", realised_sharpe=+0.8)
    assert s.verdict == "fail"
    assert "sign" in s.reasons[0]
    assert "bottom bucket" in s.reasons[0]


def test_a_zero_spread_fails_the_sign_gate():
    """Not 'insufficient'. A measured zero is a measurement."""
    assert one("momentum_12_1", realised_sharpe=0.0).verdict == "fail"


def test_the_sign_failure_is_terminal_even_with_perfect_statistics():
    s = one("momentum_12_1", realised_sharpe=-2.0, deflated_psr=1.0, p_value=0.0)
    assert s.verdict == "fail"
    assert len(s.reasons) == 1, "nothing downstream of a wrong sign is worth reporting"


# --- rule 2: deflation ------------------------------------------------------


def test_a_missing_deflation_statistic_fails_rather_than_skips():
    """SILENCE IS NOT CONSENT -- the Module 5 lesson, applied again.

    The tempting implementation treats `deflated_psr is None` as "no objection"
    and moves on. That turns every anomaly whose series was too short for the
    statistics into a candidate pass, which is the opposite of what a short
    series should do.
    """
    s = one("momentum_12_1", deflated_psr=None)
    assert s.verdict == "fail"
    assert any("missing statistic is not a satisfied condition" in r for r in s.reasons)


def test_deflation_below_the_gate_fails():
    s = one("momentum_12_1", deflated_psr=DEFLATION_THRESHOLD - 0.01)
    assert s.verdict == "fail"
    assert any("best of 6" in r for r in s.reasons)


def test_deflation_exactly_at_the_gate_passes():
    assert one("momentum_12_1", deflated_psr=DEFLATION_THRESHOLD).verdict == "pass"


def test_the_deflation_reason_names_the_trial_count():
    """A deflated Sharpe with no N attached cannot be audited."""
    s = one("momentum_12_1", n_trials=11)
    assert any("best-of-11" in r for r in s.reasons)


# --- rule 3: multiplicity ---------------------------------------------------


def test_an_adjusted_p_above_alpha_fails():
    s = one("momentum_12_1", adj=FDR_ALPHA + 0.001)
    assert s.verdict == "fail"
    assert any("BH-adjusted" in r for r in s.reasons)


def test_a_missing_adjusted_p_fails():
    assert one("momentum_12_1", adj=None).verdict == "fail"


def test_multiplicity_is_computed_across_the_suite_not_per_anomaly():
    """A p-value of 0.02 clears alpha alone and must not clear it as one of six.

    The identity: BH adjusts the smallest of m p-values by m/1. With six
    testable anomalies, 0.02 becomes 0.12, which fails at alpha 0.05. A scorer
    that corrected nothing, or corrected each anomaly against itself, would
    report a pass here.
    """
    ms = [m(R.ANOMALIES[0].key, p_value=0.02)] + [
        m(a.key, realised_sharpe=0.8 * a.direction, p_value=0.9)
        for a in R.ANOMALIES[1:]
    ]
    suite = score_suite(ms)
    assert suite.n_tested == 6
    assert suite[R.ANOMALIES[0].key].p_value_adjusted == pytest.approx(0.12)
    assert suite[R.ANOMALIES[0].key].verdict == "fail"


def test_the_same_p_value_passes_when_it_is_the_only_hypothesis_tested():
    """The mirror image, and it is what makes the previous test about
    multiplicity rather than about the number 0.02.

    One testable anomaly: BH adjusts by 1/1 and 0.02 survives. The verdict
    changed because of how many OTHER hypotheses were examined, which is the
    whole content of rule 3.
    """
    ms = [m(R.ANOMALIES[0].key, p_value=0.02)] + [
        Measurement(a.key, tested=False, history_days=800, n_trials=6)
        for a in R.ANOMALIES[1:]
    ]
    suite = score_suite(ms)
    assert suite.n_tested == 1
    assert suite[R.ANOMALIES[0].key].p_value_adjusted == pytest.approx(0.02)
    assert suite[R.ANOMALIES[0].key].verdict == "pass"


def test_untestable_anomalies_do_not_pad_the_multiplicity_denominator():
    """Padding m would make every survivor look stronger for no reason.

    Two testable anomalies with p=0.02 and p=0.9: m=2, so the smaller adjusts
    to 0.02 * 2/1 = 0.04 and survives. If the four untestable ones were counted
    as nulls, m=6 and it would adjust to 0.12 and fail -- a verdict changed by
    hypotheses that were never examined.
    """
    testable = ["momentum_12_1", "low_volatility"]
    ms = []
    for a in R.ANOMALIES:
        if a.key == "momentum_12_1":
            ms.append(m(a.key, p_value=0.02))
        elif a.key == "low_volatility":
            ms.append(m(a.key, realised_sharpe=-0.8, p_value=0.9))
        else:
            ms.append(Measurement(a.key, tested=False, history_days=800, n_trials=6))
    suite = score_suite(ms)
    assert suite.n_tested == 2
    assert suite.n_untestable == 4
    assert suite["momentum_12_1"].p_value_adjusted == pytest.approx(0.04)
    assert suite["momentum_12_1"].verdict == "pass"


def test_an_anomaly_with_a_p_value_but_too_little_history_is_still_excluded(): 
    """THE TEST THE FIRST INJECTION PASS MISSED, and the reason it missed.

    The original version of the test above used measurements with no p-value at
    all, so an implementation that filtered on `p_value is not None` alone --
    the tempting shortcut -- excluded exactly the same anomalies and the suite
    stayed green. It proved nothing about the rule it claimed to defend.

    The case that discriminates is an anomaly that HAS a p-value and is
    untestable for a DIFFERENT reason: too little history. That is not a
    contrived case, it is the state of `long_term_reversal` on the panel this
    suite actually runs on.

    The arithmetic, hand-derived. On 100 days of history only the three
    63-day-or-less anomalies are testable, so m = 3 and the smallest p adjusts
    to 0.015 * 3/1 = 0.045, which clears alpha. Counting the three untestable
    ones as nulls makes m = 6 and the same p-value adjusts to 0.09, which does
    not. A verdict changed by hypotheses that were never examined.
    """
    short_history = 100
    ms = []
    for a in R.ANOMALIES:
        p_val = 0.015 if a.key == "short_term_reversal" else 0.9
        ms.append(
            m(a.key,
              realised_sharpe=0.8 * a.direction,
              p_value=p_val,
              history_days=short_history)
        )
    suite = score_suite(ms)

    testable = {s.key for s in suite if s.verdict != "insufficient_data"}
    assert testable == {
        "short_term_reversal", "low_volatility", "idiosyncratic_volatility"
    }, "only the anomalies whose min_history_days fits in 100 days are testable"

    assert suite["short_term_reversal"].p_value_adjusted == pytest.approx(0.045)
    assert suite["short_term_reversal"].verdict == "pass"

    # And the untestable ones supplied a p-value that was deliberately ignored.
    assert suite["momentum_12_1"].verdict == "insufficient_data"
    assert suite["momentum_12_1"].p_value_adjusted is None


def test_untestable_anomalies_carry_no_adjusted_p_at_all():
    ms = [m(a.key) for a in R.ANOMALIES]
    ms[2] = Measurement(R.ANOMALIES[2].key, tested=False, history_days=800)
    suite = score_suite(ms)
    assert suite[R.ANOMALIES[2].key].p_value_adjusted is None


# --- rule 0: four outcomes, not three ---------------------------------------


def test_insufficient_data_is_not_a_failure():
    """The separation that stops a short sample manufacturing rejections."""
    s = score_anomaly(
        Measurement("long_term_reversal", tested=False, history_days=800,
                    note="rev_36_12 produced no values on this panel"),
        R.get("long_term_reversal"),
        None,
    )
    assert s.verdict == "insufficient_data"
    assert s.verdict != "fail"
    assert "not tested" in s.reasons[0]


def test_history_shorter_than_the_registered_minimum_is_insufficient_not_fail():
    """The two-year panel this suite currently runs on, in one test.

    long_term_reversal needs 756 trading days. On 500 it has not been refuted;
    it has not been tested, and the reason names both numbers.
    """
    s = one("long_term_reversal", realised_sharpe=-0.9, history_days=500)
    assert s.verdict == "insufficient_data"
    assert "756" in s.reasons[0] and "500" in s.reasons[0]


def test_a_strategy_that_never_held_a_position_is_insufficient():
    assert one("momentum_12_1", n_invested_days=0).verdict == "insufficient_data"


def test_all_four_outcomes_are_reachable():
    assert set(VERDICTS) == {"pass", "partial", "fail", "insufficient_data"}


def test_the_tally_reports_every_outcome_including_the_zeroes():
    """A verdict missing from a tally reads as "not applicable" when it means
    "none this time", and the difference matters most for insufficient_data,
    whose absence would hide that the suite ran on enough history."""
    suite = score_suite([m(a.key) for a in R.ANOMALIES])
    assert set(suite.tally()) == set(VERDICTS)
    assert suite.tally()["insufficient_data"] == 0
    assert sum(suite.tally().values()) == 6


# --- rule 4: the embarrassment check ----------------------------------------


def test_an_implausibly_large_result_is_downgraded_to_partial():
    """An eval suite that cannot be embarrassed by its own best result is not
    measuring anything."""
    a = R.get("momentum_12_1")
    huge = a.published_sharpe * (EMBARRASSMENT_MULTIPLE + 0.5)
    s = one("momentum_12_1", realised_sharpe=huge)
    assert s.verdict == "partial"
    assert any("DOWNGRADED" in r for r in s.reasons)


def test_just_under_the_ceiling_still_passes():
    a = R.get("momentum_12_1")
    s = one("momentum_12_1", realised_sharpe=a.published_sharpe * (EMBARRASSMENT_MULTIPLE - 0.01))
    assert s.verdict == "pass"


def test_the_embarrassment_check_only_moves_a_verdict_downward():
    """A huge spread with a failing sign, deflation or p-value stays a fail.

    The check must never rescue anything, or 'too good to be true' becomes a
    route to a pass.
    """
    a = R.get("momentum_12_1")
    huge = a.published_sharpe * 10
    assert one("momentum_12_1", realised_sharpe=huge, deflated_psr=0.1).verdict == "fail"
    assert one("momentum_12_1", realised_sharpe=huge, adj=0.9).verdict == "fail"
    assert one("low_volatility", realised_sharpe=huge).verdict == "fail"


def test_magnitude_far_BELOW_the_published_reference_still_passes():
    """Magnitude is not a gate. Matching a published Sharpe across a different
    sample, universe, horizon and construction is not achievable, and an eval
    that demanded it would fail every real effect too."""
    s = one("low_volatility", realised_sharpe=-0.05)
    assert s.verdict == "pass"
    assert s.sharpe_ratio_to_published < 0.1


# --- reasons, versioning and suite mechanics --------------------------------


def test_every_verdict_carries_reasons_including_a_pass():
    """Same rule as M5's publishable: a count with nothing attached is a number
    nobody can act on, and the demo shows the reasons rather than the tally."""
    for s in score_suite([m(a.key) for a in R.ANOMALIES]):
        assert s.reasons, s.key
        assert all(isinstance(r, str) and r for r in s.reasons)


def test_a_pass_reports_what_was_checked_and_survived():
    s = one("momentum_12_1")
    joined = " ".join(s.reasons)
    assert "sign" in joined and "deflation" in joined and "multiplicity" in joined


def test_every_score_carries_the_scoring_rule_version():
    """Stored beside registry_sha. Neither alone would reveal that two rows in
    one table had been held to different standards."""
    for s in score_suite([m(a.key) for a in R.ANOMALIES]):
        assert s.scoring_rule_version == SCORING_RULE_VERSION


def test_every_score_carries_the_anomaly_caveat():
    for s in score_suite([m(a.key) for a in R.ANOMALIES]):
        assert s.caveat == R.get(s.key).caveat


def test_scoring_a_partial_suite_is_refused():
    """A partial suite under-corrects every p-value in it."""
    with pytest.raises(KeyError, match="low_volatility"):
        score_suite([m(a.key) for a in R.ANOMALIES if a.key != "low_volatility"])


def test_scores_come_back_in_registry_order():
    suite = score_suite([m(a.key) for a in reversed(R.ANOMALIES)])
    assert [s.key for s in suite] == [a.key for a in R.ANOMALIES]


def test_to_dict_is_json_safe_and_complete():
    import json

    s = one("momentum_12_1")
    d = s.to_dict()
    json.dumps(d)
    assert d["verdict"] == "pass"
    assert isinstance(d["reasons"], list)
