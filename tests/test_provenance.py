"""Tests for numeric provenance.

The one that matters is `test_a_fabricated_sharpe_is_caught`. Everything else
exists to stop that test being satisfiable by a checker that always fails, or
always passes.

Two failure modes are equally bad and pull in opposite directions:

    too strict  -> real notes fail on "19.7%" when the tool said 0.196938, the
                   check gets switched off, and it protects nothing.
    too loose   -> a fabricated number finds some coincidental match, the check
                   passes everything, and it protects nothing.

So there are tests in both directions, and the transform list is closed: the
final tests assert that arithmetic the model might do is NOT silently accepted.
"""
from __future__ import annotations

import json

import pytest

from falsify.agent.provenance import (
    TRIVIAL_INTEGER_MAX,
    check_note,
    check_run,
    extract_numbers,
)

# One realistic tool result, as the model would have received it.
ANALYSIS = json.dumps(
    {
        "sharpe_per_period": 0.041,
        "sharpe_annualised": 0.65,
        "prob_sharpe_above_zero": 0.733,
        "prob_beats_best_of_n_trials": 0.592,
        "min_track_record_length_days": 1655.0,
        "n_trials_used": 3,
    }
)
BACKTEST = json.dumps(
    {
        "handle": "backtest_1",
        "total_return": 0.201812,
        "cagr": 0.196938,
        "ann_vol": 0.415901,
        "sharpe": 0.65,
        "max_drawdown": -0.318742,
        "n_invested_days": 238,
    }
)
RESULTS = [ANALYSIS, BACKTEST]


# --- extraction ------------------------------------------------------------


def test_extract_plain_numbers():
    assert [v for _, v in extract_numbers("sharpe 0.65 over 238 days")] == [0.65, 238.0]


def test_extract_handles_thousands_separators():
    assert [v for _, v in extract_numbers("1,655 days")] == [1655.0]


def test_extract_handles_negatives():
    assert [v for _, v in extract_numbers("drawdown -31.9%")] == [-31.9]


def test_extract_survives_prose_that_is_not_a_number():
    """`mom_12_1` and `12-1 momentum` are names. They must not crash the parse."""
    extract_numbers("the 12-1 momentum feature mom_12_1 ran fine")


# --- the permitted transforms ----------------------------------------------


def test_a_number_quoted_verbatim_passes():
    assert check_note("The Sharpe was 0.65.", RESULTS).ok


def test_a_fraction_quoted_as_a_percentage_passes():
    """0.196938 -> 19.7%. The single most common presentation conversion."""
    assert check_note("CAGR was 19.7%.", RESULTS).ok


def test_a_drawdown_quoted_as_a_magnitude_passes():
    """Tools report -0.318742; notes routinely write 31.9% or -31.9%."""
    assert check_note("Max drawdown 31.9%, or -31.9%.", RESULTS).ok


def test_days_quoted_as_years_passes():
    """1655 trading days is 6.6 years. Dividing by 252 is a unit, not a sum."""
    assert check_note("It would need 6.6 years of data.", RESULTS).ok


def test_a_rounded_figure_passes():
    assert check_note("A track record of 1,655 days.", RESULTS).ok


def test_a_truncated_figure_passes():
    """Models truncate as well as round: 19.6938% written as "19.6%".

    The rounding ladder alone gives 19.7, so only the tolerance covers this.
    Without a test for it the tolerance is untested code, and untested code in
    a checker is where the checker silently stops working.
    """
    assert check_note("CAGR was 19.6%.", RESULTS).ok


def test_a_realistic_note_passes_in_full():
    """The shape of a real answer, to prove the check is usable in practice.

    If this fails, the check is too strict, someone turns it off, and it stops
    protecting anything. That outcome is as bad as being too loose.
    """
    note = (
        "The decile long/short book returned 20.2% total, 19.7% CAGR, with "
        "41.6% annualised volatility and a -31.9% maximum drawdown, for a "
        "Sharpe of 0.65 over 238 invested days. The probability the true "
        "Sharpe exceeds zero is 0.733, and after deflating for 3 trials it is "
        "0.592. A track record of 1,655 days, about 6.6 years, would be "
        "needed before that Sharpe could be distinguished from zero."
    )
    report = check_note(note, RESULTS)
    assert report.ok, report.summary()


# --- the point of the module -----------------------------------------------


def test_a_fabricated_sharpe_is_caught():
    """THE TEST. A number the pipeline never produced must be reported.

    This is what turns "the LLM decides, never computes" from a claim into
    something a reader can check. The model was shown a Sharpe of 0.65; the
    note claims 1.87; nothing in the run produced 1.87.
    """
    report = check_note("The strategy achieved a Sharpe of 1.87.", RESULTS)
    assert not report.ok
    assert report.unverified[0].literal == "1.87"


def test_the_failure_names_the_number_and_shows_where_it_was():
    """A failure has to be actionable: which numeral, and in which sentence."""
    report = check_note("Momentum delivered a Sharpe of 1.87 net of costs.", RESULTS)
    assert "1.87" in report.summary()
    assert "Sharpe" in report.unverified[0].context


def test_several_fabrications_are_all_reported():
    report = check_note("Sharpe 1.87, CAGR 44.3%, drawdown -9.1%.", RESULTS)
    assert len(report.unverified) == 3


def test_one_bad_number_fails_an_otherwise_clean_note():
    """No partial credit: a note is trustworthy or it is not."""
    report = check_note("Sharpe was 0.65 and the Sortino was 2.34.", RESULTS)
    assert not report.ok
    assert report.verified == 1


def test_coverage_is_reported_even_on_a_pass():
    report = check_note("Sharpe 0.65, CAGR 19.7%.", RESULTS)
    assert report.checked == 2
    assert report.coverage == 1.0


# --- the transform list is CLOSED ------------------------------------------


def test_a_difference_between_two_given_numbers_is_not_verified():
    """0.733 - 0.592 = 0.141. Subtraction is arithmetic, and arithmetic is
    precisely what the model is not supposed to be doing. A number nobody
    computed in the pipeline cannot be checked by anyone reading the note."""
    assert not check_note("The gap was 0.141.", RESULTS).ok


def test_a_ratio_of_two_given_numbers_is_not_verified():
    """238 / 1655 = 0.1438."""
    assert not check_note("We have 0.1438 of the required history.", RESULTS).ok


def test_a_sum_of_two_given_numbers_is_not_verified():
    """0.65 + 0.592 = 1.242, a meaningless figure that must not pass."""
    assert not check_note("Combined score 1.242.", RESULTS).ok


def test_a_plausible_but_absent_sharpe_is_not_verified():
    """0.71 is the Sharpe from an earlier run in this project's history.

    Exactly the kind of number a model might carry across from context rather
    than from a tool result, and exactly what this check exists to catch.
    """
    assert not check_note("A Sharpe of 0.71 was observed.", RESULTS).ok


# --- trivial numerals ------------------------------------------------------


def test_small_integers_are_exempt_not_verified():
    """"all 3 variants" and "the top 10" are prose, not numeric claims."""
    report = check_note("We tried 3 variants across 2 constructions.", RESULTS)
    assert report.ok
    assert report.exempt == 2
    assert report.checked == 0


def test_the_exemption_cannot_cover_a_sharpe_or_a_probability():
    """The exemption is integers only, and small ones.

    A Sharpe, a probability and a return all carry decimals or exceed the
    threshold, so none of them can slip through as "trivial". If they could,
    the whole check would be decorative.
    """
    report = check_note("Sharpe 25.0 and 0.99 probability.", RESULTS)
    assert not report.ok
    assert len(report.unverified) == 2
    assert TRIVIAL_INTEGER_MAX < 25, "the exemption must stay well below any real figure"


def test_an_empty_note_passes_vacuously():
    report = check_note("No numbers here at all.", RESULTS)
    assert report.ok
    assert report.checked == 0


def test_a_note_with_no_tool_results_fails_on_any_real_number():
    """A model that answered without calling a tool has nothing to trace to."""
    assert not check_note("Sharpe was 0.65.", []).ok


# --- the RunResult wrapper -------------------------------------------------


class _FakeResult:
    def __init__(self, answer: str, blobs: list[str]):
        self.answer = answer
        self.messages = [
            {"role": "user", "content": "q"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{i}", "content": b}
                    for i, b in enumerate(blobs)
                ],
            },
        ]


def test_check_run_reads_tool_results_out_of_the_transcript():
    assert check_run(_FakeResult("Sharpe 0.65.", RESULTS)).ok


def test_check_run_catches_a_fabrication_in_a_real_transcript():
    assert not check_run(_FakeResult("Sharpe 1.87.", RESULTS)).ok


def test_a_figure_from_a_tool_description_is_verified():
    """The model was shown the schemas, so their numbers are legitimate sources.

    `compute_feature` states that mom_12_1 needs 252 days of history. A note
    mentioning the 252-day lookback is quoting the system, not inventing a
    figure. Checking a real run flagged exactly this, and a checker that cries
    wolf on a correct note is one that gets switched off.
    """
    assert check_run(_FakeResult("mom_12_1 needs 252 days of history.", [])).ok


def test_the_schemas_do_not_verify_an_arbitrary_number():
    """Adding a source must not turn the check into a sieve."""
    assert not check_run(_FakeResult("The Sharpe was 1.87.", [])).ok


def test_check_run_ignores_plain_string_messages():
    """The first message is the hypothesis, a bare string, not a block list."""
    check_run(_FakeResult("No numbers.", []))


def test_error_results_count_as_shown_to_the_model():
    """A run that recovered from an error must not fail on the error's own text.

    Error messages quote handles and counts, the model may legitimately repeat
    them, and excluding them would raise a false alarm on a healthy run.
    """
    blobs = ["n_buckets must be between 2 and 20, got 47.5"]
    assert check_note("I first tried 47.5 buckets, which was rejected.", blobs).ok
