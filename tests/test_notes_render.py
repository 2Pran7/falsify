"""Markdown rendering.

Most of these are not formatting tests. They pin the two properties the whole
note schema exists to preserve:

  1. THE TABLE COMES FROM THE DATA. Nothing in render.py reads a number out of
     the prose. The test for this renders a note whose prose contains a figure
     that appears nowhere in the pipeline output, and asserts the tables are
     unaffected.
  2. PROBABILITIES AND SHARPE RATIOS DO NOT SHARE A COLUMN. Run 1 of this agent
     tabulated 0.65 raw beside 0.73 "deflated" — impossible for a ratio,
     ordinary for a probability — and the column heading is what caused it.
     The renderer must not be able to reproduce that layout.

A failed note must also render. A page that can only display successes is the
highlight reel this project is pitched against.
"""
from __future__ import annotations

import re

from falsify.notes.render import to_markdown, to_summary_line
from tests.test_notes_schema import a_backtest, a_note, bad_provenance, good_run


# --------------------------------------------------------------------------
# the table is the pipeline's, the prose is the model's
# --------------------------------------------------------------------------

def test_a_figure_invented_in_the_prose_never_reaches_a_table():
    note = a_note(prose="The Sharpe was 9.99 and the strategy returned 412%.")
    md = to_markdown(note)
    table = md.split("## The agent's commentary")[0]
    assert "9.99" not in table
    assert "412%" not in table
    assert "0.45" in table  # the pipeline's Sharpe, from metrics


def test_the_commentary_is_labelled_as_the_model_s_reading():
    md = to_markdown(a_note())
    assert "## The agent's commentary" in md
    assert "Written by the model" in md
    assert "the reasoning is not checked" in md


def test_provenance_is_described_as_a_floor_not_a_proof():
    """Run 1 passed provenance completely and reached a wrong conclusion. A
    page that shows a green tick without that sentence is overclaiming."""
    md = to_markdown(a_note())
    assert "numbers, not claims" in md
    assert "floor, not a proof" in md


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

def test_probabilities_and_sharpe_ratios_are_in_separate_tables():
    md = to_markdown(a_note())
    metrics_block = md.split("### Statistics")[0]
    stats_block = md.split("### Statistics")[1]
    assert "Sharpe" in metrics_block
    assert "Sharpe" not in stats_block.split("Both `P(` columns")[0]
    assert "P(beats best of N)" in stats_block


def test_the_statistics_table_says_the_p_columns_are_not_sharpe_ratios():
    md = to_markdown(a_note())
    assert "not Sharpe ratios" in md
    assert "below 0.50 the evidence does not survive" in md.replace("\n", " ")


def test_the_trial_variance_assumption_is_printed_under_the_table_it_governs():
    """Not appended at the end of the document. Every deflated figure moves
    with it, and an assumption that is not next to the number it governs is
    filed rather than disclosed."""
    md = to_markdown(a_note())
    stats_block = md.split("### Statistics")[1].split("## The agent's commentary")[0]
    assert "0.0009" in stats_block
    assert "assumed, not measured" in stats_block


def test_the_trial_count_the_rows_were_deflated_at_is_stated():
    md = to_markdown(a_note(assumptions={"trial_variance": 0.0009, "n_trials_for_deflation": 3}))
    assert "N = 3" in md


def test_mintrl_is_given_in_years_as_well_as_days_because_days_hide_the_scale():
    md = to_markdown(a_note())
    assert "1655" in md          # 1654.6 days
    assert "6.6" in md           # the same number, in years


# --------------------------------------------------------------------------
# failures render
# --------------------------------------------------------------------------

def test_a_failed_note_leads_with_why_before_any_number():
    note = a_note(provenance=bad_provenance())
    md = to_markdown(note)
    assert "NOT PUBLISHABLE" in md
    assert md.index("NOT PUBLISHABLE") < md.index("## What the pipeline computed")
    assert "provenance failed" in md


def test_a_failed_note_lists_the_unverified_literals():
    md = to_markdown(a_note(provenance=bad_provenance()))
    assert "10.16" in md


def test_a_cut_short_run_is_marked_at_the_stop_reason_itself():
    md = to_markdown(a_note(run=good_run(stop_reason="max_turns")))
    assert "a guard fired" in md
    assert "max_turns" in md


def test_a_note_with_no_backtests_renders_without_blowing_up():
    md = to_markdown(a_note(backtests=(), run=good_run(tool_sequence=("fetch_data",))))
    assert "No backtest was run" in md
    assert "NOT PUBLISHABLE" in md


def test_a_backtest_whose_analysis_failed_shows_the_error_not_a_blank_row():
    note = a_note(
        backtests=(a_backtest(analysed=False, analysis_error="series too short"),)
    )
    md = to_markdown(note)
    assert "series too short" in md
    assert "could not be computed" in md


# --------------------------------------------------------------------------
# the rest of the record reaches the page
# --------------------------------------------------------------------------

def test_the_hypothesis_is_the_title_verbatim():
    asked = "Do stocks that went up over the last year keep going up?"
    assert to_markdown(a_note(hypothesis=asked)).startswith(f"# {asked}")


def test_the_run_section_reports_what_it_cost():
    md = to_markdown(a_note())
    assert "$0.0360" in md
    assert "claude-sonnet-5" in md


def test_the_tool_sequence_is_shown_so_a_missing_analyze_results_is_visible():
    md = to_markdown(a_note(run=good_run(tool_sequence=("fetch_data", "run_backtest"))))
    assert "fetch_data → run_backtest" in md
    assert "analyze_results was never called" in md


def test_long_only_is_named_in_the_variant_because_its_sharpe_carries_beta():
    md = to_markdown(a_note(backtests=(a_backtest(long_short=False),)))
    assert "long-only" in md


def test_every_markdown_table_row_has_the_same_column_count_as_its_header():
    md = to_markdown(a_note())
    for block in re.findall(r"(\|[^\n]*\|\n\|[-: |]+\|\n(?:\|[^\n]*\|\n)+)", md):
        lines = [ln for ln in block.strip().splitlines() if ln.startswith("|")]
        widths = {ln.count("|") for ln in lines}
        assert len(widths) == 1, f"ragged table:\n{block}"


def test_the_summary_line_leads_with_the_verdict():
    assert to_summary_line(a_note()).startswith("PUBLISHABLE")
    assert to_summary_line(a_note(provenance=bad_provenance())).startswith("NOT PUBLISHABLE")


def test_the_summary_line_quotes_the_deflated_probability_not_the_raw_sharpe():
    line = to_summary_line(a_note())
    assert "0.422" in line   # prob_beats_best_of_n_trials
    assert "0.45" not in line
