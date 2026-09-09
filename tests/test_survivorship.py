"""Survivorship audit.

No database: membership frames are built by hand, so a failure here is logic
and never connectivity. The audit's own arithmetic is checked against
metrics.py rather than against remembered constants.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from falsify.backtest import metrics
from falsify.stats.survivorship import (
    coverage_report,
    restrict_to_members,
    survivorship_gap,
)

D = dt.date


def _signal(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"ts": pl.Date, "ticker": pl.Utf8, "sig": pl.Float64}, orient="row"
    )


def _membership(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"ts": pl.Date, "ticker": pl.Utf8}, orient="row")


# ==========================================================================
# restrict_to_members
# ==========================================================================

def test_keeps_only_point_in_time_members():
    sig = _signal([
        (D(2025, 1, 2), "AAPL", 0.5),
        (D(2025, 1, 2), "XOM", 0.3),
        (D(2025, 1, 2), "GONE", 0.9),
    ])
    mem = _membership([(D(2025, 1, 2), "AAPL"), (D(2025, 1, 2), "XOM")])
    out = restrict_to_members(sig, mem)
    assert set(out["ticker"]) == {"AAPL", "XOM"}


def test_membership_is_evaluated_per_date():
    """The same ticker can be a member on one date and not another. Filtering
    on the ticker alone would keep XOM after it left the index."""
    sig = _signal([
        (D(2025, 1, 2), "XOM", 0.3),
        (D(2025, 6, 2), "XOM", 0.4),
    ])
    mem = _membership([(D(2025, 1, 2), "XOM")])
    out = restrict_to_members(sig, mem)
    assert out.height == 1
    assert out["ts"].to_list() == [D(2025, 1, 2)]


def test_non_member_disappears_rather_than_becoming_null():
    """An inner join, not a left join. A null signal would still be a row, and
    downstream null handling would have to guess what it meant."""
    sig = _signal([(D(2025, 1, 2), "GONE", 0.9)])
    mem = _membership([(D(2025, 1, 2), "AAPL")])
    out = restrict_to_members(sig, mem)
    assert out.is_empty()
    assert out.schema["sig"] == pl.Float64


def test_preserves_schema_and_sort():
    sig = _signal([
        (D(2025, 1, 3), "MSFT", 0.1),
        (D(2025, 1, 2), "AAPL", 0.5),
    ])
    mem = _membership([(D(2025, 1, 2), "AAPL"), (D(2025, 1, 3), "MSFT")])
    out = restrict_to_members(sig, mem)
    assert out.columns == ["ts", "ticker", "sig"]
    assert out["ts"].to_list() == sorted(out["ts"].to_list())


def test_does_not_duplicate_rows():
    """A membership frame with a repeated (ts, ticker) must not multiply the
    signal row, which would double that name's weight in the decile sort."""
    sig = _signal([(D(2025, 1, 2), "AAPL", 0.5)])
    mem = _membership([(D(2025, 1, 2), "AAPL"), (D(2025, 1, 2), "AAPL")])
    assert restrict_to_members(sig, mem).height == 1


def test_empty_membership_yields_empty_signal():
    """Dates before the first snapshot have no members. Returning the signal
    unfiltered would silently run the biased universe instead."""
    sig = _signal([(D(2020, 1, 2), "AAPL", 0.5)])
    out = restrict_to_members(sig, _membership([]))
    assert out.is_empty()


def test_empty_signal_is_handled():
    out = restrict_to_members(_signal([]), _membership([(D(2025, 1, 2), "AAPL")]))
    assert out.is_empty()


# ==========================================================================
# survivorship_gap
# ==========================================================================

@pytest.fixture
def biased_and_honest() -> tuple[pl.Series, pl.Series]:
    """The biased run beats the honest one on every day, which is the expected
    direction: the names missing from today's constituent list are the losers."""
    current = pl.Series([0.004, 0.002, 0.006, -0.001, 0.003] * 20)
    pit = pl.Series([0.002, 0.001, 0.004, -0.003, 0.001] * 20)
    return current, pit


def test_gap_reports_every_metric(biased_and_honest):
    current, pit = biased_and_honest
    gap = survivorship_gap(current, pit)
    for name in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"):
        for suffix in ("current", "pit", "gap"):
            assert f"{name}_{suffix}" in gap
    assert gap["n_days_current"] == 100
    assert gap["n_days_pit"] == 100


def test_gap_values_agree_with_metrics_module(biased_and_honest):
    """The audit must not re-implement the metrics. If these disagree there are
    two definitions of Sharpe in the project and the note quotes whichever is
    convenient."""
    current, pit = biased_and_honest
    gap = survivorship_gap(current, pit)
    assert gap["sharpe_current"] == pytest.approx(metrics.sharpe(current))
    assert gap["sharpe_pit"] == pytest.approx(metrics.sharpe(pit))
    assert gap["total_return_current"] == pytest.approx(metrics.total_return(current))


def test_gap_is_current_minus_pit(biased_and_honest):
    current, pit = biased_and_honest
    gap = survivorship_gap(current, pit)
    for name in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"):
        assert gap[f"{name}_gap"] == pytest.approx(
            gap[f"{name}_current"] - gap[f"{name}_pit"]
        )


def test_biased_run_looks_better(biased_and_honest):
    """The expected sign. A backtest that cannot see the companies that failed
    should report a higher Sharpe than one that can."""
    current, pit = biased_and_honest
    gap = survivorship_gap(current, pit)
    assert gap["sharpe_gap"] > 0
    assert gap["total_return_gap"] > 0


def test_identical_series_show_no_gap():
    r = pl.Series([0.01, -0.005, 0.002, 0.004] * 25)
    gap = survivorship_gap(r, r)
    for name in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"):
        assert gap[f"{name}_gap"] == pytest.approx(0.0, abs=1e-12)


def test_negative_gap_is_reported_not_suppressed():
    """If the dropped names were ones the strategy was correctly short, the gap
    goes negative. That is a finding about this sample, not a bug, and the
    audit must return it rather than clamping at zero.
    """
    current = pl.Series([0.001, 0.000, 0.002, -0.001] * 25)
    pit = pl.Series([0.005, 0.004, 0.006, 0.003] * 25)
    gap = survivorship_gap(current, pit)
    assert gap["sharpe_gap"] < 0


def test_mismatched_lengths_are_visible_in_the_day_counts():
    """A dropped ticker with no ingested prices shortens the point-in-time run.
    The counts must surface that rather than hiding it inside a ratio."""
    current = pl.Series([0.001] * 100)
    pit = pl.Series([0.001] * 60)
    gap = survivorship_gap(current, pit)
    assert gap["n_days_current"] == 100
    assert gap["n_days_pit"] == 60


# ==========================================================================
# coverage_report
# ==========================================================================

def test_full_coverage_reports_nothing_missing():
    mem = _membership([(D(2025, 1, 2), "AAPL"), (D(2025, 1, 3), "AAPL")])
    panel = _membership([(D(2025, 1, 2), "AAPL"), (D(2025, 1, 3), "AAPL")])
    row = coverage_report(mem, panel).row(0, named=True)
    assert row["ticker"] == "AAPL"
    assert row["n_member_days"] == 2
    assert row["n_price_days"] == 2
    assert row["missing_days"] == 0


def test_ticker_with_no_prices_is_fully_missing():
    """The failure mode this function exists to catch. A dropped ticker never
    ingested contributes nothing to the point-in-time run, so the audit
    under-measures the bias and reports a comfortingly small gap."""
    mem = _membership([(D(2025, 1, 2), "GONE"), (D(2025, 1, 3), "GONE")])
    panel = _membership([(D(2025, 1, 2), "AAPL")])
    row = coverage_report(mem, panel).filter(pl.col("ticker") == "GONE").row(0, named=True)
    assert row["n_member_days"] == 2
    assert row["n_price_days"] == 0
    assert row["missing_days"] == 2


def test_partial_coverage_is_counted():
    mem = _membership([(D(2025, 1, i), "PART") for i in (2, 3, 6, 7)])
    panel = _membership([(D(2025, 1, i), "PART") for i in (2, 3)])
    row = coverage_report(mem, panel).row(0, named=True)
    assert row["n_member_days"] == 4
    assert row["n_price_days"] == 2
    assert row["missing_days"] == 2


def test_sorted_worst_first():
    """The point of the table is that the biggest hole is the first thing seen."""
    mem = _membership(
        [(D(2025, 1, 2), "OK")]
        + [(D(2025, 1, i), "BAD") for i in (2, 3, 6, 7, 8)]
        + [(D(2025, 1, i), "MID") for i in (2, 3)]
    )
    panel = _membership([(D(2025, 1, 2), "OK"), (D(2025, 1, 2), "MID")])
    out = coverage_report(mem, panel)
    assert out["ticker"].to_list() == ["BAD", "MID", "OK"]
    assert out["missing_days"].to_list() == [5, 1, 0]


def test_prices_outside_membership_do_not_count():
    """SPY has prices and is never an index member. Counting its rows would
    make coverage look better than it is."""
    mem = _membership([(D(2025, 1, 2), "AAPL")])
    panel = _membership([(D(2025, 1, 2), "AAPL"), (D(2025, 1, 2), "SPY")])
    out = coverage_report(mem, panel)
    assert out["ticker"].to_list() == ["AAPL"]


def test_empty_membership_gives_typed_empty_frame():
    out = coverage_report(_membership([]), _membership([(D(2025, 1, 2), "AAPL")]))
    assert out.is_empty()
    assert out.columns == ["ticker", "n_member_days", "n_price_days", "missing_days"]
