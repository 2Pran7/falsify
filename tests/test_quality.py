"""Data-quality gate tests.

The governing constraint: catch a ticker-reuse splice WITHOUT discarding
genuine large moves. Real equities move 40% on earnings and over 100% on
takeover news, and dropping those rows would bias a momentum backtest in a
flattering direction, so both behaviours are pinned below.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from falsify.data.quality import (
    coverage_gaps,
    drop_suspect_tickers,
    flag_suspect_tickers,
    level_shifts,
)

D0 = dt.date(2024, 1, 1)


def _panel(series: dict[str, list[tuple[int, float]]]) -> pl.DataFrame:
    """{'AAA': [(day_offset, close), ...]} -> long panel."""
    rows = [
        (tk, D0 + dt.timedelta(days=off), close)
        for tk, points in series.items()
        for off, close in points
    ]
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}, orient="row"
    ).sort(["ticker", "ts"])


def test_detects_the_bny_ticker_reuse_splice():
    """The real case: a $10 closed-end fund merged away, the ticker reassigned
    to a $139 bank three months later. Both signatures fire at once."""
    panel = _panel(
        {"BNY": [(0, 10.30), (1, 10.25), (2, 10.20), (105, 138.98), (106, 139.40)]}
    )

    shifts = level_shifts(panel)
    assert shifts.height == 1
    assert shifts["ratio"][0] > 13.0

    gaps = coverage_gaps(panel)
    assert gaps.height == 1
    assert gaps["gap_days"][0] == 103

    flagged = flag_suspect_tickers(panel)
    assert flagged["ticker"].to_list() == ["BNY"]
    assert flagged["reason"][0] == "level_shift"


def test_does_not_flag_a_genuine_earnings_crash():
    """Centene fell 40% on withdrawn guidance. Real, and must survive."""
    panel = _panel({"CNC": [(0, 56.65), (1, 33.76), (2, 34.10)]})
    assert flag_suspect_tickers(panel).is_empty()


def test_does_not_flag_a_genuine_takeover_or_trial_move():
    """Moderna moved +177% on Phase 3 melanoma data. A 2.77x ratio is real;
    only ratios past 4x are treated as impossible."""
    panel = _panel({"MRNA": [(0, 62.96), (1, 174.40), (2, 170.00)]})
    assert flag_suspect_tickers(panel).is_empty()


def test_does_not_flag_normal_weekend_and_holiday_gaps():
    """A three-day weekend and a four-day holiday break are ordinary."""
    panel = _panel({"AAA": [(0, 100.0), (3, 101.0), (7, 102.0), (8, 103.0)]})
    assert coverage_gaps(panel).is_empty()
    assert flag_suspect_tickers(panel).is_empty()


def test_flags_a_dormant_ticker_even_without_a_price_jump():
    """A splice between two securities at similar prices leaves no level shift,
    so the coverage gap has to carry the detection on its own."""
    panel = _panel({"XYZ": [(0, 50.0), (1, 50.5), (200, 51.0), (201, 51.2)]})

    assert level_shifts(panel).is_empty()
    flagged = flag_suspect_tickers(panel)
    assert flagged["ticker"].to_list() == ["XYZ"]
    assert flagged["reason"][0] == "coverage_gap"


def test_each_ticker_is_reported_once():
    panel = _panel({"BAD": [(0, 10.0), (1, 200.0), (150, 5.0), (151, 400.0)]})
    assert flag_suspect_tickers(panel).height == 1


def test_drop_removes_only_the_flagged_names():
    panel = _panel(
        {
            "GOOD": [(0, 100.0), (1, 101.0), (2, 99.0)],
            "BNY": [(0, 10.20), (105, 139.00)],
        }
    )
    clean, flagged = drop_suspect_tickers(panel)

    assert set(clean["ticker"].unique()) == {"GOOD"}
    assert flagged["ticker"].to_list() == ["BNY"]
    assert len(clean) == 3


def test_clean_panel_reports_nothing_and_keeps_its_schema():
    panel = _panel({"AAA": [(0, 100.0), (1, 101.0)], "BBB": [(0, 50.0), (1, 49.0)]})
    clean, flagged = drop_suspect_tickers(panel)

    assert flagged.is_empty()
    assert flagged.schema["reason"] == pl.Utf8
    assert clean.equals(panel)
