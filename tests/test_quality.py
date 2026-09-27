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
    trim_ragged_end,
    truncate_suspect_tickers,
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


# --- Module 7: the point-in-time gate ---------------------------------------


def test_truncation_keeps_history_before_the_first_defect():
    """BNY's fund history is genuine and stays; the bank after the hole goes."""
    panel = _panel(
        {
            "GOOD": [(0, 100.0), (1, 101.0), (2, 99.0)],
            "BNY": [(0, 10.30), (1, 10.25), (2, 10.20), (105, 138.98), (106, 139.40)],
        }
    )
    clean, report = truncate_suspect_tickers(panel)

    bny = clean.filter(pl.col("ticker") == "BNY")
    assert bny["close"].to_list() == [10.30, 10.25, 10.20]
    assert clean.filter(pl.col("ticker") == "GOOD").height == 3
    assert report["ticker"].to_list() == ["BNY"]
    assert report["last_kept"][0] == D0 + dt.timedelta(days=2)
    assert report["rows_dropped"][0] == 2


def test_truncation_never_forms_the_return_across_the_join():
    """The last kept close is the OLD security's; the first dropped is the new
    one's. The 13x return between them must not survive in any form."""
    panel = _panel({"X": [(0, 10.0), (1, 10.0), (2, 50.0), (3, 51.0)]})
    clean, _ = truncate_suspect_tickers(panel)
    closes = clean["close"].to_list()
    assert closes == [10.0, 10.0]
    assert max(b / a for a, b in zip(closes, closes[1:])) < 4.0


def test_truncation_cuts_at_the_first_of_several_defects():
    panel = _panel(
        {"Z": [(0, 10.0), (1, 10.0), (30, 10.5), (31, 10.6), (32, 60.0), (33, 60.0)]}
    )
    clean, report = truncate_suspect_tickers(panel)
    assert clean.height == 2
    assert report["reason"][0] == "coverage_gap"
    assert report["rows_dropped"][0] == 4


def test_truncation_leaves_genuine_large_moves_alone():
    """A 3x takeover pop is real and must not cost the ticker any history."""
    panel = _panel({"TGT": [(0, 20.0), (1, 60.0), (2, 61.0)]})
    clean, report = truncate_suspect_tickers(panel)
    assert clean.equals(panel)
    assert report.is_empty()


def test_truncation_on_a_clean_panel_is_the_identity_with_a_typed_report():
    panel = _panel({"AAA": [(0, 100.0), (1, 101.0)], "BBB": [(0, 50.0), (1, 49.0)]})
    clean, report = truncate_suspect_tickers(panel)
    assert clean.equals(panel)
    assert report.is_empty()
    assert report.schema["last_kept"] == pl.Date
    assert report.schema["rows_dropped"] == pl.UInt32


def _cut(df: pl.DataFrame, t: dt.date) -> pl.DataFrame:
    return df.filter(pl.col("ts") <= t).sort(["ticker", "ts"])


def test_truncation_is_point_in_time():
    """THE property. For every date T, gating the panel as it stood at T gives
    exactly what the full-panel gate keeps up to T. If it did not, something
    after T changed the universe at T, which is lookahead by definition."""
    panel = _panel(
        {
            "GOOD": [(d, 100.0 + d) for d in range(0, 60)],
            "BNY": [(d, 10.0) for d in range(0, 10)] + [(d, 139.0) for d in range(40, 60)],
            "JMP": [(d, 20.0) for d in range(0, 25)] + [(d, 95.0) for d in range(25, 60)],
        }
    )
    full, _ = truncate_suspect_tickers(panel)
    for t in panel["ts"].unique().sort().to_list():
        at_t, _ = truncate_suspect_tickers(panel.filter(pl.col("ts") <= t))
        assert _cut(at_t, t).equals(_cut(full, t)), f"lookahead at {t}"


def test_the_whole_sample_gate_fails_the_same_property():
    """The control. If this ever passes, the property test above is not
    discriminating and has not been shown to work."""
    panel = _panel(
        {
            "GOOD": [(d, 100.0) for d in range(0, 30)],
            "JMP": [(d, 20.0) for d in range(0, 15)] + [(d, 95.0) for d in range(15, 30)],
        }
    )
    full, _ = drop_suspect_tickers(panel)
    t = D0 + dt.timedelta(days=10)
    at_t, _ = drop_suspect_tickers(panel.filter(pl.col("ts") <= t))
    assert not _cut(at_t, t).equals(_cut(full, t))


# --- the ragged end (found 27 Sep: one re-ingested ticker ran two weeks past the rest)


def _wide(n_tickers: int, days: range) -> dict[str, list[tuple[int, float]]]:
    return {f"T{i}": [(d, 100.0 + d) for d in days] for i in range(n_tickers)}


def test_a_thin_tail_is_trimmed_and_reported():
    series = _wide(10, range(0, 20))
    series["BNY"] = [(d, 50.0) for d in range(0, 34)]  # runs 14 days past everyone
    clean, dropped = trim_ragged_end(_panel(series))
    assert clean["ts"].max() == D0 + dt.timedelta(days=19)
    assert dropped.height == 14
    assert dropped["n_tickers"].max() == 1


def test_a_thin_date_in_the_middle_is_left_alone():
    """Interior holes are a coverage defect for the coverage checks to report,
    not something to delete silently."""
    series = _wide(10, range(0, 10))
    for i in range(10):
        series[f"T{i}"] += [(d, 1.0) for d in range(12, 20)]
    series["T0"].append((10, 1.0))  # day 10: one ticker only
    panel = _panel(series)
    clean, dropped = trim_ragged_end(panel)
    assert clean.equals(panel)
    assert dropped.is_empty()


def test_a_full_panel_is_the_identity_with_a_typed_report():
    panel = _panel(_wide(5, range(0, 10)))
    clean, dropped = trim_ragged_end(panel)
    assert clean.equals(panel)
    assert dropped.schema["n_tickers"] == pl.UInt32


def test_the_kept_panel_is_a_prefix_of_the_dates():
    series = _wide(10, range(0, 20))
    series["X"] = [(d, 1.0) for d in range(0, 30)]
    panel = _panel(series)
    clean, _ = trim_ragged_end(panel)
    kept = clean["ts"].unique().sort().to_list()
    assert kept == panel["ts"].unique().sort().to_list()[: len(kept)]
