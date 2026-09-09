"""Portfolio construction tests.

Null handling is the substantive case here; the remainder is arithmetic.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from falsify.backtest.portfolio import (
    decile_weights,
    fixed_weights,
    hold_until_next_rebalance,
    month_end_dates,
)

D0 = dt.date(2024, 1, 1)


def _dates(n: int) -> list[dt.date]:
    return [D0 + dt.timedelta(days=i) for i in range(n)]


def _signal(rows) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"ts": pl.Date, "ticker": pl.Utf8, "sig": pl.Float64}, orient="row"
    )


def test_decile_takes_exactly_ten_names_from_a_hundred():
    d = _dates(1)[0]
    sig = _signal([(d, f"T{i:03d}", float(i)) for i in range(100)])
    w = decile_weights(sig, n_buckets=10, long_short=True)

    longs = w.filter(pl.col("w") > 0)
    shorts = w.filter(pl.col("w") < 0)
    assert len(longs) == 10
    assert len(shorts) == 10
    # highest signals are T090..T099
    assert set(longs["ticker"]) == {f"T{i:03d}" for i in range(90, 100)}
    assert set(shorts["ticker"]) == {f"T{i:03d}" for i in range(0, 10)}


def test_long_short_is_dollar_neutral_with_gross_two():
    d = _dates(1)[0]
    sig = _signal([(d, f"T{i:03d}", float(i)) for i in range(100)])
    w = decile_weights(sig)

    assert w["w"].sum() == pytest.approx(0.0, abs=1e-12)
    assert w["w"].abs().sum() == pytest.approx(2.0, rel=1e-12)


def test_long_only_sums_to_one():
    d = _dates(1)[0]
    sig = _signal([(d, f"T{i:03d}", float(i)) for i in range(100)])
    w = decile_weights(sig, long_short=False)

    assert len(w) == 10
    assert w["w"].sum() == pytest.approx(1.0, rel=1e-12)
    assert (w["w"] > 0).all()


def test_nulls_are_dropped_not_ranked_as_zero():
    """Null signals must be excluded, not ranked as zero.

    mom_12_1 is null for a ticker's first 252 days. A null ranked as 0.0 sorts
    below every positive signal and lands in the short bucket, systematically
    shorting names merely too young to score: a real portfolio producing
    real-looking numbers and meaning nothing. Every scored name here has a
    positive signal, so nulls-as-zero would form the bottom and be shorted.
    """
    d = _dates(1)[0]
    rows = [(d, f"T{i:02d}", float(i + 1)) for i in range(20)]     # signals 1..20
    rows += [(d, f"N{i:02d}", None) for i in range(20)]            # 20 unscored
    w = decile_weights(_signal(rows), n_buckets=10)

    assert not any(t.startswith("N") for t in w["ticker"]), "null-signal ticker was traded"
    # bucket size comes from the 20 that scored, not the 40 rows present
    assert len(w.filter(pl.col("w") > 0)) == 2
    assert len(w.filter(pl.col("w") < 0)) == 2


def test_thin_cross_section_degrades_instead_of_crashing():
    """Five names with deciles: floor(5/10) = 0, so bucket size floors at 1."""
    d = _dates(1)[0]
    sig = _signal([(d, f"T{i}", float(i)) for i in range(5)])
    w = decile_weights(sig, n_buckets=10)

    assert len(w.filter(pl.col("w") > 0)) == 1
    assert len(w.filter(pl.col("w") < 0)) == 1
    assert w["w"].sum() == pytest.approx(0.0)


def test_single_name_produces_no_long_short_position():
    """One name cannot be both the top and the bottom of its own cross-section."""
    d = _dates(1)[0]
    assert decile_weights(_signal([(d, "AAA", 1.0)])).is_empty()


def test_all_null_signal_returns_empty_not_error():
    d = _dates(1)[0]
    out = decile_weights(_signal([(d, "AAA", None), (d, "BBB", None)]))
    assert out.is_empty()
    assert out.schema["w"] == pl.Float64


def test_fixed_weights_covers_every_date():
    ds = pl.Series("ts", _dates(5))
    w = fixed_weights(ds, "SPY")
    assert len(w) == 5
    assert set(w["ticker"]) == {"SPY"}
    assert (w["w"] == 1.0).all()


def test_month_end_picks_the_last_trading_day_not_the_calendar_end():
    """31 March 2024 was a Sunday, and Good Friday closed the 29th, so the last
    trading day was Thursday 28 March. Rebalancing on a closed date silently
    loses a day of returns."""
    ds = pl.Series("ts", [dt.date(2024, 3, d) for d in (26, 27, 28)]
                        + [dt.date(2024, 4, d) for d in (1, 2)])
    out = month_end_dates(ds).to_list()
    assert out == [dt.date(2024, 3, 28), dt.date(2024, 4, 2)]


def test_each_rebalance_is_a_complete_portfolio_snapshot():
    """A name absent from a rebalance is exited, not held indefinitely.

    AAA is selected on day 0, BBB on day 3. Each rebalance being a full
    snapshot, day 3 also implies AAA goes to zero; per-ticker updates would
    leave AAA open alongside BBB.
    """
    ds = _dates(6)
    rebal = pl.DataFrame(
        [(ds[0], "AAA", 1.0), (ds[3], "BBB", 1.0)],
        schema={"ts": pl.Date, "ticker": pl.Utf8, "w": pl.Float64},
        orient="row",
    )
    held = hold_until_next_rebalance(rebal, pl.Series("ts", ds))

    aaa = held.filter(pl.col("ticker") == "AAA")["ts"].to_list()
    bbb = held.filter(pl.col("ticker") == "BBB")["ts"].to_list()

    assert aaa == ds[:3]        # held from day 0, closed at the day-3 rebalance
    assert bbb == ds[3:]        # opened on day 3, held from then on
    assert ds[2] not in bbb     # and not held before it was selected


def test_gross_exposure_does_not_accumulate_across_rebalances():
    """Every date carries the book size the rebalance intended, and no more.

    A fresh long/short pair is selected at each rebalance, so carrying weights
    forward per ticker would leave every earlier selection open and grow gross
    exposure by 2.0 each time. The result is a silently levered portfolio whose
    returns and Sharpe look plausible.
    """
    ds = _dates(12)
    rows = []
    for i, d in enumerate((0, 4, 8)):
        rows.append((ds[d], f"L{i}", 1.0))
        rows.append((ds[d], f"S{i}", -1.0))
    rebal = pl.DataFrame(
        rows, schema={"ts": pl.Date, "ticker": pl.Utf8, "w": pl.Float64}, orient="row"
    )

    held = hold_until_next_rebalance(rebal, pl.Series("ts", ds))
    per_date = held.group_by("ts").agg(
        pl.col("w").abs().sum().alias("gross"), pl.col("w").sum().alias("net")
    )

    assert per_date["gross"].max() == pytest.approx(2.0)
    assert per_date["net"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert held["ts"].n_unique() == 12


def test_hold_never_looks_forward():
    """A rebalance on day 3 must not appear on day 0: the same lookahead
    failure mode as the engine's, one layer up."""
    ds = _dates(4)
    rebal = pl.DataFrame(
        [(ds[3], "AAA", 1.0)],
        schema={"ts": pl.Date, "ticker": pl.Utf8, "w": pl.Float64},
        orient="row",
    )
    held = hold_until_next_rebalance(rebal, pl.Series("ts", ds))
    assert held["ts"].to_list() == [ds[3]]
