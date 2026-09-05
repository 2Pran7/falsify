"""Module 2, metrics. Every expected value here is hand-computed, not
copy-pasted from a run of the code under test. That is the whole point: a test
that asserts what the code already does proves nothing.

Run:  pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from falsify.backtest import metrics as m

TD = 252


def test_zero_returns_are_flat_not_crashed():
    """252 days of exactly nothing. Every metric must be defined and boring."""
    r = pl.Series("ret", [0.0] * TD)

    assert m.total_return(r) == pytest.approx(0.0)
    assert m.ann_vol(r) == pytest.approx(0.0)
    assert m.max_drawdown(r) == pytest.approx(0.0)
    assert m.cagr(r) == pytest.approx(0.0)
    # Zero vol => Sharpe is 0/0. nan, not 0.0, not an exception.
    assert math.isnan(m.sharpe(r))


def test_sharpe_matches_hand_computation():
    """Series engineered to have mean exactly 0.001 and sample sd exactly 0.01.

    Then Sharpe = 0.001 / 0.01 * sqrt(252) = 0.1 * sqrt(252) ~= 1.5875.
    A result of 1.5843 indicates ddof=0. That is a real difference and
    the reason the convention is pinned in the docstring.
    """
    raw = pl.Series("x", [float(i) for i in range(TD)])
    z = (raw - raw.mean()) / raw.std(ddof=1)      # mean 0, sd 1
    r = (z * 0.01 + 0.001).rename("ret")          # mean 0.001, sd 0.01

    assert r.mean() == pytest.approx(0.001, abs=1e-12)
    assert r.std(ddof=1) == pytest.approx(0.01, abs=1e-12)

    expected = 0.001 / 0.01 * math.sqrt(TD)
    assert m.sharpe(r) == pytest.approx(expected, rel=1e-9)
    assert m.ann_vol(r) == pytest.approx(0.01 * math.sqrt(TD), rel=1e-9)


def test_sharpe_subtracts_the_risk_free_rate_annually():
    """rf is an ANNUAL rate and must be de-annualised before subtracting.

    Constant daily return of 0.001 with rf = 0.252 annual => daily rf = 0.001
    => excess is identically zero => zero vol => nan.
    A finite result indicates 0.252 was subtracted from each day.
    """
    r = pl.Series("ret", [0.001] * TD)
    assert math.isnan(m.sharpe(r, rf=0.252))


def test_cagr_doubles_over_one_year():
    """252 equal daily returns that compound to exactly +100%. CAGR must be 1.0."""
    daily = 2 ** (1 / TD) - 1
    r = pl.Series("ret", [daily] * TD)

    assert m.total_return(r) == pytest.approx(1.0, rel=1e-12)
    assert m.cagr(r) == pytest.approx(1.0, rel=1e-12)


def test_cagr_annualises_a_partial_year():
    """126 days (half a year) compounding to +100% annualises to +300%,
    because doubling twice is 4x. Catches people who divide instead of compound."""
    n = 126
    daily = 2 ** (1 / n) - 1
    r = pl.Series("ret", [daily] * n)

    assert m.total_return(r) == pytest.approx(1.0, rel=1e-12)
    assert m.cagr(r) == pytest.approx(3.0, rel=1e-9)


def test_max_drawdown_100_to_50_to_75():
    """Equity path 100 -> 50 -> 75. Deepest decline from a peak is -50%."""
    r = pl.Series("ret", [-0.5, 0.5])
    assert m.max_drawdown(r) == pytest.approx(-0.5, rel=1e-12)


def test_max_drawdown_counts_the_first_day():
    """A series that ONLY falls. Equity 1.0 -> 0.9 -> 0.81.

    The peak is the starting capital of 1.0, so the drawdown is -19%.
    An equity curve built without a leading 1.0 makes the first observation
    becomes its own peak and this reports -10%. That is the bug this test exists
    to catch, and it is easy to ship.
    """
    r = pl.Series("ret", [-0.1, -0.1])
    assert m.max_drawdown(r) == pytest.approx(0.81 - 1.0, rel=1e-12)


def test_max_drawdown_is_zero_for_a_monotonic_riser():
    r = pl.Series("ret", [0.01] * 10)
    assert m.max_drawdown(r) == pytest.approx(0.0)


def test_summary_has_the_agreed_keys():
    """Module 3 and the eval scorer both read this dict. Lock the keys now."""
    r = pl.Series("ret", [0.001, -0.002, 0.003] * 50)
    s = m.summary(r)

    assert set(s) == {"total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "n_days"}
    assert s["n_days"] == 150
    assert s["sharpe"] == pytest.approx(m.sharpe(r))
    assert s["max_drawdown"] <= 0.0
