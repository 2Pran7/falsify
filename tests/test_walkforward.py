"""Tests for walk-forward splitting.

The split boundaries ARE the methodological claim, so most tests attack them
directly rather than through a backtest. The last section runs a strategy
through the driver: correct boundaries can still be wired up wrongly.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from falsify.stats.walkforward import (
    Split,
    expanding_splits,
    oos_returns,
    rolling_splits,
    validate_splits,
)


def trading_days(n: int, start: dt.date = dt.date(2020, 1, 1)) -> pl.Series:
    """n consecutive weekdays. Skipping weekends surfaces calendar off-by-ones
    that contiguous daily dates would hide."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return pl.Series("ts", out)


# ==========================================================================
# expanding_splits
# ==========================================================================

def test_expanding_returns_the_requested_number_of_splits():
    d = trading_days(1000)
    assert len(expanding_splits(d, n_splits=4, min_train=400)) == 4


def test_expanding_anchor_never_moves():
    """Every training window starts on the first available date: what
    distinguishes expanding from rolling."""
    d = trading_days(1000)
    splits = expanding_splits(d, n_splits=4, min_train=400)
    assert {s.train_start for s in splits} == {d[0]}


def test_expanding_training_window_grows():
    d = trading_days(1000)
    ends = [s.train_end for s in expanding_splits(d, n_splits=4, min_train=400)]
    assert ends == sorted(ends)
    assert len(set(ends)) == 4


def test_expanding_first_train_respects_min_train():
    """A first test day before min_train observations would trade 12-1 momentum
    on a signal that is null for every name."""
    d = trading_days(1000)
    s0 = expanding_splits(d, n_splits=3, min_train=400)[0]
    n_train = d.filter((d >= s0.train_start) & (d <= s0.train_end)).len()
    assert n_train >= 400


def test_expanding_never_overlaps_train_and_test():
    """The property that matters most: a test day at or before train_end is
    in-sample, and the resulting Sharpe describes the fit."""
    d = trading_days(1200)
    for s in expanding_splits(d, n_splits=5, min_train=300):
        assert s.test_start > s.train_end


def test_expanding_test_windows_are_disjoint_and_ordered():
    d = trading_days(1200)
    splits = expanding_splits(d, n_splits=5, min_train=300)
    for a, b in zip(splits, splits[1:]):
        assert a.test_end < b.test_start


def test_expanding_covers_every_date_after_min_train():
    """A dropped remainder is a period the strategy was never evaluated on, and
    nothing in the output would say so."""
    d = trading_days(1007)          # not divisible by n_splits
    splits = expanding_splits(d, n_splits=4, min_train=400)
    assert splits[-1].test_end == d[-1]
    covered = 0
    for s in splits:
        covered += d.filter((d >= s.test_start) & (d <= s.test_end)).len()
    assert covered == 1007 - 400


def test_expanding_boundaries_are_real_trading_days():
    """A boundary on a market holiday shifts to the next available date, moving
    the split silently."""
    d = trading_days(800)
    valid = set(d.to_list())
    for s in expanding_splits(d, n_splits=3, min_train=300):
        assert {s.train_start, s.train_end, s.test_start, s.test_end} <= valid


def test_expanding_embargo_is_carved_out_of_training_not_testing():
    """The embargo removes the LAST `embargo` training days and leaves the test
    windows where they were.

    Purging discards training observations whose feature windows reach into the
    test period. Pushing the test window later instead spends out-of-sample
    days, the scarce resource, so coverage must be embargo-invariant.
    """
    d = trading_days(1000)
    plain = expanding_splits(d, n_splits=3, min_train=400, embargo=0)
    gapped = expanding_splits(d, n_splits=3, min_train=400, embargo=21)
    for p_, g in zip(plain, gapped):
        assert d.filter((d > g.train_end) & (d < g.test_start)).len() == 21
        assert g.test_start == p_.test_start          # test windows unmoved
        assert g.test_end == p_.test_end
        assert g.train_end < p_.train_end             # training window shortened


def test_expanding_rejects_impossible_requests():
    """Failing beats returning fewer splits than asked: the split count feeds the
    trial count in the multiple-testing correction, so silently changing it
    re-inflates every deflated Sharpe downstream."""
    d = trading_days(500)
    with pytest.raises(ValueError):
        expanding_splits(d, n_splits=10, min_train=495)     # 5 dates, 10 splits
    with pytest.raises(ValueError):
        expanding_splits(d, n_splits=3, min_train=600)      # min_train > sample
    with pytest.raises(ValueError):
        expanding_splits(d, n_splits=0, min_train=100)
    with pytest.raises(ValueError):
        expanding_splits(d, n_splits=3, min_train=100, embargo=-1)


def test_expanding_handles_unsorted_and_duplicated_dates():
    """A date index from a DB join may be neither sorted nor unique."""
    d = trading_days(600)
    messy = pl.concat([d.slice(300, 300), d.slice(0, 300), d.slice(0, 50)])
    assert expanding_splits(messy, 3, 200) == expanding_splits(d, 3, 200)


# ==========================================================================
# rolling_splits
# ==========================================================================

def test_rolling_training_window_is_constant_length():
    d = trading_days(1200)
    for s in rolling_splits(d, train_size=400, test_size=100):
        assert d.filter((d >= s.train_start) & (d <= s.train_end)).len() == 400


def test_rolling_anchor_moves():
    """Old data is dropped, not accumulated."""
    d = trading_days(1200)
    starts = [s.train_start for s in rolling_splits(d, train_size=400, test_size=100)]
    assert len(set(starts)) == len(starts)
    assert starts == sorted(starts)


def test_rolling_default_step_makes_tests_contiguous():
    """Default step = test_size, so test windows tile the period exactly. A
    smaller step double-counts days in the stitched series and inflates n, which
    enters PSR as sqrt(n - 1): inflated n manufactures confidence."""
    d = trading_days(1200)
    splits = rolling_splits(d, train_size=400, test_size=100)
    for a, b in zip(splits, splits[1:]):
        assert a.test_end < b.test_start
        n_between = d.filter((d > a.test_end) & (d < b.test_start)).len()
        assert n_between == 0


def test_rolling_never_overlaps_train_and_test():
    d = trading_days(1500)
    for s in rolling_splits(d, train_size=500, test_size=125):
        assert s.test_start > s.train_end


def test_rolling_embargo_opens_a_gap():
    d = trading_days(1200)
    for s in rolling_splits(d, train_size=400, test_size=100, embargo=10):
        assert d.filter((d > s.train_end) & (d < s.test_start)).len() == 10


def test_rolling_yields_as_many_splits_as_fit():
    d = trading_days(1000)
    splits = rolling_splits(d, train_size=400, test_size=100)
    assert len(splits) == 6           # 400 train, then 600 / 100
    assert splits[-1].test_end == d[-1]


def test_rolling_rejects_impossible_requests():
    d = trading_days(300)
    with pytest.raises(ValueError):
        rolling_splits(d, train_size=400, test_size=100)   # not even one split
    with pytest.raises(ValueError):
        rolling_splits(d, train_size=100, test_size=0)
    with pytest.raises(ValueError):
        rolling_splits(d, train_size=0, test_size=50)


# ==========================================================================
# validate_splits — the runtime guard
# ==========================================================================

D = dt.date


def test_validate_accepts_a_clean_split_set():
    validate_splits(
        [
            Split(D(2020, 1, 1), D(2020, 6, 30), D(2020, 7, 1), D(2020, 12, 31)),
            Split(D(2020, 1, 1), D(2020, 12, 31), D(2021, 1, 1), D(2021, 6, 30)),
        ]
    )


def test_validate_catches_an_overlapping_split():
    """The leak this guard exists for."""
    with pytest.raises(ValueError):
        validate_splits(
            [Split(D(2020, 1, 1), D(2020, 7, 1), D(2020, 7, 1), D(2020, 12, 31))]
        )


def test_validate_catches_test_before_train():
    with pytest.raises(ValueError):
        validate_splits(
            [Split(D(2020, 6, 1), D(2020, 12, 31), D(2020, 1, 1), D(2020, 5, 31))]
        )


def test_validate_catches_inverted_bounds():
    with pytest.raises(ValueError):
        validate_splits(
            [Split(D(2020, 12, 31), D(2020, 1, 1), D(2021, 1, 1), D(2021, 6, 30))]
        )


def test_validate_catches_overlapping_test_windows_across_splits():
    with pytest.raises(ValueError):
        validate_splits(
            [
                Split(D(2020, 1, 1), D(2020, 6, 30), D(2020, 7, 1), D(2020, 12, 31)),
                Split(D(2020, 1, 1), D(2020, 9, 30), D(2020, 10, 1), D(2021, 6, 30)),
            ]
        )


def test_validate_accepts_an_empty_list():
    validate_splits([])


# ==========================================================================
# oos_returns — the stitched out-of-sample series
# ==========================================================================

def _fake_backtest(dates: pl.Series, value: float = 0.001):
    """backtest_fn returning a constant return on each test-window day."""
    def fn(split: Split) -> pl.DataFrame:
        window = dates.filter((dates >= split.test_start) & (dates <= split.test_end))
        return pl.DataFrame({"ts": window, "ret": [value] * window.len()})
    return fn


def test_oos_returns_stitches_every_test_day_exactly_once():
    d = trading_days(1000)
    splits = expanding_splits(d, n_splits=4, min_train=400)
    out = oos_returns(splits, _fake_backtest(d))
    assert out.height == 600
    assert out["ts"].n_unique() == 600
    assert out["ts"].to_list() == sorted(out["ts"].to_list())


def test_oos_returns_tags_the_source_split():
    """Per-split attribution without re-running: a strategy whose whole
    out-of-sample return came from one split is fragile, and the tag shows it."""
    d = trading_days(1000)
    splits = expanding_splits(d, n_splits=4, min_train=400)
    out = oos_returns(splits, _fake_backtest(d))
    assert sorted(out["split"].unique().to_list()) == [0, 1, 2, 3]
    assert out.group_by("split").len()["len"].to_list() == [150] * 4


def test_oos_returns_starts_after_the_training_period():
    """An out-of-sample day predating the first test window means the stitched
    series contains in-sample days and its Sharpe is meaningless."""
    d = trading_days(1000)
    splits = expanding_splits(d, n_splits=4, min_train=400)
    out = oos_returns(splits, _fake_backtest(d))
    assert out["ts"].min() == splits[0].test_start
    assert out["ts"].min() > splits[0].train_end


def test_oos_returns_rejects_duplicate_dates():
    """A boundary off-by-one doubles the weight of the overlapping days and is
    invisible in the metrics. Fail instead."""
    d = trading_days(600)
    bad = [
        Split(d[0], d[299], d[300], d[449]),
        Split(d[0], d[399], d[400], d[599]),   # test windows overlap
    ]
    with pytest.raises(ValueError):
        oos_returns(bad, _fake_backtest(d))


def test_oos_returns_validates_the_splits_it_is_given():
    d = trading_days(600)
    leaking = [Split(d[0], d[400], d[300], d[599])]
    with pytest.raises(ValueError):
        oos_returns(leaking, _fake_backtest(d))


def test_oos_returns_empty_splits_gives_typed_empty_frame():
    out = oos_returns([], _fake_backtest(trading_days(10)))
    assert out.is_empty()
    assert out.schema["ts"] == pl.Date


# ==========================================================================
# The negative control: walk-forward must destroy an in-sample-fitted edge
# ==========================================================================

def test_walk_forward_kills_a_signal_fitted_on_the_training_data():
    """The discriminating test for this module.

    A "strategy" that looks up each day's actual return is perfect in-sample and
    worthless out-of-sample. If both look good the splits leak, and every
    downstream statistic measures the fit rather than the strategy. Mirrors
    `test_engine.py`'s negative control: prove the guard fires by building
    something it must catch.
    """
    import random

    random.seed(42)
    d = trading_days(1000)
    truth = {ts: random.gauss(0.0, 0.01) for ts in d.to_list()}

    splits = expanding_splits(d, n_splits=4, min_train=400)

    def in_sample_cheat(split: Split) -> pl.DataFrame:
        """Fit sign on TRAIN, apply that fixed sign on TEST. The train period is
        memorised perfectly; the series is pure noise, so the memorised sign
        carries nothing forward and the test period averages to ~0."""
        train = [ts for ts in d.to_list() if split.train_start <= ts <= split.train_end]
        fitted_sign = {ts: (1 if truth[ts] > 0 else -1) for ts in train}
        window = [ts for ts in d.to_list() if split.test_start <= ts <= split.test_end]
        # Unseen dates have no fitted sign: fall back to always-long.
        rets = [fitted_sign.get(ts, 1) * truth[ts] for ts in window]
        return pl.DataFrame({"ts": window, "ret": rets})

    in_sample = sum(abs(truth[ts]) for s in splits
                    for ts in d.to_list() if s.train_start <= ts <= s.train_end)
    oos = oos_returns(splits, in_sample_cheat)

    # Compare the two rather than thresholding the out-of-sample sum absolutely,
    # which would be a bet on the seed.
    assert in_sample > 5.0
    assert abs(oos["ret"].sum()) < in_sample / 5
