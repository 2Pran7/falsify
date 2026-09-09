"""Walk-forward evaluation: splitting time so no result is fitted in-sample.

A backtest over the whole sample reports how a strategy would have done if you
had known everything at the start. Walk-forward replaces that with a sequence
of honest questions: fit on what was knowable by date T, measure what happened
after T, move T forward, stitch the measured pieces into one series.

For a parameter-free signal like 12-1 momentum this changes little. It becomes
load-bearing the moment anything is chosen from the data, which is the whole of
M6: picking a rebalancing frequency, a decile count or the anomaly to headline
is fitting, whether or not it is called that.

The split boundaries are the entire methodological claim, so they live in their
own file where the tests can attack them directly rather than through a
backtest. Same discipline that put every forward shift in engine.forward_returns.

Conventions: boundaries are TRADING DAYS from the supplied index, never
calendar dates; intervals are inclusive at both ends; test windows are
contiguous and disjoint, so the stitched series double-counts nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt

import polars as pl


@dataclass(frozen=True)
class Split:
    """One train/test division. All bounds inclusive, all real trading days."""
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date


def expanding_splits(
    dates: pl.Series,
    n_splits: int,
    min_train: int,
    embargo: int = 0,
) -> list[Split]:
    """Anchored walk-forward: the training window grows, the anchor never moves.

    Split 1 trains on [0, min_train) and tests on the first block; split 2
    trains through the end of that block and tests on the next. Simulates a
    researcher who accumulates history and never forgets it.

    Args:
        dates:     trading days available, any order (deduped and sorted here).
        n_splits:  number of test blocks. Dates after `min_train` divide as
                   evenly as possible; the remainder goes to the LAST block, so
                   no trading day is silently discarded.
        min_train: minimum training observations before the first test day.
                   Must exceed 252 for 12-1 momentum, or the first block trades
                   a signal that is null for every name.
        embargo:   trading days carved off the END of training, opening a gap
                   before test_start. Test windows do NOT move.

    Returns:
        `n_splits` Splits in chronological order.

    Raises:
        ValueError: bad arguments, or too few dates for the requested splits.
            Returning fewer splits than asked would quietly change the
            multiple-testing count downstream.

    Embargo direction, per Lopez de Prado's purging argument: a feature built
    from a window reaching into the training period makes the days just after
    train_end not truly out of sample. mom_12_1 at t uses closes from t-252 to
    t-21, so the first year of any block overlaps training. For an unfitted
    signal that leaks nothing, which is why the default is 0; once anything is
    selected on training data, set it to the feature's lookback. It shrinks
    training rather than delaying testing because out-of-sample days are the
    scarce resource.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if min_train < 1:
        raise ValueError(f"min_train must be >= 1, got {min_train}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")

    # Everything below indexes POSITIONALLY, so an unsorted or duplicated
    # index would put every boundary on the wrong day.
    d = dates.unique().sort()
    n_dates = d.len()

    n_test_total = n_dates - min_train
    if n_test_total < n_splits:
        raise ValueError(
            f"{n_dates} dates leaves {n_test_total} after min_train={min_train}, "
            f"which cannot fill {n_splits} test blocks"
        )

    block = n_test_total // n_splits

    splits: list[Split] = []
    for k in range(n_splits):
        test_lo = min_train + k * block

        # The last block runs to the end of the sample, absorbing whatever
        # integer division left over. Dropping those days would create a
        # period the strategy was never evaluated on, silently.
        test_hi = n_dates - 1 if k == n_splits - 1 else min_train + (k + 1) * block - 1

        train_hi = test_lo - 1 - embargo
        if train_hi < 0:
            raise ValueError(
                f"embargo={embargo} consumes the whole training window "
                f"(min_train={min_train})"
            )

        # d[0] every time: that is what makes this expanding.
        splits.append(Split(d[0], d[train_hi], d[test_lo], d[test_hi]))

    return splits


def rolling_splits(
    dates: pl.Series,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
) -> list[Split]:
    """Rolling walk-forward: a fixed-length training window slides forward.

    Args:
        dates:      the trading days available.
        train_size: training observations per split, constant by construction.
        test_size:  test observations per split.
        step:       days to advance between splits. Defaults to `test_size`,
                    which makes the test windows exactly contiguous and
                    disjoint. A smaller step overlaps them, which inflates the
                    apparent sample and must not be used to build the stitched
                    out-of-sample series.
        embargo:    as in `expanding_splits`.

    Returns:
        Splits in chronological order, as many as fit. The count is derived
        rather than requested, so a short index yields fewer splits.

    Raises:
        ValueError: any size < 1, embargo < 0, or too few dates for one split.

    Rolling versus expanding is a real modelling trade-off: a fixed window
    adapts to regime change and discards old information, an expanding one uses
    everything and assumes stability. falsify headlines the expanding result
    and reports rolling as a robustness check, because a result surviving only
    one of the two is fragile.
    """
    if train_size < 1:
        raise ValueError(f"train_size must be >= 1, got {train_size}")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")

    # Defaulting to test_size makes the test windows tile exactly: no gaps,
    # no overlaps, so the stitched series double-counts nothing.
    step = test_size if step is None else step
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    d = dates.unique().sort()
    n_dates = d.len()

    splits = []
    start = 0
    while True:
        train_lo = start
        train_hi = start + train_size - 1
        test_lo  = train_hi + 1 + embargo
        test_hi  = test_lo + test_size - 1
        if test_hi > n_dates - 1:       # ran off the end, stop
            break
        splits.append(Split(d[train_lo], d[train_hi], d[test_lo], d[test_hi]))
        start += step
    if not splits:
        raise ValueError("not enough dates for a single split")
    return splits


def validate_splits(splits: list[Split]) -> None:
    """Assert the properties every split set must have. Raises on violation.

    Checks, in order:
      1. train_start <= train_end, test_start <= test_end for every split.
      2. test_start > train_end for every split: no overlap, ever.
      3. Test windows are disjoint and chronologically ordered across splits.

    A runtime guard, not a substitute for the tests: it runs inside the
    walk-forward driver so a future refactor cannot silently produce leaking
    splits. The run fails instead of printing an excellent, wrong Sharpe.
    """
    # `<=` not `<`: a test period STARTING on train_end shares that day with
    # training, and one shared day is in-sample.
    for i, s in enumerate(splits):
        if s.train_start > s.train_end:
            raise ValueError(f"split {i}: train_start {s.train_start} after train_end {s.train_end}")
        if s.test_start > s.test_end:
            raise ValueError(f"split {i}: test_start {s.test_start} after test_end {s.test_end}")
        if s.test_start <= s.train_end:
            raise ValueError(
                f"split {i}: test_start {s.test_start} overlaps train_end {s.train_end} "
                f"— this split leaks"
            )

    # Two splits sharing a test day double-count it, and n enters PSR as
    # sqrt(n - 1), so an inflated n manufactures confidence.
    for i, (a, b) in enumerate(zip(splits, splits[1:])):
        if b.test_start <= a.test_end:
            raise ValueError(
                f"splits {i} and {i + 1}: test windows overlap "
                f"({a.test_end} then {b.test_start})"
            )


def oos_returns(
    splits: list[Split],
    backtest_fn,
) -> pl.DataFrame:
    """Stitch each split's TEST-period returns into one out-of-sample series.

    Args:
        splits:      from `expanding_splits` or `rolling_splits`.
        backtest_fn: callable taking one Split and returning that split's test
                     -period daily returns as a frame (ts, ret). Everything
                     strategy-specific lives behind this callable, so this
                     module never learns what a signal is.

    Returns:
        Frame (ts, ret, split) sorted by ts, `split` being the 0-based index of
        the split each day came from — enough for per-split attribution without
        re-running anything.

    Raises:
        ValueError: `validate_splits` fails, or a date appears in more than one
            split's output. An off-by-one in a boundary is invisible in the
            metrics and doubles the weight of the overlapping days.

    The ONLY series whose Sharpe may be quoted as out-of-sample, and what feeds
    `deflated.deflated_sharpe`. A full-sample Sharpe belongs in the note only
    when labelled in-sample.
    """
    schema = {"ts": pl.Date, "ret": pl.Float64, "split": pl.Int64}

    validate_splits(splits)
    if not splits:
        return pl.DataFrame(schema=schema)

    parts = []
    for i, s in enumerate(splits):
        frame = backtest_fn(s)                      # -> (ts, ret) for that window
        frame = frame.select(["ts", "ret"]).with_columns(
            pl.lit(i, dtype=pl.Int64).alias("split")
        )
        parts.append(frame.cast(schema))

    out = pl.concat(parts).sort("ts")

    if out["ts"].n_unique() != out.height:
        raise ValueError("a date appears in more than one split's output")

    return out



    
