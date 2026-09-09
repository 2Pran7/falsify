"""Survivorship-bias audit: measuring the bias instead of disclaiming it.

`universe.fetch_sp500_tickers` returns TODAY's S&P 500, so a backtest on it
never sees a company that was in the index during the sample and has since been
removed. Removals are not random: they are disproportionately the names that
fell.

`data/pit_universe.py` recovers point-in-time membership from version-control
history, which makes the counterfactual runnable. This module runs the same
strategy on both universes and reports the difference. That difference IS the
survivorship bias, in Sharpe.

Almost every backtest write-up has a sentence admitting survivorship bias.
Almost none quantify it, because point-in-time membership is normally paid for.

LIMITATION, to be stated in the write-up: restoring a dropped name to the
universe is not the same as capturing its delisting return. A company acquired
for cash simply stops having prices, so the measured gap is a LOWER BOUND on
the true bias. Shumway (1997) covers the size of the missing piece.
"""
from __future__ import annotations

import polars as pl

from falsify.backtest import metrics

_METRICS = {
    "total_return": metrics.total_return,
    "cagr": metrics.cagr,
    "ann_vol": metrics.ann_vol,
    "sharpe": metrics.sharpe,
    "max_drawdown": metrics.max_drawdown,
}


def restrict_to_members(signal: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    """Keep only (ts, ticker) signal rows whose ticker was in the index on ts.

    Args:
        signal:     long frame (ts, ticker, sig), as `portfolio.decile_weights`
                    consumes.
        membership: long frame (ts, ticker) from `pit_universe.membership_panel`.

    Returns:
        The signal frame filtered to point-in-time members, same schema.

    An inner join, deliberately: a ticker with a signal but no membership row
    that date was not in the index and must disappear, not survive as a null.
    Runs BEFORE `decile_weights`, so bucket boundaries come from the names
    actually investable that day. Filtering after ranking would leave the
    deciles defined by a universe the strategy could not have traded.
    """
    schema = signal.schema
    if signal.is_empty() or membership.is_empty():
        return pl.DataFrame(schema=schema)

    # unique() first: a repeated (ts, ticker) in the membership frame would
    # duplicate the signal row and double that name's weight in the sort.
    return (
        signal.join(
            membership.select(["ts", "ticker"]).unique(),
            on=["ts", "ticker"],
            how="inner",
        )
        .select(list(schema))
        .sort(["ts", "ticker"])
    )


def survivorship_gap(
    current_returns: pl.Series,
    pit_returns: pl.Series,
) -> dict[str, float]:
    """Metric-by-metric difference between the two runs.

    Args:
        current_returns: daily returns from the strategy run on today's
            constituent list — the biased run.
        pit_returns:     daily returns from the same strategy run on
            point-in-time membership — the honest run.

    Returns:
        For each of total_return, cagr, ann_vol, sharpe and max_drawdown: the
        `current` value, the `pit` value, and the `gap` (current - pit). Plus
        `n_days_current` and `n_days_pit`.

    A positive sharpe gap is the expected sign. A negative one is not a bug to
    suppress: it would say the dropped names were ones momentum was correctly
    short in this sample, which is a finding about the sample rather than a
    refutation of survivorship bias.

    Day counts are returned so a date mismatch, a dropped ticker with no
    ingested prices shortening the PIT run, stays visible instead of hiding
    inside a ratio.
    """
    out: dict[str, float] = {}
    for name, fn in _METRICS.items():
        c = fn(current_returns)
        p = fn(pit_returns)
        out[f"{name}_current"] = c
        out[f"{name}_pit"] = p
        out[f"{name}_gap"] = c - p
    out["n_days_current"] = float(len(current_returns))
    out["n_days_pit"] = float(len(pit_returns))
    return out


def coverage_report(
    membership: pl.DataFrame,
    panel: pl.DataFrame,
) -> pl.DataFrame:
    """Which point-in-time members are missing from the price panel, and when.

    Args:
        membership: (ts, ticker) from `pit_universe.membership_panel`.
        panel:      the price panel, any frame with (ts, ticker).

    Returns:
        Frame (ticker, n_member_days, n_price_days, missing_days), sorted by
        missing_days descending.

    Run BEFORE trusting any survivorship number. A dropped ticker whose prices
    were never ingested contributes nothing to the PIT run, so the audit would
    silently under-measure the bias and report a comfortingly small gap.
    """
    schema = {
        "ticker": pl.Utf8,
        "n_member_days": pl.UInt32,
        "n_price_days": pl.UInt32,
        "missing_days": pl.UInt32,
    }
    if membership.is_empty():
        return pl.DataFrame(schema=schema)

    member_days = (
        membership.select(["ts", "ticker"]).unique()
        .group_by("ticker").len().rename({"len": "n_member_days"})
    )

    # Only price rows on days the ticker was a member count. A name with prices
    # outside its membership window, or never a member at all like SPY, would
    # otherwise make coverage look better than it is.
    price_days = (
        membership.select(["ts", "ticker"]).unique()
        .join(panel.select(["ts", "ticker"]).unique(), on=["ts", "ticker"], how="inner")
        .group_by("ticker").len().rename({"len": "n_price_days"})
    )

    return (
        member_days.join(price_days, on="ticker", how="left")
        .with_columns(pl.col("n_price_days").fill_null(0))
        .with_columns(
            (pl.col("n_member_days") - pl.col("n_price_days")).alias("missing_days")
        )
        .select(["ticker", "n_member_days", "n_price_days", "missing_days"])
        .cast(schema)
        .sort(["missing_days", "ticker"], descending=[True, False])
    )
