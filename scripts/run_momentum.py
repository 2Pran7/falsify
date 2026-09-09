"""Run the 12-1 momentum strategy over the ingested universe.

Cross-sectional Jegadeesh-Titman momentum: each month-end, rank the universe by
the trailing 12-month return skipping the most recent month, go long the top
decile and short the bottom decile, hold until the next month-end.

This produces a return series and a summary table. It does NOT produce a verdict
on whether momentum is real: nothing here accounts for the number of strategies
tried, and the sample is a single in-sample pass over one window. Those are the
concerns of the statistics layer.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import polars as pl

from falsify.backtest import metrics as m
from falsify.backtest.engine import BacktestConfig, forward_returns, run_backtest
from falsify.backtest.loader import load_panel
from falsify.backtest.portfolio import (
    decile_weights,
    fixed_weights,
    hold_until_next_rebalance,
    month_end_dates,
)
from falsify.features.library import add_momentum_12_1

COST_BPS = 10.0
N_BUCKETS = 10
BENCHMARK = "SPY"


def momentum_weights(panel: pl.DataFrame, n_buckets: int = N_BUCKETS) -> pl.DataFrame:
    """Panel of prices -> daily weights for monthly-rebalanced 12-1 momentum."""
    signal = (
        add_momentum_12_1(panel)
        .select(["ts", "ticker", pl.col("mom_12_1").alias("sig")])
        .drop_nulls("sig")
    )
    if signal.is_empty():
        return pl.DataFrame(schema={"ts": pl.Date, "ticker": pl.Utf8, "w": pl.Float64})

    rebalance_dates = pl.DataFrame({"ts": month_end_dates(signal["ts"])})
    targets = decile_weights(
        signal.join(rebalance_dates, on="ts", how="semi"),
        n_buckets=n_buckets,
        long_short=True,
    )
    return hold_until_next_rebalance(targets, panel["ts"].unique())


def _fmt(summary: dict[str, float]) -> str:
    return (
        f"  total return   {summary['total_return']:>9.2%}\n"
        f"  CAGR           {summary['cagr']:>9.2%}\n"
        f"  ann. vol       {summary['ann_vol']:>9.2%}\n"
        f"  Sharpe         {summary['sharpe']:>9.2f}\n"
        f"  max drawdown   {summary['max_drawdown']:>9.2%}\n"
        f"  trading days   {summary['n_days']:>9.0f}"
    )


def _data_sanity(panel: pl.DataFrame, top: int = 5) -> None:
    """Largest single-day moves in the universe.

    An unadjusted split or a bad tick shows up here as a return near -50% or
    +100%. At a 2% portfolio weight a single such row moves the whole book by
    more than a percent, so an implausible volatility figure is often a data
    problem rather than a market one.
    """
    fwd = forward_returns(panel.select(["ticker", "ts", "close"])).drop_nulls("fwd_ret")
    worst = fwd.with_columns(pl.col("fwd_ret").abs().alias("abs_ret")).sort(
        "abs_ret", descending=True
    ).head(top)

    print(f"\nLARGEST SINGLE-DAY MOVES IN THE UNIVERSE (top {top})")
    for r in worst.iter_rows(named=True):
        print(f"  {r['ticker']:<6} {r['ts']}  {r['fwd_ret']:>8.1%}")
    extreme = fwd.filter(pl.col("fwd_ret").abs() > 0.40).height
    print(f"  rows beyond +/-40%: {extreme}")


def _monthly_curve(returns: pl.DataFrame) -> None:
    """Month-end cumulative return, as a text table with a bar."""
    curve = (
        returns.sort("ts")
        .with_columns((pl.col("ret") + 1.0).cum_prod().alias("equity"))
        .with_columns(pl.col("ts").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col("equity").last())
        .sort("month")
    )
    print(f"\n{'month':<10}{'cumulative':>12}")
    for row in curve.iter_rows(named=True):
        cum = row["equity"] - 1.0
        bar = "#" * min(40, int(abs(cum) * 100))
        print(f"{row['month']!s:<10}{cum:>11.1%}  {bar}")


def main() -> None:
    panel = load_panel()
    if panel.is_empty():
        print("daily_bars is empty. Run scripts/run_ingest.py first.")
        return

    universe = panel.filter(pl.col("ticker") != BENCHMARK)
    print(
        f"universe: {universe['ticker'].n_unique()} tickers, "
        f"{len(universe):,} rows, {universe['ts'].min()} to {universe['ts'].max()}"
    )

    weights = momentum_weights(universe)
    if weights.is_empty():
        print(
            "\nNo dates carry a 12-1 momentum score.\n"
            "mom_12_1 needs 252 trading days of history per ticker; the free\n"
            "Polygon tier supplies about two years, so a short window can leave\n"
            "nothing scored. Check the date range printed above."
        )
        return

    n_rebalances = weights["ts"].n_unique()
    print(
        f"positions:  {weights['ticker'].n_unique()} names traded across "
        f"{n_rebalances} days, first weight {weights['ts'].min()}"
    )

    prices = universe.select(["ticker", "ts", "close"])
    res = run_backtest(prices, weights, BacktestConfig(cost_bps=COST_BPS))

    # The signal needs 252 days of history, so the early part of the sample has
    # no positions at all. Those flat days sit in the return series (correctly,
    # a strategy in cash earns nothing) but they are not the strategy's
    # behaviour, and averaging over them understates every metric. Report the
    # invested window separately and compare the benchmark over the same dates.
    invested_dates = weights.select("ts").unique()
    invested = res.returns.join(invested_dates, on="ts", how="semi").sort("ts")

    print(f"\n12-1 MOMENTUM, long/short deciles, monthly, {COST_BPS:.0f}bps costs")
    print(f"  invested from  {invested['ts'].min()} to {invested['ts'].max()}")
    print(_fmt(m.summary(invested["ret"])))

    bench = panel.filter(pl.col("ticker") == BENCHMARK)
    if len(bench) > 1:
        b = run_backtest(
            bench.select(["ticker", "ts", "close"]),
            fixed_weights(bench["ts"], BENCHMARK),
        )
        b_invested = b.returns.join(invested_dates, on="ts", how="semi").sort("ts")
        print(f"\n{BENCHMARK} BUY AND HOLD, same invested window")
        print(_fmt(m.summary(b_invested["ret"])))
        print(
            "\n  Not a like-for-like contest: the strategy is dollar-neutral long/short\n"
            "  and the benchmark is 100% long the market. They carry different risks."
        )

    _data_sanity(universe)
    _monthly_curve(invested)

    print(
        "\nOne in-sample pass over a single window, with no correction for the\n"
        "number of strategies tried. Evidence that the pipeline runs end to end,\n"
        "not a result."
    )


if __name__ == "__main__":
    main()
