"""Attribution and data-quality diagnostics for a backtest result.

A summary table cannot distinguish a strategy from an accident. This script
decomposes a return series into the rows that produced it, and answers three
questions that a Sharpe ratio cannot:

  1. Does the universe contain price moves that are not physically plausible?
  2. Were those rows actually held, and what did they contribute?
  3. How concentrated is the result? A return delivered by a handful of days is
     not evidence of an effect.

Usage:  python scripts/diagnose.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import polars as pl

from falsify.backtest import metrics as m
from falsify.backtest.engine import BacktestConfig, forward_returns, run_backtest
from falsify.backtest.loader import load_panel

sys.path.insert(0, "scripts")
from run_momentum import BENCHMARK, COST_BPS, momentum_weights  # noqa: E402

# A single-name daily move beyond this is treated as suspect. Real equities do
# exceed 35% on earnings and takeover news, so this flags candidates for
# inspection rather than errors.
SUSPECT_MOVE = 0.35


def suspect_rows(panel: pl.DataFrame, threshold: float = SUSPECT_MOVE) -> pl.DataFrame:
    """Flag single-name moves beyond `threshold`, with the preceding day's move.

    `prev_ret` separates the two failure modes. A large move that immediately
    reverses (-90% then +900%) is a bad print: the price was briefly wrong and
    corrected. A large move with a quiet day before it is a level shift, which
    means either genuine news or an unadjusted corporate action, and the way to
    tell those apart is to look the date up outside this system.
    """
    fwd = (
        forward_returns(panel.select(["ticker", "ts", "close"]))
        .drop_nulls("fwd_ret")
        .sort(["ticker", "ts"])
        .with_columns(pl.col("fwd_ret").shift(1).over("ticker").alias("prev_ret"))
    )
    return (
        fwd.filter(pl.col("fwd_ret").abs() > threshold)
        .with_columns(pl.col("close").alias("close_t"))
        .select(["ticker", "ts", "close_t", "prev_ret", "fwd_ret"])
        .sort(["ticker", "ts"])
    )


def contributions(panel: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    """Per (date, ticker) contribution to that date's gross portfolio return."""
    fwd = forward_returns(panel.select(["ticker", "ts", "close"]))
    return (
        weights.join(fwd.select(["ts", "ticker", "fwd_ret"]), on=["ts", "ticker"], how="left")
        .drop_nulls("fwd_ret")
        .with_columns((pl.col("w") * pl.col("fwd_ret")).alias("contrib"))
    )


def main() -> None:
    panel = load_panel()
    if panel.is_empty():
        print("daily_bars is empty.")
        return

    universe = panel.filter(pl.col("ticker") != BENCHMARK)
    weights = momentum_weights(universe)
    if weights.is_empty():
        print("No weights; nothing to attribute.")
        return

    res = run_backtest(
        universe.select(["ticker", "ts", "close"]), weights, BacktestConfig(cost_bps=COST_BPS)
    )
    invested = res.returns.join(weights.select("ts").unique(), on="ts", how="semi").sort("ts")
    contrib = contributions(universe, weights)

    # ------------------------------------------------------------------
    print("=" * 68)
    print("1. IMPLAUSIBLE PRICE MOVES IN THE UNIVERSE")
    print("=" * 68)
    suspects = suspect_rows(universe)
    held = weights.select(["ts", "ticker", "w"])
    suspects = suspects.join(held, on=["ts", "ticker"], how="left")
    print(
        f"{'ticker':<8}{'date':<12}{'close':>10}{'prev-day':>10}"
        f"{'next-day':>11}{'weight':>11}{'impact':>9}"
    )
    for r in suspects.iter_rows(named=True):
        w = r["w"]
        prev = "n/a" if r["prev_ret"] is None else f"{r['prev_ret']:.1%}"
        impact = "" if w is None else f"{w * r['fwd_ret']:>8.1%}"
        wtxt = "not held" if w is None else f"{w:.3f}"
        print(
            f"{r['ticker']:<8}{r['ts']!s:<12}{r['close_t']:>10.2f}{prev:>10}"
            f"{r['fwd_ret']:>11.1%}{wtxt:>11}{impact:>9}"
        )
    if suspects.is_empty():
        print("  none beyond +/-35%")
    print(
        "\n  'impact' is that single row's contribution to that day's portfolio\n"
        "  return. Anything past a couple of percent from one name is a red flag.\n"
        "\n  Reading 'prev-day': a large move immediately after an opposite move of\n"
        "  similar size is a bad print that corrected itself. A large move with a\n"
        "  quiet day before it is a level shift, so either real news or an\n"
        "  unadjusted corporate action. Verify those dates against an outside\n"
        "  source before believing any result that depends on them."
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("2. LARGEST PORTFOLIO DAYS, AND WHAT DROVE THEM")
    print("=" * 68)
    worst = invested.with_columns(pl.col("ret").abs().alias("a")).sort(
        "a", descending=True
    ).head(8)
    for r in worst.iter_rows(named=True):
        top = (
            contrib.filter(pl.col("ts") == r["ts"])
            .with_columns(pl.col("contrib").abs().alias("a"))
            .sort("a", descending=True)
            .head(3)
        )
        drivers = "  ".join(
            f"{t['ticker']}:{t['contrib']:+.2%}" for t in top.iter_rows(named=True)
        )
        print(f"  {r['ts']}  {r['ret']:>8.2%}   {drivers}")
    print(
        "\n  A diversified 100-name book should rarely move more than a few\n"
        "  percent in a day, and no single name should dominate."
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("3. CONCENTRATION: IS THIS A STRATEGY OR A FEW LUCKY DAYS?")
    print("=" * 68)
    total = m.total_return(invested["ret"])
    best = invested.sort("ret", descending=True)
    worst = invested.sort("ret")
    print(f"  full sample: {total:>8.2%} over {len(invested)} days\n")
    for k in (1, 5, 10):
        print(
            f"  excluding the {k:>2} best day(s):  {m.total_return(best.slice(k)['ret']):>8.2%}"
            f"     excluding the {k:>2} worst: {m.total_return(worst.slice(k)['ret']):>8.2%}"
        )
    n_pos = (invested["ret"] > 0).sum()
    print(f"\n  up days: {n_pos} of {len(invested)}  ({n_pos / len(invested):.1%})")
    print(
        "\n  The left column is the real test. If dropping a few best days wipes\n"
        "  out the return, the return was those days rather than the hypothesis.\n"
        "  The right column shows how much of the risk sits in a few sessions;\n"
        "  a large gap there means the edge is being paid for with crash risk."
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("4. IS THE BOOK ACTUALLY DIVERSIFIED?")
    print("=" * 68)
    name_vol = (
        universe.sort(["ticker", "ts"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("r")
        )
        .group_by("ticker")
        .agg(pl.col("r").std(ddof=1).alias("sd"))
        .drop_nulls("sd")
    )
    indep = (
        weights.join(name_vol, on="ticker", how="inner")
        .with_columns((pl.col("w") * pl.col("sd")).pow(2).alias("v"))
        .group_by("ts")
        .agg(pl.col("v").sum().sqrt().alias("indep_sd"))
    )
    indep_ann = float(indep["indep_sd"].mean()) * (252**0.5)
    actual_ann = m.ann_vol(invested["ret"])
    print(f"  volatility if holdings moved independently: {indep_ann:>7.1%}")
    print(f"  volatility actually realised:               {actual_ann:>7.1%}")
    print(f"  ratio:                                      {actual_ann / indep_ann:>7.1f}x")
    print(
        "\n  Holding 100 names is not diversification if they are the same bet.\n"
        "  A ratio near 1 means the positions genuinely offset. A large ratio\n"
        "  means the book is one concentrated exposure wearing 100 tickers,\n"
        "  which is what an unconstrained momentum sort produces when a single\n"
        "  sector has been running."
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("5. DID SUSPECT TICKERS EVER ENTER THE PORTFOLIO?")
    print("=" * 68)
    print("  A corrupted price level poisons that name's momentum score for a")
    print("  full year afterwards, not just on the day of the jump.\n")
    names = suspects["ticker"].unique().to_list()
    held_suspects = contrib.filter(pl.col("ticker").is_in(names))
    if held_suspects.is_empty():
        print("  none of the flagged names were ever held")
    else:
        summary = (
            held_suspects.group_by("ticker")
            .agg(
                pl.len().alias("days_held"),
                pl.col("contrib").sum().alias("total_contrib"),
            )
            .sort("total_contrib", descending=True)
        )
        print(f"  {'ticker':<8}{'days held':>11}{'total contribution':>21}")
        for r in summary.iter_rows(named=True):
            print(f"  {r['ticker']:<8}{r['days_held']:>11}{r['total_contrib']:>20.2%}")
        print(f"\n  full-sample total return for comparison: {total:.2%}")


if __name__ == "__main__":
    main()
