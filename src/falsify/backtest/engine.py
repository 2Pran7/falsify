"""Return accounting for a weighted portfolio.

One timing rule, which is the entire specification:

    Weights decided using data up to and including day t are applied to the
    return from day t to day t+1.

The engine knows nothing about signals, ranking or rebalancing frequency. It
accounts for weights chosen elsewhere, so new strategies can be added without
re-auditing the timing logic. portfolio.py carries rebalance weights forward,
so the engine sees a weight for every day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True)
class BacktestConfig:
    """cost_bps: one-way transaction cost in basis points, charged on turnover.
    10.0 means 10bps = 0.001."""
    cost_bps: float = 0.0


@dataclass
class BacktestResult:
    """Full backtest output: the daily return series, not a summary statistic.

    Walk-forward splits, deflated Sharpe and multiple-testing correction all
    need the complete series, so it is returned intact. Metrics are computed
    from this object rather than stored on it.
    """
    returns: pl.DataFrame            # ts, gross_ret, cost, ret   (ret = net)
    weights: pl.DataFrame            # ts, ticker, w
    turnover: pl.DataFrame           # ts, turnover
    config: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def ret(self) -> pl.Series:
        """Net daily returns, after costs."""
        return self.returns["ret"]


_W_SCHEMA = {"ts": pl.Date, "ticker": pl.Utf8, "w": pl.Float64}


def forward_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Append `fwd_ret`: the return from t to t+1, per ticker.

        fwd_ret_t = close_{t+1} / close_t - 1

    Args:
        prices: long frame with columns (ticker, ts, close).

    Returns:
        The frame sorted by (ticker, ts) with `fwd_ret` appended. The final
        observation per ticker has a null fwd_ret, which run_backtest drops.

    This is the only forward-looking shift in the codebase. Confining it to one
    function is what makes the absence of lookahead bias auditable.
    """
    # Sort first: a shift is POSITIONAL, so "the next row" only means "tomorrow"
    # if rows are already in date order within each ticker. .over("ticker")
    # keeps the shift inside each company's block, so AAPL's last row cannot
    # reach into AMZN's first.
    return prices.sort(["ticker", "ts"]).with_columns(
        (pl.col("close").shift(-1).over("ticker") / pl.col("close") - 1.0).alias("fwd_ret")
    )


def compute_turnover(weights: pl.DataFrame) -> pl.DataFrame:
    """Per-date turnover: sum of absolute weight CHANGES.

        turnover_t = sum_i | w_{i,t} - w_{i,t-1} |

    Treat a ticker absent on the previous date as w = 0. On the first date,
    every position is new, so turnover is the gross exposure.

    Two simplifications, both disclosed in the README: this is the un-halved
    convention (moving from flat to 100% long a single name is turnover 1.0,
    not 0.5), and weight drift between rebalances is ignored.

    Args:
        weights: long frame (ts, ticker, w).
    Returns:
        Frame (ts, turnover), one row per date, sorted by ts.
    """
    if weights.is_empty():
        return pl.DataFrame(schema={"ts": pl.Date, "turnover": pl.Float64})

    # Every (date, ticker) pair, so a ticker that DISAPPEARS still gets a row
    # with w = 0 and counts as a sale. Without this, exits are invisible and
    # turnover is understated.
    dates = pl.DataFrame({"ts": weights["ts"].unique().sort()})
    tickers = pl.DataFrame({"ticker": weights["ticker"].unique().sort()})
    grid = dates.join(tickers, how="cross")

    return (
        grid.join(weights, on=["ts", "ticker"], how="left")
        .with_columns(pl.col("w").fill_null(0.0))       # not held that day = 0
        .sort(["ticker", "ts"])
        .with_columns(
            # fill_null(0.0) covers the first date: the book starts flat, so
            # every position is new.
            (pl.col("w") - pl.col("w").shift(1).over("ticker").fill_null(0.0))
            .abs().alias("dw")
        )
        .group_by("ts").agg(pl.col("dw").sum().alias("turnover"))
        .sort("ts")
    )


def run_backtest(
    prices: pl.DataFrame,
    weights: pl.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Turn weights into an honest daily return series.

    Args:
        prices:  long frame (ticker, ts, close).
        weights: long frame (ts, ticker, w). w is the target weight DECIDED
                 using information available up to and including ts.
        config:  BacktestConfig; defaults to zero costs.

    Returns:
        BacktestResult.

    Each date's gross return is the weighted sum of that date's forward returns;
    costs are turnover * cost_bps / 10_000, subtracted to give the net return.

    Two edge cases, both deliberate: the final date carries no forward return
    and is dropped rather than recorded as 0.0, since a silent zero would bias
    every downstream mean and volatility. Dates present in prices but absent
    from weights are reported flat at 0.0 rather than omitted, since dropping
    out-of-position days would report the Sharpe of a strategy never run.
    """
    config = config or BacktestConfig()
    fwd = forward_returns(prices)

    # Date spine: every trading day except the last, which has no forward
    # return. Driving output from the spine rather than the weights is what
    # keeps out-of-position days in the series at 0.0 instead of dropping them.
    all_dates = fwd["ts"].unique().sort()
    spine = pl.DataFrame({"ts": all_dates.head(len(all_dates) - 1)})

    w = weights if not weights.is_empty() else pl.DataFrame(schema=_W_SCHEMA)

    # The timing rule, enforced: each date's weight is paired with that date's
    # forward return. w is decided from data through ts; fwd_ret is what follows.
    gross = (
        w.join(fwd.select(["ts", "ticker", "fwd_ret"]), on=["ts", "ticker"], how="left")
        .with_columns((pl.col("w") * pl.col("fwd_ret")).alias("contrib"))
        .group_by("ts").agg(pl.col("contrib").sum().alias("gross_ret"))
    )

    turn = compute_turnover(w)

    out = (
        spine
        .join(gross, on="ts", how="left")
        .join(turn, on="ts", how="left")
        .with_columns(
            pl.col("gross_ret").fill_null(0.0),   # no position that day = flat
            pl.col("turnover").fill_null(0.0),
        )
        .with_columns((pl.col("turnover") * config.cost_bps / 10_000.0).alias("cost"))
        .with_columns((pl.col("gross_ret") - pl.col("cost")).alias("ret"))
        .sort("ts")
    )

    return BacktestResult(
        returns=out.select(["ts", "gross_ret", "cost", "ret"]),
        weights=w,
        turnover=turn,
        config=config,
    )
