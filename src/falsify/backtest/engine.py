"""Return accounting for a weighted portfolio.

    Weights decided using data up to and INCLUDING day t are applied to the
    return from day t to day t+1.

That sentence is the entire specification. Everything below is detail.

Design note:
The engine knows nothing about momentum, or signals, or ranking. It takes
weights someone else chose and accounts for them honestly. That separation is
why adding five more anomalies in Module 6 does not require re-auditing the
timing logic five more times.

Rebalancing frequency is deliberately not an engine concern. To
rebalance monthly, portfolio.py emits weights on month-ends and forward-fills
them. The engine just sees a weight for every day.
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
    """Deliberately NOT a scalar Sharpe.

    Module 3 needs the full daily return series for walk-forward splits,
    deflated Sharpe and FDR. Returning a number here would force a rewrite in
    three weeks. Metrics are computed FROM this object, never stored in it.
    """
    returns: pl.DataFrame            # ts, gross_ret, cost, ret   (ret = net)
    weights: pl.DataFrame            # ts, ticker, w
    turnover: pl.DataFrame           # ts, turnover
    config: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def ret(self) -> pl.Series:
        """Net daily returns, after costs."""
        return self.returns["ret"]


def forward_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Append `fwd_ret`: the return from t to t+1, per ticker.

        fwd_ret_t = close_{t+1} / close_t - 1

    Args:
        prices: long frame with columns (ticker, ts, close).

    Returns:
        The frame sorted by (ticker, ts) with `fwd_ret` appended. The last
        observation per ticker has a null fwd_ret; leave it null here, the
        engine drops it.

    This is the ONLY place a negative shift is allowed to appear in this
    codebase. Confining it to one function keeps lookahead bias auditable.
    """
    raise NotImplementedError


def compute_turnover(weights: pl.DataFrame) -> pl.DataFrame:
    """Per-date turnover: sum of absolute weight CHANGES.

        turnover_t = sum_i | w_{i,t} - w_{i,t-1} |

    Treat a ticker absent on the previous date as w = 0. On the first date,
    every position is new, so turnover is the gross exposure.

    Note this is the un-halved convention (going from flat to 100% long one
    name is turnover 1.0, not 0.5) and it ignores weight drift between
    rebalances. Both are simplifications. Document them in the README rather
    than pretending they are not there.

    Args:
        weights: long frame (ts, ticker, w).
    Returns:
        Frame (ts, turnover), one row per date, sorted by ts.
    """
    raise NotImplementedError


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

    The steps, in order:
      1. fwd_ret = forward_returns(prices).
      2. Join weights onto fwd_ret on (ts, ticker). Left-join FROM the weights
         so a weight with no matching price is loud, not silently dropped.
      3. gross_ret_t = sum_i w_{i,t} * fwd_ret_{i,t}, grouped by ts.
      4. cost_t = turnover_t * config.cost_bps / 10_000.
      5. ret_t = gross_ret_t - cost_t.
      6. Drop the final date, which has no forward return. Do NOT let it become
         a silent 0.0 — that quietly biases every metric downstream.
      7. Sort by ts. Return the full BacktestResult.

    Dates present in prices but absent from weights are flat: ret 0.0, and they
    still appear in the output series. A backtest that silently skips its
    out-of-position days reports the Sharpe of a strategy nobody ran.
    """
    raise NotImplementedError
