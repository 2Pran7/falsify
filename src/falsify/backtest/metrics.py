"""Performance metrics.

Every function takes `returns`: a pl.Series of daily SIMPLE returns (not log
returns), in chronological order. 0.01 means +1% that day.

Conventions, pinned so these metrics and the statistics layer above them agree.
Changing any of them requires changing the tests:
  - 252 trading days per year.
  - Standard deviation uses ddof=1 (sample), which is Polars' default.
  - max_drawdown is returned NEGATIVE (a 50% drawdown is -0.5).
  - Nothing here annualises by compounding except cagr.

Deflated Sharpe and multiple-testing correction layer on top, so these stay
pure: no printing, no plotting, no database access.
"""
from __future__ import annotations

import polars as pl
import math 

TRADING_DAYS = 252


def total_return(returns: pl.Series) -> float:
    """Cumulative return over the whole series: prod(1 + r) - 1."""
    if len(returns) == 0:
        return float("nan")
    return float((returns + 1.0).product() - 1)
    


def cagr(returns: pl.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate.

        (1 + total_return) ** (periods_per_year / n) - 1

    n = len(returns): the NUMBER OF OBSERVATIONS, not the calendar span, which
    is the right choice for a series of trading days only.
    """

    n = len(returns)
    if n == 0:
        return float("nan")
    return float((1 + total_return(returns)) ** (periods_per_year / n) - 1)
    


def ann_vol(returns: pl.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised volatility: std(returns, ddof=1) * sqrt(periods_per_year)."""
    sd = returns.std(ddof=1)
    if sd is None:
        return float("nan")
    return float(sd * math.sqrt(periods_per_year))

    


def sharpe(
    returns: pl.Series,
    rf: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio.

        mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)

    where excess = returns - rf / periods_per_year (rf is an ANNUAL rate).

    Zero volatility leaves the ratio undefined, so the result is nan, not 0.0:
    a silent zero would read a degenerate strategy as mediocre, not broken.
    """
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=1)
    if sd is None or sd == 0.0:
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(periods_per_year))
    
    


def max_drawdown(returns: pl.Series) -> float:
    """Worst peak-to-trough decline of the equity curve, as a negative number.
    0.0 for a series that never declines.
    """
    if len(returns) == 0:
        return float("nan")
    # Prepend a 0% day so the curve starts at 1.0: starting capital is the first
    # high-water mark. Without it, a series that only falls reports a drawdown
    # of 0, its first observation being its own peak.
    r = pl.concat([pl.Series("r", [0.0]), returns.rename("r").cast(pl.Float64)])
    equity = (r + 1.0).cum_prod()
    drawdown = equity / equity.cum_max() - 1.0
    return float(drawdown.min())



def summary(returns: pl.Series) -> dict[str, float]:
    """Every metric in one dict.

    Keys: total_return, cagr, ann_vol, sharpe, max_drawdown, n_days.
    """
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "ann_vol": ann_vol(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "n_days": float(len(returns)),
    }
    
