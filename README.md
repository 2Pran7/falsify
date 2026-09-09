# falsify

An autonomous quantitative research agent that tries to disprove a market
hypothesis rather than confirm it: walk-forward backtests, deflated Sharpe,
multiple-testing correction, and research notes that report the failures.

The name is Popperian. Any backtest can be made to lie; the engineering problem
worth solving is building one that refuses to.

## Status

| Stage | Scope | State |
|---|---|---|
| Data layer | Polygon.io daily bars into TimescaleDB, S&P 500 universe, verified Polars feature library | Complete |
| Backtester | Return accounting, portfolio construction, performance metrics | Complete, 46 tests |
| Statistics | Walk-forward splits, deflated Sharpe, FDR correction, survivorship audit | In progress |
| Agent | Model-driven hypothesis to experiment to research note | Planned |
| Demo | Pre-computed eval-suite results served as a static page | Planned |

## Design

**The engine computes; the model decides.** Every number is produced by
deterministic, tested pipeline code. The language model plans experiments and
interprets verified output, and never computes a result itself.

**Lookahead bias lives in exactly one function.** Features are computed from
data up to and including time `t`. The shift to trading time happens once, in
`backtest/engine.py`, where each date's weights are paired with that date's
forward return. `shift(-1)` appears nowhere else in the codebase.

**Correctness is established by controls, not by inspection.** The engine test
suite includes a positive control (a perfect-foresight signal must produce an
enormous Sharpe, establishing that the engine can express a profitable strategy
at all) and a negative control (buying the most recent winner on a strictly
alternating return series must lose money). An off-by-one in the return
accounting flips the negative control from -88% to +573%, which reading a Sharpe
ratio would not reveal.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env        # add POLYGON_API_KEY
docker compose up -d        # TimescaleDB; schema applies on first run
```

Verify:

```bash
pytest -q                                   # 43 tests, 46 once SPY is ingested
python scripts/run_ingest.py AAPL MSFT      # two-ticker smoke test

docker exec -it falsify-db psql -U falsify -c \
  "SELECT ticker, count(*), min(ts), max(ts) FROM daily_bars GROUP BY ticker;"
```

## Full ingest

```bash
python scripts/run_ingest.py         # 503 tickers
python scripts/run_ingest.py SPY     # benchmark, not an index constituent
```

The free Polygon tier (5 req/min) takes roughly 100 minutes and returns 2 years
of history. On the Starter plan, `POLYGON_RPM=100` gives ~5 minutes and 5 years.
The ingest is idempotent and resumes after an interrupt.

## Known limitations

Disclosed rather than hidden, and quantified in the survivorship audit:

- **Survivorship bias.** The universe is the *current* S&P 500 membership, not
  point-in-time historical membership, so delisted and dropped companies are
  absent and returns are overstated. Point-in-time membership requires paid
  institutional data.
- **Turnover accounting** uses the un-halved convention and ignores weight drift
  between rebalances.
- **The risk-free rate** is a scalar rather than a daily series.

## Layout

```
src/falsify/
  config.py                 environment-based settings
  data/polygon_client.py    rate-limited API client with 429 backoff
  data/universe.py          S&P 500 constituents (CSV primary, HTML fallback)
  data/ingest.py            idempotent Polygon to TimescaleDB pipeline
  features/library.py       composable Polars features, lookahead-safe by design
  backtest/loader.py        database access
  backtest/portfolio.py     signal to weights: decile sorts, rebalancing
  backtest/engine.py        weights to daily return series
  backtest/metrics.py       Sharpe, CAGR, volatility, drawdown
tests/                      synthetic data with hand-computed expected values
db/schema.sql               hypertables: daily_bars, universe_snapshot, ingest_log
```
