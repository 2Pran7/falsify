# falsify

An autonomous quantitative research agent that tries to disprove a market
hypothesis — walk-forward backtests, deflated Sharpe, FDR correction, and
honest research notes including the failures.

**Module 1 (this stage):** Polygon.io daily bars → TimescaleDB, plus a
verified Polars feature library.

## Setup (once)

```bash
# 1. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Secrets
cp .env.example .env        # then add POLYGON_API_KEY

# 3. Database (needs Docker Desktop running)
docker compose up -d        # starts TimescaleDB, schema auto-applies
```

## Verify

```bash
pytest -v                                   # 6 feature tests must pass
python scripts/run_ingest.py AAPL MSFT      # smoke test: 2 tickers only
```

Check the data landed:
```bash
docker exec -it falsify-db psql -U falsify -c \
  "SELECT ticker, count(*), min(ts), max(ts) FROM daily_bars GROUP BY ticker;"
```
Expect ~1250 rows per ticker spanning ~5 years.

## Full ingest (503 tickers)

```bash
python scripts/run_ingest.py
```
- Free Polygon tier (5 req/min): ~100 minutes, and only ~2yr history
- Starter plan ($29/mo): set `POLYGON_RPM=100` in .env → ~5 minutes, 5yr history
- Idempotent: safe to Ctrl-C and re-run, it resumes

## Known limitation (deliberate, documented)

Universe = **current** S&P 500 members (Wikipedia snapshot), not point-in-time
membership → survivorship bias. Quantified in Module 3's audit, disclosed in
the methodology post. Point-in-time membership requires paid institutional data.

## Structure

```
src/falsify/
  config.py              env-based settings
  data/polygon_client.py rate-limited API client (429 backoff)
  data/universe.py       S&P 500 constituent scraper
  data/ingest.py         idempotent Polygon → TimescaleDB pipeline
  features/library.py    composable Polars features (returns, vol, 12-1
                         momentum, SMA, z-score) — lookahead-safe by design
tests/test_features.py   synthetic-data tests with hand-computed answers
db/schema.sql            hypertables: daily_bars, universe_snapshot, ingest_log
```
