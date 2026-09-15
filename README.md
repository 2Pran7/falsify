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
| Statistics | Walk-forward splits, deflated Sharpe, FDR correction, survivorship audit | Complete, 111 tests |
| Agent | Model-driven hypothesis to experiment to research note, four tools, five cost guards | Complete |
| Research notes | Verified notes persisted to Postgres, publishable rule, markdown rendering | Complete |
| Eval suite | Six published anomalies rediscovered, scored, failures shown | Planned |
| Demo | Pre-computed eval-suite results served as a static page | Planned |

Full suite: **381 passing, 19 skipped** on a fresh clone with no database;
**397 passing, 3 skipped** with Postgres up.

**The headline result.** Reconstructing point-in-time S&P 500 membership from
the git history of the constituents CSV — 191 dated snapshots back to 2012, no
paid data and no API calls — shows that **survivorship bias accounted for
roughly half the headline momentum return**: 20.18% against 10.02% over a
common 239-day invested window, Sharpe 0.68 against 0.45. Volatility and max
drawdown barely moved (41.6% vs 39.3%, -33.2% vs -33.4%), which is the
signature that matters: the invisible companies were not adding risk, they were
removing return that was never earned.

The measured gap is a **lower bound**. Restoring a dropped name is not the same
as capturing its delisting return — a cash acquisition simply stops having
prices. Shumway (1997) covers the size of the missing piece.

## Design

**The engine computes; the model decides.** Every number is produced by
deterministic, tested pipeline code. The language model plans experiments and
interprets verified output, and never computes a result itself.

**Lookahead bias lives in exactly one function.** Features are computed from
data up to and including time `t`. The shift to trading time happens once, in
`backtest/engine.py`, where each date's weights are paired with that date's
forward return. That is the only `shift(-1)` on the P&L path: the two others in
the codebase are both in `data/quality.py`, where they look one bar ahead to
measure a coverage gap and a price level shift. Neither ever reaches a return
series.

**The one lookahead that is disclosed rather than removed.** `quality.py`
decides which tickers to exclude from the whole sample at once, so a splice
detected in 2026 removes that ticker from a 2025 backtest too. That is future
information shaping the universe. It is kept because the alternative is worse:
trading a fictitious price produced by two different companies spliced end to
end. The bias direction is toward cleanliness, not toward returns, and a
point-in-time gate that excludes a ticker only from the date its defect becomes
detectable is the correct fix. Scheduled for the eval-suite module.

**The model decides, and that is enforced rather than asserted.** Four
mechanisms, in increasing order of strength: the tool menu is closed (a dict of
pre-bound callables, never a `getattr` on a model-supplied string); arguments
are validated before dispatch and unknown keys are rejected rather than
ignored; results cross to the model as summaries under a 2,000-byte enforced
cap, never as payloads; and every numeral in a research note must appear in, or
be a permitted transform of, a number the model was actually shown
(`agent/provenance.py`). Percent, rounding, days-to-years and sign are
permitted transforms; differences, ratios and sums are not, because those are
arithmetic.

The fourth mechanism is what makes the claim falsifiable, and its limitation is
stated in its own docstring: **provenance checks numbers, not claims.** A note
in which every figure is traceable and the conclusion is wrong passes
completely, and the first real run did exactly that.

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

- **Survivorship bias is now measured rather than merely disclosed.**
  `data/pit_universe.py` reconstructs point-in-time membership from the git
  history of the constituents CSV, and `stats/survivorship.py` quantifies the
  gap — see the headline result above. Backtests on the current-constituents
  universe remain available and remain overstated; having both is the point.
- **Prices are split-adjusted only, never dividend-adjusted.** Polygon's
  `adjusted=true` applies splits and not dividends, so every return in this
  repo is a **price return**, not a total return, and the Ken French UMD
  comparison is not like-for-like.
- **`TRIAL_VARIANCE = 0.0009` is an assumed value, not a measured one**, and
  every deflated figure moves with it. It is printed under every table rather
  than buried, and is measured from the six anomalies in the eval-suite module.
- **There is no benchmark tool**, so a long-only result's market beta cannot be
  separated out. A long-only Sharpe is not a test of a cross-sectional
  hypothesis; the long/short spread is.
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
  data/pit_universe.py      point-in-time membership from the constituents git log
  data/quality.py           splice and gap detection (NOT point-in-time; see above)
  stats/walkforward.py      walk-forward splits with no leakage across a boundary
  stats/deflated.py         PSR, E[max SR], deflated Sharpe, minimum track record
  stats/multipletest.py     Benjamini-Hochberg and Holm
  stats/survivorship.py     current-constituents vs point-in-time, measured
  agent/session.py          handle store: payloads stay server-side, capped digests
  agent/tools.py            the boundary: closed menus, validated arguments
  agent/loop.py             the conversation loop and five cost guards
  agent/provenance.py       every numeral in a note must trace to a tool result
  notes/schema.py           the Note record and the publishable rule
  notes/store.py            notes in Postgres, failures kept as failures
  notes/render.py           Note -> markdown, table from data and prose as commentary
scripts/validate_stats.py   the statistics re-derived by simulation, not unit test
scripts/run_agent.py        one hypothesis end to end; prints and stores what it cost
scripts/show_notes.py       read notes back with no API key and no spend
tests/                      synthetic data with hand-computed expected values
db/schema.sql               daily_bars, universe_snapshot, ingest_log, research_note
```

## Research notes

A run's note is stored, not printed and forgotten. `scripts/show_notes.py`
reads one back in a process that shares nothing with the one that produced it
and that cannot make a model call, which is the whole claim: the demo serves
pre-computed rows, without a rerun and without an API key.

A note stores the **verified numbers as data** and the prose as commentary
alongside them. The page renders its tables from the data, never by parsing the
prose, and says which is which. That is the honest answer to "how do I know
this is not just a language model writing plausible text?"

A note is **publishable** only if provenance passed, the run completed on its
own (`stop_reason == "end_turn"`), and `analyze_results` was actually called.
Each of the three conditions has already been violated by a real run here.
`publishable` is a computed property rather than a stored flag, so neither a
caller nor a hand-edited database row can assert it.

**Notes that fail are stored as failures.** Deleting them is how an eval suite
becomes a highlight reel, and the count of them is a number the demo shows.

```
python scripts/run_agent.py "do low-volatility stocks outperform?"
python scripts/show_notes.py                      # list, newest first
python scripts/show_notes.py <note_id> --out note.md
```
