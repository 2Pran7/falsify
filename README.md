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
| Eval suite | Six published anomalies, pre-registered and hashed; four-rule scoring; universe argument | Harness complete, 120 tests. **Results blocked on a longer panel** |
| Demo | Pre-computed eval-suite results served as a static page | Planned |

Full suite: **486 passing, 34 skipped** on a fresh clone with no database;
**517 passing, 3 skipped** with Postgres up. Every new suite from Module 3
onward was validated by injecting the bug it claims to catch; Module 6 ran
eighteen injections and catches all eighteen.

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
  every deflated figure moves with it. `scripts/run_evals.py` now prints the
  assumption and the value measured from the suite's own six trials **side by
  side, and does not substitute one for the other**: replacing a labelled
  assumption with an unlabelled six-point estimate would be invisible in every
  number downstream of it.
- **The eval-suite RESULTS are not yet results.** The harness runs; the panel
  under it is roughly two years of one up market, of which the first year is
  momentum warmup. Six anomalies on ~250 invested days of a single regime
  cannot support a published claim, and the suite is built to say so —
  `insufficient_data` is a separate outcome from `fail` — rather than to fill
  the table. Nothing in `eval_result` is quotable until the panel lengthens.
- **The quality gate is still not point-in-time.** `drop_suspect_tickers`
  decides exclusions over the whole panel at once, so a splice detected in 2026
  removes that ticker from a 2025 backtest too. That is future information
  shaping the universe: the same class of bias `stats/survivorship.py` exists to
  measure, appearing in the tool written to prevent a different one. Verified by
  test and recorded in every stored note.
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
  eval/registry.py          the pre-registration: six anomalies, frozen and hashed
  eval/score.py             pass / partial / fail / insufficient_data, with reasons
  eval/runner.py            the suite, run through the same tools the agent uses
  eval/store.py             verdicts in Postgres, keyed by (anomaly, universe)
scripts/validate_stats.py   the statistics re-derived by simulation, not unit test
scripts/run_agent.py        one hypothesis end to end; prints and stores what it cost
scripts/show_notes.py       read notes back with no API key and no spend
scripts/run_evals.py        the eval suite: --compare, --check, --store, --registry
scripts/diagnose_membership.py  why every ticker shows 374 member-days
tests/                      synthetic data with hand-computed expected values
db/schema.sql               daily_bars, universe_snapshot, ingest_log, research_note,
                            eval_result
```

## The eval suite

The project's central claim is that this pipeline rediscovers published effects
and says so honestly when it does not. Module 6 makes that a checkable table
rather than a sentence here.

```
python scripts/run_evals.py --registry    # the pre-registration and its hash
python scripts/run_evals.py --compare     # both universes, gap per anomaly
python scripts/run_evals.py --check       # CI gate; exits non-zero on stale rows
```

No API key and no spend: the suite runs the pipeline, not the model. Putting a
language model in the loop would make a failing row unattributable between the
data and the model's choices.

**The predictions are fixed before the runs and hashed.** `eval/registry.py`
pins the feature, the expected direction, the published Sharpe and the history
each anomaly needs; `registry_digest()` hashes exactly those fields and every
stored verdict carries it, so *"we did not edit the expectation after seeing the
result"* is a string comparison rather than a promise. Prose fields are
deliberately excluded — a typo fixed in a citation must not invalidate stored
results, or nobody would ever fix one.

| key | citation | feature | dir | published SR | needs |
|---|---|---|---|---|---|
| `momentum_12_1` | Jegadeesh & Titman (1993) | `mom_12_1` | +1 | 0.50 | 252d |
| `short_term_reversal` | Jegadeesh (1990) | `ret_21d` | −1 | 0.35 | 21d |
| `long_term_reversal` | De Bondt & Thaler (1985) | `rev_36_12` | −1 | 0.20 | 756d |
| `low_volatility` | Ang, Hodrick, Xing & Zhang (2006) | `vol_63d` | −1 | 0.78 | 63d |
| `idiosyncratic_volatility` | Ang et al. (2006) | `ivol_63d` | −1 | 0.60 | 63d |
| `fifty_two_week_high` | George & Hwang (2004) | `pct_52w_high` | +1 | 0.55 | 252d |

**Direction is the load-bearing field.** `run_backtest` always goes long the top
bucket and short the bottom, so its Sharpe is the sign of the spread — and four
of these six predict that spread to be *negative*. Without a sign fixed in
advance, "the spread was −0.6" is unscoreable, and the temptation is to look at
the number and then decide which way the paper said it should run.

**The rule: sign, then deflation, then multiplicity. Magnitude is reported,
never gated — except downward.** A spread more than 3× the published reference
downgrades a pass to partial, because on a short sample a number that large is
likelier a defect than a discovery, and an eval suite that cannot be embarrassed
by its own best result is not measuring anything.

**Four outcomes, not three.** `insufficient_data` is separate from `fail`: an
anomaly needing three years of history on a two-year panel has not been refuted,
it has not been tested. Untestable anomalies are excluded from the
Benjamini-Hochberg correction rather than counted as nulls, because padding the
denominator would make the survivors look better for no reason.

Every published Sharpe is a **reference level, not a target**, and records where
it came from; every anomaly records the **known gap between this implementation
and the published one**, printed beside its verdict. An anomaly that fails for a
reason already known is a different finding from one that fails on its merits.

## Point-in-time universes

`fetch_data(universe="current" | "point_in_time")`. Before Module 6 there was no
such argument: every agent run loaded the current constituent list and was
therefore survivorship-inflated, while `stats/survivorship.py` — the module that
measures exactly that — was reachable only from a script. The project's headline
finding and its headline artifact did not touch. They do now, and
`run_evals.py --compare` generalises the momentum measurement to all six
anomalies.

Two implementation decisions are the whole of it, and in both cases the
plausible alternative silently produces a wrong number:

- **The mask restricts what may be HELD, never what the features may SEE.** A
  company that joined the index in March had a price history in February, and
  its 12-month momentum on the day it joined is a real, knowable number.
  Filtering the price frame to member-days would leave every entrant unscored
  for a year: a lookahead bug in reverse.
- **It is applied to the signal immediately BEFORE ranking.** Bucket edges must
  come from the names investable that day. Filtering after the sort leaves the
  deciles defined by a universe the strategy could not have traded, and the
  output still looks exactly like a backtest.

`point_in_time` **refuses to run on a single snapshot date**, which is the state
`run_ingest.py` leaves behind on its own. Membership that never changes is
today's membership: it reproduces the current-constituents result exactly, with
full coverage, no gap, and a survivorship audit that measures zero. It is the
failure mode that fails by looking healthy.

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
