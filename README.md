# falsify

**An autonomous quant research agent that tries to disprove market hypotheses instead of confirming them.**

You type a hypothesis in English. It fetches point-in-time data, builds features, runs walk-forward backtests, applies deflated Sharpe and multiple-testing correction, and writes a research note that reports failure when the evidence fails. The name is Popperian: any backtest can be made to lie, and the engineering problem worth solving is building one that refuses to.

**[Live demo](https://falsify-ten.vercel.app)** · [Eval table](https://falsify-ten.vercel.app) · [Research notes](https://falsify-ten.vercel.app/notes) · [Try it live](https://falsify-ten.vercel.app/try) · [Owner view](https://falsify-ten.vercel.app/admin)

The demo pages compute nothing: they render one frozen JSON file exported from Postgres. Only *Try it live* calls a server. The owner view needs a token, typed into a password field and sent as a bearer header, never placed in a URL.

---

## Three results

**1. Survivorship bias was two thirds of the momentum return.**
12-1 momentum, 984 invested days, Oct 2022 to Oct 2026. The only difference between the two runs is which companies the universe can see.

| | today's S&P 500 | point-in-time | gap |
|---|---|---|---|
| Total return | 79.8% | 24.9% | **+54.9 pts** |
| CAGR | 16.2% | 5.9% | +10.4 pts |
| Sharpe | 0.66 | 0.35 | **+0.31** |
| Annualised vol | 29.4% | 27.2% | +2.2 pts |
| Max drawdown | −34.1% | −32.8% | −1.3 pts |

Volatility and drawdown barely move. The invisible companies were not adding risk, they were removing return that was never earned. **A second code path, through the agent's own tools, independently measures +0.28.**

Point-in-time membership is normally a paid dataset. This reconstructs it from the **git history of a public CSV**: 194 dated snapshots back to 2012, zero API calls.

**2. The bias changes sign by strategy type, and nets to nothing.**
Across the six anomalies the gap is −0.008. It is not small, it is *cancelling*: momentum +0.28, low volatility −0.27, idiosyncratic volatility −0.26. A list of today's survivors is a list of past winners, so survivorship flatters anything that buys winners and punishes anything defensive. On long-term reversal it **flips the sign**: +0.08 on today's list, −0.26 point-in-time. Reporting one suite-level average would have hidden both halves.

**3. The eval suite passes nothing, and says why.**
Six pre-registered anomalies, hashed before the runs, scored on 1,255 trading days including the 2022 bear market. **0 pass, 0 partial, 5 fail, 1 fail-on-sign-flip, on both universes.**

| anomaly | oriented SR today / PIT | deflated today / PIT | verdict |
|---|---|---|---|
| `momentum_12_1` | +0.62 / +0.35 | 0.50 / 0.30 | fail, deflation |
| `short_term_reversal` | −0.20 / −0.15 | | fail, sign |
| `long_term_reversal` | +0.08 / **−0.26** | 0.17 / 0.31 | fail, sign flips with survivorship |
| `low_volatility` | −0.83 / −0.56 | | fail, sign |
| `idiosyncratic_volatility` | −0.73 / −0.48 | | fail, sign |
| `fifty_two_week_high` | −0.47 / −0.39 | | fail, sign |

Momentum has the only correct sign and still misses the 0.95 deflation gate. The five years were bought specifically to test whether the sign failures were one bull-market regime; the bear market is in the sample and they failed anyway.

> These are results about this window and this implementation (monthly rebalance, price returns, market-only residual), not refutations of the papers.

---

## Scale and verification

| | |
|---|---|
| Panel | 809,102 daily bars, 714 tickers, 2021-10-04 to 2026-10-02 |
| Tests | **620 passing**, 0 failing (3 need ingested SPY prices) |
| Injection audits | **19/19** and **26/26** caught, re-runnable, both in CI |
| Total LLM spend | **$0.24** across every agent run in the project |
| Hosting | $0: Vercel static, Render free, Neon free |

---

## The engine computes; the model decides

Every number comes from deterministic, tested code. The model plans experiments and interprets verified output. It never computes, and that is enforced rather than claimed:

| # | Mechanism | What it stops |
|---|---|---|
| 1 | Closed tool menu: a dict of pre-bound callables | A model-supplied string reaching `getattr`. No eval, no exec |
| 2 | Arguments validated before dispatch | Unknown keys silently ignored |
| 3 | Results cross as digests under a 2,000-byte cap | Payloads reaching the context at all |
| 4 | **Numeric provenance**: every numeral in a note must trace to a tool result | Invented figures. Percent, rounding, days-to-years and sign are permitted transforms; differences, ratios and sums are not, because those are arithmetic |
| 5 | **Comparisons are pipeline fields**, not model judgements | `0.9688 < 0.95`. `analyze_results` returns `clears_confirmation_gate`; the model reports it |
| 6 | **Pre-registration hashed** before the runs | Editing an expectation after seeing a result |

**The honest limit, stated in provenance's own docstring: it checks numbers, not claims.** A note where every figure traces and the conclusion is wrong passes completely, and the first real run did exactly that. Mechanism 5 exists because a later run traced all 16 of its numerals and still wrote that 0.9688 was "below the 0.95 bar".

The reader-facing verdict (*supported* / *not confirmed* / *contradicted*) is a **computed property**, never a stored field, so an edited database row cannot assert one. So is `publishable`, which requires provenance to pass, the run to finish on its own, and the analysis to actually have been called. Each of those three has been violated by a real run here.

<details>
<summary><b>Lookahead lives in exactly one function</b></summary>

Features use data up to and including `t`. The shift to trading time happens once, in `backtest/engine.py`, pairing each date's weights with that date's forward return. That is the only `shift(-1)` on the P&L path. Two others exist in `data/quality.py`, looking one bar ahead to measure a coverage gap and a price level shift, and neither reaches a return series.

**Correctness is established by controls, not inspection.** A positive control (perfect-foresight signal must produce an enormous Sharpe) proves the engine can express a profitable strategy at all. A negative control (buying the most recent winner on a strictly alternating series must lose money) catches the off-by-one: injecting it moves that strategy from **−88% to +573%**, which reading a Sharpe ratio would never reveal.
</details>

<details>
<summary><b>The splice gate is point-in-time, and was not always</b></summary>

A vendor ticker can be reassigned to a different company, splicing two securities end to end. Until Module 7 the gate deciding this looked at the whole panel at once, so a splice detected in 2026 removed that ticker from a 2025 backtest: future information shaping the universe.

`truncate_suspect_tickers` keeps a ticker's history up to its first defect and drops only what follows. **The property is a test, not a claim:** gating the panel as it stood on any date `T` equals the full-panel gate cut at `T`. A control test shows the old gate failing that same property. `--whole-sample-gate` reproduces the original figures.

It runs inside `agent/tools.fetch_data`, so the agent and the eval suite both get it. Before Module 7 that path had **no splice gate at all**.

A second check trims a **ragged panel end**. One ticker ingested later than the rest left trailing dates holding ~29 names, and a decile sort over 29 names is noise traded as a strategy. It raised no error: the days looked like ordinary trading days. Only the tail is trimmed; interior holes stay visible for the coverage checks.
</details>

<details>
<summary><b>Direction is the load-bearing field, and getting it wrong was a real bug</b></summary>

`run_backtest` used to go long the top bucket unconditionally. Four of the six anomalies predict the **bottom** bucket wins, so a correct effect printed a negative Sharpe, and deflation was then computed on that negative series. **A genuine low-volatility or reversal effect could never have cleared the gate, and a wrong-way spread could.** The eval suite was unaffected, because it orients by registered direction; the agent path was not.

`run_backtest` now takes a `prediction` (`top_beats_bottom` or `bottom_beats_top`), declared before the result exists. The signal is negated before ranking, so the portfolio is long the predicted winners, costs follow real turnover, and `runs_as_predicted` is the sign of that strategy's Sharpe. Flipping the prediction flips the Sharpe and leaves buckets unchanged: that is a test. "The declared prediction is recorded but never built" is an injection.
</details>

<details>
<summary><b>The scoring rule: sign, then deflation, then multiplicity</b></summary>

`eval/registry.py` pins the feature, expected direction, published Sharpe and history required. `registry_digest()` hashes exactly those fields and every stored verdict carries it, so *"we did not edit the expectation after seeing the result"* is a string comparison. Prose is deliberately excluded: a typo fixed in a citation must not invalidate stored results, or nobody would fix one.

| key | citation | feature | dir | published SR | needs |
|---|---|---|---|---|---|
| `momentum_12_1` | Jegadeesh & Titman (1993) | `mom_12_1` | +1 | 0.50 | 252d |
| `short_term_reversal` | Jegadeesh (1990) | `ret_21d` | −1 | 0.35 | 21d |
| `long_term_reversal` | De Bondt & Thaler (1985) | `rev_36_12` | −1 | 0.20 | 756d |
| `low_volatility` | Ang, Hodrick, Xing & Zhang (2006) | `vol_63d` | −1 | 0.78 | 63d |
| `idiosyncratic_volatility` | Ang et al. (2006) | `ivol_63d` | −1 | 0.60 | 63d |
| `fifty_two_week_high` | George & Hwang (2004) | `pct_52w_high` | +1 | 0.55 | 252d |

- **Magnitude is reported, never gated, except downward.** A spread above 3x the published reference downgrades a pass to partial: on a short sample a number that large is likelier a defect than a discovery, and a suite that cannot be embarrassed by its own best result is not measuring anything.
- **Four outcomes, not three.** `insufficient_data` is separate from `fail`: an anomaly needing three years of history on a two-year panel has not been refuted, it has not been tested. Untestable anomalies are excluded from the Benjamini-Hochberg correction rather than counted as nulls, because padding the denominator flatters the survivors.
- **Published Sharpes are reference levels, not targets.** Each records its source, and each anomaly records the known gap between this implementation and the published one. Failing for a reason already known is a different finding from failing on merit.
- No API key and no spend: the suite runs the pipeline, not the model. A model in the loop would make a failing row unattributable between the data and the model's choices.
</details>

<details>
<summary><b>Point-in-time universes: two decisions, both silently wrong the other way</b></summary>

`fetch_data(universe="current" | "point_in_time")`. Before Module 6 every agent run loaded today's constituents and was survivorship-inflated, while `stats/survivorship.py`, the module that measures exactly that, was reachable only from a script. The headline finding and the headline artifact did not touch. They do now.

- **The mask restricts what may be HELD, never what features may SEE.** A company that joined in March had a price history in February, and its 12-month momentum on the day it joined is real and knowable. Filtering the price frame to member-days would leave every entrant unscored for a year: a lookahead bug in reverse.
- **It is applied immediately BEFORE ranking.** Bucket edges must come from the names investable that day. Filtering after the sort leaves deciles defined by a universe the strategy could not trade, and the output still looks exactly like a backtest.

`point_in_time` **refuses to run on a single snapshot date**, which is the state a fresh ingest leaves behind. Unchanging membership is today's membership: it reproduces the biased result exactly, with full coverage, no gap, and an audit measuring zero. It is the failure mode that fails by looking healthy.

**Membership resolution is quoted, not claimed away.** The join is backward-only, so a removal is dated to the next snapshot, never to the day it happened. Error on any date equals local snapshot spacing: 137 snapshots over the priced window, **median 5 days, worst case 444** (no commits exist between 2021-10-06 and 2022-12-24, so 28 removals share that end date; a second 204-day gap does the same to 11 more). Re-running the backfill confirmed the commits do not exist: **a source limit, not an ingest bug.** Over a gap, names that left the index are still held, and index removals are disproportionately fallers.
</details>

---

## Correctness by injection, not by assertion

Every suite from Module 3 on was validated by injecting the bug it claims to catch, then confirming the suite goes red. Both audits ship and both run in CI.

```bash
python scripts/inject_module6.py    # 19/19 caught
python scripts/inject_module7.py    # 26/26 caught
python scripts/validate_stats.py    # 13/13: the statistics re-derived by simulation
```

`validate_stats.py` is the one worth reading: it re-derives each statistic empirically **without using the formula under test**. E[max SR] against Monte Carlo at four trial counts, PSR calibration at four quantiles, Benjamini-Hochberg and Holm brute-forced 20,000/20,000, FWER control at 4.6% against alpha 0.05.

Four lessons that cost something real:

- **A check that refuses to run beats a check that warns.** Two tickers silently failed to ingest and were missing for a week. The coverage gate now exits non-zero.
- **A test that has never failed has not been shown to work.** One injection passed while changing nothing on the input the test supplied, so the rule was undefended and the suite stayed green.
- **A fix you verify can move your own headline.** Correcting the splice-gate lookahead cut survivorship from +0.22 to +0.10. Report the smaller, correct number; the five-year panel later took it to +0.31.
- **The model learns thresholds from every text it reads, including your own tool docs.** A note called 0.579 "barely above the 0.5 coin-flip threshold". The real gate is 0.95, and the model had learned 0.5 from the prompt, a units note, a CLI printout and the renderer. All four now state both.

---

## Known limitations

Disclosed, quantified, and on the live site's method page:

| Limitation | Detail |
|---|---|
| Membership blind spot | 444 days, 2021-10-06 to 2022-12-24. 28 removals dated late. Source has no commits there |
| Dropped-name coverage | 162 of 278 priced. The other 116 were acquired before the window, so the measured gap is a **lower bound** |
| Delisting returns | Restoring a dropped name is not capturing its final move to a takeout price. Shumway (1997) |
| Prices | Split-adjusted only, never dividend-adjusted. Every return is a **price return**, so the Ken French UMD comparison is not like-for-like |
| `TRIAL_VARIANCE` | 0.0009 is **assumed, not substituted**. The suite prints it beside its own measurement (0.00104 / 0.00046) rather than replacing a labelled assumption with a six-point estimate |
| Reused tickers | FB→META, ABC→COR, plus COHR, ECHO, BNY are cut from **both** runs, so the comparison stays like-for-like. The fix is re-keying |
| Five years is one sample | 1,255 days, one index, monthly rebalance. A result about this window, not a refutation |
| Staleness guard | The export refuses an edited registry but cannot tell the *panel* changed under a stored row. Stamping the panel end date on each verdict is the open fix |
| Live runs | See ~3.25 years, not five: a 512 MB server peaked near 700 MB on five. Published results use all five, and the page says so |
| No benchmark tool | Long-only market beta cannot be separated out, which is why the long/short spread is the test |
| Minor | Turnover uses the un-halved convention and ignores weight drift; risk-free rate is a scalar |

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # add POLYGON_API_KEY
docker compose up -d        # TimescaleDB, schema applies on first run

pytest -q                               # 620 passing
python scripts/run_ingest.py AAPL MSFT  # smoke test
```

Full ingest: `python scripts/run_ingest.py` (503 tickers) then `python scripts/run_ingest.py SPY`. The free Polygon tier (5 req/min) takes ~100 minutes for 2 years; `POLYGON_RPM=100` on Starter gives ~5 minutes for 5. Ingest is idempotent and resumes after an interrupt.

```bash
python scripts/run_agent.py "do low-volatility stocks outperform?"   # one hypothesis, end to end
python scripts/show_notes.py                                         # read notes back, no API key, no spend
python scripts/run_evals.py --compare                                # both universes, gap per anomaly
python scripts/run_survivorship.py                                   # the headline audit
```

<details>
<summary><b>Layout</b></summary>

```
src/falsify/
  data/polygon_client.py    rate-limited API client with 429 backoff
  data/universe.py          S&P 500 constituents (CSV primary, HTML fallback)
  data/ingest.py            idempotent Polygon to TimescaleDB pipeline
  data/pit_universe.py      point-in-time membership from the constituents git log
  data/quality.py           splice and gap detection; the point-in-time gate
  features/library.py       composable Polars features, lookahead-safe by design
  backtest/portfolio.py     signal to weights: decile sorts, rebalancing
  backtest/engine.py        weights to daily return series
  backtest/metrics.py       Sharpe, CAGR, volatility, drawdown
  backtest/diagnostics.py   bucket staircase, half-sample Sharpe, cost sensitivity, curve
  stats/walkforward.py      walk-forward splits with no leakage across a boundary
  stats/deflated.py         PSR, E[max SR], deflated Sharpe, minimum track record
  stats/multipletest.py     Benjamini-Hochberg and Holm
  stats/survivorship.py     current-constituents vs point-in-time, measured
  agent/session.py          handle store: payloads stay server-side, capped digests
  agent/tools.py            the boundary: closed menus, validated arguments
  agent/loop.py             the conversation loop and five cost guards
  agent/provenance.py       every numeral in a note must trace to a tool result
  notes/schema.py           the Note record, the publishable rule, the verdict
  notes/store.py            notes in Postgres, failures kept as failures
  notes/render.py           Note to markdown: tables from data, prose as commentary
  eval/registry.py          the pre-registration: six anomalies, frozen and hashed
  eval/score.py             pass / partial / fail / insufficient_data, with reasons
  eval/runner.py            the suite, run through the same tools the agent uses
  eval/store.py             verdicts in Postgres, keyed by (anomaly, universe)
  live/app.py               FastAPI: the public Try-it-live endpoint and /admin
  live/limits.py            worst-case reservation, daily cap, per-visitor cap
  live/store.py             live_run rows, kept apart from research_note
  demo.py                   stored evidence to one frozen JSON, refusing anything misleading
scripts/                    ingest, evals, survivorship, membership diagnostics,
                            both injection audits, the simulation validator, export
web/                        Next.js static export; renders data/demo.json, computes nothing
tests/                      synthetic data with hand-computed expected values
db/schema.sql               daily_bars, universe_snapshot, ingest_log, research_note,
                            eval_result, live_run
```
</details>

<details>
<summary><b>Research notes and the demo export</b></summary>

A note stores **verified numbers as data** and prose as commentary beside them. The page renders tables from the data, never by parsing prose, and says which is which. That is the answer to "how do I know this is not just a language model writing plausible text?"

`scripts/show_notes.py` reads a note back in a process that shares nothing with the one that produced it and cannot make a model call. **Notes that fail are stored as failures**, and the count is a number the demo shows: deleting them is how an eval suite becomes a highlight reel.

`scripts/export_demo.py` writes `web/data/demo.json` and **refuses** rather than warns when the result could mislead: a verdict scored against an edited registry, rows held to different scoring-rule versions, a table with a missing anomaly, a survivorship figure produced with `--force`, or uncommitted code. `publishable` is recomputed, never read. The web build fails if the data file is missing rather than falling back to a sample.

```bash
python scripts/run_evals.py --compare --store
python scripts/run_survivorship.py --save web/data/survivorship.json
python scripts/export_demo.py --survivorship web/data/survivorship.json
cd web && npm ci && npm run build
```
</details>

<details>
<summary><b>Live runs: the owner pays, under a hard cap</b></summary>

A visitor types a hypothesis; a small FastAPI server runs the same agent loop against a hosted copy of the price data and returns the note to that visitor only.

- **The cap is never crossed, rather than noticed afterwards.** Each run reserves its worst case ($0.45) the moment it is queued, and starts only if that reservation still fits under the $2 daily cap. The reservation is sound only because a per-run token budget bounds what one run can cost: a run count alone would not, since a run that loops to its budget costs several times a normal one.
- Also: 25 runs a day, 10 per visitor, 5 queued at once.
- **Visitors see only their own run**, readable by random id. Live runs live in `live_run`, never `research_note`, so the export cannot put a stranger's text on the public page. IPs are never stored, only salted hashes.
- **The privacy boundary is attacked by four of the 26 injections:** a refused run reaching the public notes, hiding a run failing to remove it, a public note carrying the visitor id and name, and a NaN blanking the page. All caught.
- **The owner view** at [/admin](https://falsify-ten.vercel.app/admin) shows every question, the optional name, a hashed visitor id, the cost and the full note, plus a control to hide a run so it can never be exported. Token required.
- Free hosting spins down, so the page wakes the server and retries for up to two minutes instead of erroring. Tested by bringing the server up mid-request.

```bash
uvicorn falsify.live.app:app --reload   # DATABASE_URL, ANTHROPIC_API_KEY, ADMIN_TOKEN, VISITOR_SALT
```
</details>

---

**Stack.** Python, Polars (not pandas), psycopg3, FastAPI, TimescaleDB, Anthropic SDK direct (no agent framework), TypeScript, Next.js. NumPy and SciPy are deliberately **not** dependencies: the statistics layer uses stdlib `statistics.NormalDist`.
