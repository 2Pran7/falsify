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
| Verdict and diagnostics | Supported / not confirmed / contradicted as a computed property; bucket staircase, cost sensitivity, equity curve | Complete |
| Research notes | Verified notes persisted to Postgres, publishable rule, markdown rendering | Complete |
| Eval suite | Six published anomalies, pre-registered and hashed; four-rule scoring; universe argument | Complete. **5-year results: 0 of 6 survive, on both universes** |
| Demo | Frozen verdicts and notes on a static Next.js page, plus capped live runs (FastAPI on Render, Neon) | Complete. [falsify-ten.vercel.app](https://falsify-ten.vercel.app) |

Full suite: **620 passing** on a populated database (three of those need
ingested SPY prices and skip without them). Every new suite from Module 3 onward was validated by injecting the
bug it claims to catch: `scripts/inject_module6.py` catches 19 of 19 and
`scripts/inject_module7.py` 23 of 23, and CI runs both on every push.

**Live demo: [falsify-ten.vercel.app](https://falsify-ten.vercel.app)** — the
frozen verdicts, the research notes including the refused ones, and a capped
*Try it live* page that runs the real agent loop on a hypothesis you type.

**The headline result.** Reconstructing point-in-time S&P 500 membership from
the git history of the constituents CSV (194 dated snapshots back to 2012, no
paid membership data) shows that on five years of prices, **survivorship bias
accounts for more than two thirds of the headline momentum return**: 79.8%
against 24.9% over a common 984-day invested window (Oct 2022 to Oct 2026),
CAGR 16.2% against 5.9%, Sharpe 0.66 against 0.35. Volatility and max drawdown
barely moved (29.4% vs 27.2%, -34.1% vs -32.8%): the invisible companies were
not adding risk, they were removing return that was never earned.

**Two independent code paths agree.** The eval suite measures the same
strategy through the agent's tools and gets 0.62 against 0.35 (+0.28) over the
same window, against +0.31 from `scripts/run_survivorship.py`.

**The bias is not one-directional.** Across the six anomalies it nets to about
zero: it inflates strategies that buy past winners (momentum +0.28) and
penalises the defensive ones (low volatility -0.27, idiosyncratic volatility
-0.26), because a list of today's survivors is a list of past winners. On
long-term reversal it **flips the sign**: +0.08 on today's list, -0.26 on
point-in-time membership.

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

**The splice gate is point-in-time.** A vendor ticker can be reassigned to a
different company, and a naive series then splices two securities end to end.
Until Module 7 the gate that caught this decided over the whole panel at once,
so a splice detected in 2026 removed that ticker from a 2025 backtest too:
future information shaping the universe. `truncate_suspect_tickers` now keeps
a ticker's genuine history up to its first defect and drops only what follows,
and the property is a test rather than a claim: gating the panel as it stood
on any date T gives exactly what the full-panel gate keeps up to T. A control
test shows the old gate failing the same property. It is applied inside
`agent/tools.fetch_data`, so the agent and the eval suite both see it; before
Module 7 that path had no splice gate at all. `--whole-sample-gate` on the two
Module 3 scripts reproduces the original figures. A second check trims a
**ragged panel end**: one ticker ingested later than the rest leaves trailing
dates with a handful of names each, and a decile sort over four names is noise
traded as a strategy. Only the tail is trimmed; interior holes are left for the
coverage checks to report.

**The model decides, and that is enforced rather than asserted.** Six
mechanisms, in increasing order of strength: the tool menu is closed (a dict of
pre-bound callables, never a `getattr` on a model-supplied string); arguments
are validated before dispatch and unknown keys are rejected rather than
ignored; results cross to the model as summaries under a 2,000-byte enforced
cap, never as payloads; every numeral in a research note must appear in, or
be a permitted transform of, a number the model was actually shown
(`agent/provenance.py`); a note is publishable only if provenance passed, the
run completed and the analysis was actually called; and the predictions in the
eval suite are hashed before the runs. Percent, rounding, days-to-years and sign
are permitted transforms; differences, ratios and sums are not, because those
are arithmetic.

The provenance mechanism is what makes the claim falsifiable, and its limitation
is stated in its own docstring: **provenance checks numbers, not claims.** A
note in which every figure is traceable and the conclusion is wrong passes
completely, and the first real run did exactly that.

**It does not check comparisons either, so comparisons became pipeline fields.**
A later run traced all sixteen of its numerals and still wrote that a deflation
probability of 0.9688 was "below the 0.95 bar". Every figure was real and the
inequality was backwards. The fix was not a wider checker: `analyze_results` now
returns `confirmation_gate` and `clears_confirmation_gate`, computed by the
pipeline, and the system prompt instructs the model to report that field rather
than compare the numbers itself. An injection that inverts the gate verdict is
part of the audit.

**The same applies to the one-line answer.** The verdict a reader sees —
*supported*, *not confirmed*, *contradicted* — is a computed property of the
note like `publishable`, never a stored field, so an edited row cannot claim a
different one. The headline backtest it describes is chosen by rule (the first
analysed long/short run with a declared prediction), not by the model. The
bucket returns and their Spearman "staircase" score, the Sharpe of each half,
the Sharpe at 0, 10 and 25 bps, and the equity curve are all computed in
`backtest/diagnostics.py` and rendered as charts; the model sees only rounded
digests and never the curve. It writes the commentary, and the page says which
is which.

**An agent that narrates a step has not taken it.** One live run ended with the
text "Now I'll run the rigour analysis" and no call to `analyze_results`; the
publishable rule refused the note, correctly. The loop now sends exactly one
reminder when a model tries to finish with unanalysed backtests, and if it stops
anyway the refusal stands. Two injections cover it: the reminder never being
sent, and the reminder repeating until the model complies.

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
pytest -q                                   # full suite; 3 tests need ingested SPY prices
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
- **Five years is still one sample.** The eval suite now runs on 1,255 trading
  days including the 2022 bear market, and none of the six anomalies survives
  deflation and multiple-testing correction on either universe. That is a
  result about this window and this implementation (monthly rebalance, price
  returns, a market-only residual), not a refutation of the papers.
- **Membership has a 444-day blind spot inside the panel.** The constituents
  file has no commits between 2021-10-06 and 2022-12-24, so 28 removals in that
  stretch are dated to the snapshot that closed it. Acquired names stop having
  prices and drop out regardless; the real error is demoted names that kept
  trading, held for up to 444 days too long. For 12-1 momentum it overlaps only
  the first ~2 months of trading, because the feature needs a year of history.
- **Reused tickers cost whole companies.** The splice gate keeps a ticker's
  earliest occupant and drops what follows, so renames where the new ticker
  previously belonged to someone else (FB to META, ABC to COR) remove the
  renamed company from both runs: Meta after mid-2022, Cencora, Coherent,
  EchoStar, and BNY after a vendor data gap. Both runs lose them equally, so the
  gap is compared like for like. The fix is re-keying a reused ticker as a new
  instrument, with membership mapped by date.
- **Live runs see three years, not five.** The hosted server has 512 MB, and a
  five-year, three-idea run peaked near 700 MB. Live runs read three columns of
  the most recent ~3.25 years (`FALSIFY_PANEL_START`); the published results
  are computed offline on the full history. The Try page says so.
- **The splice gate truncates rather than re-keys.** The rows after a
  ticker's first defect may be a genuine successor company, and they are
  dropped rather than treated as a new instrument. That costs sample, not
  correctness: it never reaches a return, because the return across the join is
  the one that is never formed.
- **There is no benchmark tool**, so a long-only result's market beta cannot be
  separated out. A long-only Sharpe is not a test of a cross-sectional
  hypothesis; the long/short spread is.
- **The staleness guard protects the predictions, not the data.** The export
  refuses a verdict scored against an edited registry, but it cannot tell that
  the *panel* changed underneath a stored row — which happened once, when the
  site was exported before the evals were re-run on the trimmed panel. Stamping
  the panel end date on each verdict row is the open fix.
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
  backtest/diagnostics.py   bucket staircase, half-sample Sharpe, cost sensitivity, curve
  data/pit_universe.py      point-in-time membership from the constituents git log
  data/quality.py           splice and gap detection; the point-in-time gate
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
  live/app.py               FastAPI: the public Try-it-live endpoint and /admin
  live/limits.py            worst-case reservation, daily cap, per-visitor cap
  live/store.py             live_run rows, kept apart from research_note
  live/worker.py            runs the agent loop off the request thread
  demo.py                   stored evidence -> one frozen JSON, refusing anything misleading
scripts/validate_stats.py   the statistics re-derived by simulation, not unit test
scripts/run_agent.py        one hypothesis end to end; prints and stores what it cost
scripts/show_notes.py       read notes back with no API key and no spend
scripts/run_evals.py        the eval suite: --compare, --check, --store, --registry
scripts/run_survivorship.py the headline audit: current vs point-in-time, same window
scripts/build_pit_universe.py  membership snapshots recovered from the constituents git log
scripts/diagnose_membership.py  snapshot gaps, member-day clusters, membership resolution
scripts/ingest_dropped.py   prices for the names that LEFT the index
scripts/diagnose.py         attribution, concentration and best/worst-day sensitivity
scripts/inject_module6.py   the nineteen-injection audit, re-runnable
scripts/inject_module7.py   twenty-three more: the splice gate, the ragged end, the
                            declared prediction, the gate verdict, every export refusal
scripts/export_demo.py      Postgres -> web/data/demo.json for the static page
scripts/sync_live_db.py     copy the panel to the hosted database for live runs
web/                        Next.js static export; renders data/demo.json, computes nothing
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

**Direction is the load-bearing field**, and getting it wrong was a real bug
rather than a hypothetical one. `run_backtest` used to go long the top bucket
and short the bottom unconditionally, so for the four of these six that predict
the *bottom* bucket wins, a correct effect printed a negative Sharpe — and the
deflated probability was then computed on that negative series. **A genuine
low-volatility or reversal effect could never have cleared the gate, and a
wrong-way spread could.** The eval suite was unaffected, because it orients each
anomaly by its registered direction, but the agent path was not.

`run_backtest` now takes a `prediction` (`top_beats_bottom` or
`bottom_beats_top`) that must be declared before the result exists. The signal
is negated before ranking, so the portfolio is long the predicted winners,
transaction costs follow the real turnover, and `runs_as_predicted` is simply
the sign of that strategy's Sharpe. Flipping the prediction flips the Sharpe and
leaves the buckets unchanged, which is a test; "the declared prediction is
recorded but never built" is an injection. Without a sign fixed in advance, "the
spread was −0.6" is unscoreable, and the temptation is to look at the number and
then decide which way the paper said it should run.

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

### Membership resolution, and why it is quoted rather than claimed away

Point-in-time membership here is reconstructed from the commit history of a
maintained CSV: **194 dated snapshots from December 2012**, recovered by
`scripts/build_pit_universe.py` for the cost of a clone. The join is
backward-only — each date inherits the most recent snapshot at or before it —
so **a removal is dated to the next snapshot, never to the day it happened.**

That makes the error on any membership date equal to the local snapshot
spacing, and the spacing is not uniform. Over the five-year priced window there
are 137 snapshots, the **median spacing is 5 days** and **the worst case is 444
days**: the maintainers committed nothing between 2021-10-06 and 2022-12-24, so
every index removal in that stretch is recorded as happening on 2022-12-24 (28
tickers share that end date). A second gap, 204 days to 2026-03-04, does the
same to 11 more.

**This is a source limitation, not an ingest bug**, and re-running the backfill
confirmed it: the commits do not exist. So it is disclosed and bounded rather
than fixed. `scripts/diagnose_membership.py` prints the spacing, separates gaps
that overlap the priced window from gaps that do not, and identifies the
clusters a gap creates.

**The practical consequence:** over a gap, names that have left the index are
still held, and index removals are disproportionately fallers. A second error
runs the other way — a point-in-time member whose prices were never ingested
contributes nothing, thinning the leg — and `scripts/ingest_dropped.py` exists
to close that one, because unlike the first it is closeable. Neither error
cancels the other, and both are stated wherever the survivorship number is.

## Demo

`web/` is a static Next.js site that renders one file, `web/data/demo.json`, and
computes nothing: no model call, no database, no backtest. That file is written
by `scripts/export_demo.py` from what Postgres holds, and the export **refuses**
rather than warns when the result could mislead: a verdict scored against an
edited registry, rows held to different scoring-rule versions, a table with a
missing anomaly, or a survivorship figure produced with `--force`. It exports
refused notes and failing verdicts with their reasons, because the page shows
the reasons, not the tally.

```
python scripts/run_evals.py --compare --store
python scripts/run_survivorship.py --save web/data/survivorship.json
python scripts/export_demo.py --survivorship web/data/survivorship.json
cd web && npm ci && npm run build      # static site in web/out
```

The build fails if `data/demo.json` is missing rather than falling back to a
sample. Deployed on Vercel with `web` as the project root.

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

## Live runs

The public site also has a **Try it live** page. A visitor types a hypothesis;
a small FastAPI server (`src/falsify/live/`, on Render) runs the same agent loop
against a hosted copy of the price data (Neon Postgres) and returns the note to
that visitor only.

- **The owner pays, under a hard cap.** Each run reserves its worst-case cost
  ($0.45) the moment it is queued, and a run starts only if that reservation
  still fits under the $2 daily cap, so the cap is never crossed rather than
  noticed afterwards. On top: 25 runs a day in total and 10 per visitor.
- **The server is allowed to be asleep.** Free hosting spins down, so the page
  wakes the server and retries for up to two minutes instead of showing an
  error, and a free external ping keeps it warm during the day. Tested by
  bringing the server up mid-request.
- **Visitors see only their own run.** It is readable only by its random id.
  Live runs are stored in `live_run`, never `research_note`, so
  `export_demo.py` cannot put a stranger's text on the public page.
- **The owner sees everything** at `/admin` with a token: every question, the
  optional name a visitor left, a hashed visitor id (no IPs are stored), the
  cost and the full note.
- `scripts/sync_live_db.py --target <hosted URL>` copies `daily_bars` and
  `universe_snapshot` to the hosted database. Re-run it after every ingest.

```
uvicorn falsify.live.app:app --reload     # needs DATABASE_URL, ANTHROPIC_API_KEY, ADMIN_TOKEN, VISITOR_SALT
```
