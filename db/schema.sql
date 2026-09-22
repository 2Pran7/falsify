-- falsify schema — Module 1
-- Runs automatically on first `docker compose up` via docker-entrypoint-initdb.d

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Daily OHLCV bars. Prices are SPLIT-ADJUSTED ONLY at ingest time.
-- Polygon's adjusted=true applies split adjustments and NOT dividend
-- adjustments, so everything derived from these closes is a PRICE return,
-- never a total return. See src/falsify/data/polygon_client.py for the full
-- note and the consequences for the Ken French UMD comparison.
-- This is the canonical price table.
CREATE TABLE IF NOT EXISTS daily_bars (
    ticker      TEXT             NOT NULL,
    ts          DATE             NOT NULL,   -- trading date
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT           NOT NULL,
    vwap        DOUBLE PRECISION,
    n_trades    BIGINT,
    PRIMARY KEY (ticker, ts)
);

SELECT create_hypertable('daily_bars', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_daily_bars_ticker ON daily_bars (ticker, ts DESC);

-- Universe snapshot: which tickers were in the S&P 500, and as of when.
--
-- Two populations of rows share this table, distinguished only by as_of:
--   * run_ingest.py writes ONE snapshot dated the day the ingest ran. On its
--     own that is useless for a survivorship audit: a single as_of cannot show
--     anyone leaving the index.
--   * build_pit_universe.py backfills ~190 dated snapshots reconstructed from
--     the git history of the maintained constituents CSV, reaching to 2012.
--     Those are what make the Module 3 audit runnable.
-- as_of is the date membership was RECORDED, not the effective date of the
-- index change; see src/falsify/data/pit_universe.py.
CREATE TABLE IF NOT EXISTS universe_snapshot (
    ticker      TEXT NOT NULL,
    index_name  TEXT NOT NULL DEFAULT 'SP500',
    as_of       DATE NOT NULL,
    PRIMARY KEY (ticker, index_name, as_of)
);

-- Ingest bookkeeping: lets re-runs skip completed tickers (idempotent ingest).
CREATE TABLE IF NOT EXISTS ingest_log (
    ticker      TEXT NOT NULL,
    from_date   DATE NOT NULL,
    to_date     DATE NOT NULL,
    n_rows      INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, from_date, to_date)
);

-- Research notes — Module 5.
--
-- A run's note used to exist in two places, neither durable: the model's final
-- turn, printed to a terminal, and a JSON transcript under .cache/runs/, which
-- is gitignored. Module 7 must serve a note without a Claude call and without
-- a rerun, so the note lives here.
--
-- WHAT IS AND IS NOT A COLUMN. metrics, statistics, provenance and run_meta go
-- in as JSONB: the pipeline's output keys change (prob_beats_best_of_n_trials
-- was renamed after the first real run put it in the wrong column) and a rename
-- should not need a migration on a table holding six rows. What IS a column is
-- anything the web page filters on.
--
-- publishable is DERIVED — provenance passed AND the run completed AND
-- analyze_results was called — and is stored only so the page can filter in
-- SQL. src/falsify/notes/schema.py recomputes it on read from the record
-- itself, so a row edited here to say true does not come back publishable.
-- The column is an index, never the source of truth.
--
-- UNPUBLISHABLE NOTES ARE KEPT. They are the eval suite's honest failures, and
-- the count of them is a number the demo shows. Deleting them is how an eval
-- suite becomes a highlight reel.
--
-- Mirrored in src/falsify/notes/store.SCHEMA_SQL so a test or a Module 6
-- backfill can create the table without the Docker init path. Both are
-- IF NOT EXISTS; keep them in step.
CREATE TABLE IF NOT EXISTS research_note (
    note_id               UUID        PRIMARY KEY,
    -- Module 6's anomaly slug. NULL for an ad-hoc run.
    eval_key              TEXT,
    created_at            TIMESTAMPTZ NOT NULL,
    schema_version        INTEGER     NOT NULL,
    -- The question as it was asked. Verbatim: a note whose hypothesis has been
    -- paraphrased cannot be audited against the run that produced it.
    hypothesis            TEXT        NOT NULL,
    -- The model's commentary. The least reliable part of the run, stored
    -- alongside the numbers rather than in place of them.
    prose                 TEXT        NOT NULL,
    publishable           BOOLEAN     NOT NULL,
    unpublishable_reasons JSONB       NOT NULL,
    -- Per backtest: variant, engine metrics, and the statistics recomputed at
    -- the run's FINAL trial count.
    backtests             JSONB       NOT NULL,
    provenance            JSONB       NOT NULL,
    run_meta              JSONB       NOT NULL,
    -- TRIAL_VARIANCE above all. An assumption that is not stored with the
    -- result becomes invisible the moment the result is quoted.
    assumptions           JSONB       NOT NULL
);

-- Partial, so ad-hoc notes accumulate as history while an eval case has exactly
-- one current row. Module 6 re-runs every anomaly and overwrites.
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_note_eval_key
    ON research_note (eval_key) WHERE eval_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_research_note_publishable
    ON research_note (publishable, created_at DESC);

-- Eval verdicts — Module 6.
--
-- THE KEY IS THE DESIGN DECISION. A verdict is not a property of an anomaly,
-- it is a property of an anomaly RUN ON A UNIVERSE. Momentum scores
-- differently on today's constituent list than on point-in-time membership —
-- that difference is the project's headline finding — so both rows must
-- coexist. Keying on eval_key alone would make the second run silently
-- overwrite the first, and the survivorship comparison would come back as a
-- single row and look perfectly fine.
--
-- registry_sha and scoring_rule_version sit on EVERY row. Together they say
-- "this prediction, judged by this rule". Neither alone would reveal that two
-- rows in the same table had been held to different standards, which is
-- exactly what happens when the registry is edited between runs.
-- scripts/run_evals.py --check exits non-zero rather than quoting a table
-- containing a stale sha.
--
-- reasons is stored, not just the verdict. Module 7 renders the reasons; a
-- bare count of passes is a number nobody can act on, and it is also how an
-- eval suite turns into a highlight reel.
--
-- WARNING, learned by injection while building this module and worth more than
-- the table itself: CREATE TABLE IF NOT EXISTS IS A NO-OP AGAINST AN EXISTING
-- TABLE. Any edit to this file after the first `docker compose up` is silently
-- ignored on that database, forever, while every read and write keeps
-- succeeding. Dropping `universe` from the primary key below changed this
-- string and nothing else, and the injected bug was the one the suite missed
-- on its first pass. src/falsify/eval/store.primary_key_columns() now reads
-- the key from pg_index ON THE LIVE SERVER, and a test asserts it — the
-- invariant is checked against the database, not against the string that was
-- supposed to create it.
--
-- Mirrored in src/falsify/eval/store.SCHEMA_SQL; keep the two in step.
CREATE TABLE IF NOT EXISTS eval_result (
    eval_key             TEXT        NOT NULL,
    -- 'current' or 'point_in_time'. Half the primary key; see above.
    universe             TEXT        NOT NULL,
    run_at               TIMESTAMPTZ NOT NULL,
    -- sha256 of the PREDICTED registry fields only. Prose is excluded, so
    -- fixing a citation typo does not invalidate stored verdicts.
    registry_sha         TEXT        NOT NULL,
    scoring_rule_version INTEGER     NOT NULL,
    -- pass | partial | fail | insufficient_data. Four, not three:
    -- insufficient_data is not a refutation, and collapsing the two would let
    -- a short sample manufacture rejections.
    verdict              TEXT        NOT NULL,
    reasons              JSONB       NOT NULL,
    -- +1 if the paper predicts the top bucket wins, -1 the bottom. Three of
    -- the six predict -1, so this is the load-bearing field.
    direction            INTEGER     NOT NULL,
    -- Signed as measured, so a reader can reconcile it against the backtest.
    realised_sharpe      DOUBLE PRECISION,
    -- realised * direction. This is what the verdict was computed from.
    oriented_sharpe      DOUBLE PRECISION,
    -- Reference level, never a target.
    published_sharpe     DOUBLE PRECISION NOT NULL,
    p_value              DOUBLE PRECISION,
    p_value_adjusted     DOUBLE PRECISION,
    deflated_psr         DOUBLE PRECISION,
    n_invested_days      INTEGER     NOT NULL,
    history_days         INTEGER     NOT NULL,
    min_history_days     INTEGER     NOT NULL,
    n_trials             INTEGER     NOT NULL,
    -- The known gap between this implementation and the published one. An
    -- anomaly that fails for a reason already known is a different finding
    -- from one that fails on its merits.
    caveat               TEXT        NOT NULL,
    detail               JSONB       NOT NULL,
    PRIMARY KEY (eval_key, universe)
);

CREATE INDEX IF NOT EXISTS idx_eval_result_verdict
    ON eval_result (verdict, universe);
