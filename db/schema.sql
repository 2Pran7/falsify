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
