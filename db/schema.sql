-- falsify schema — Module 1
-- Runs automatically on first `docker compose up` via docker-entrypoint-initdb.d

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Daily OHLCV bars. Prices are SPLIT/DIVIDEND ADJUSTED at ingest time
-- (Polygon adjusted=true). This is the canonical price table.
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

-- Universe snapshot: which tickers we consider part of the S&P 500 and WHEN
-- we recorded that membership. This is NOT point-in-time historical membership
-- (that data isn't on cheap Polygon tiers) — it's a snapshot with a recorded
-- as_of date so the survivorship-bias audit in Module 3 has something to work with.
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
