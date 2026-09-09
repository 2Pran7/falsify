"""Minimal, rate-limited Polygon.io client.

One method covers the ingest requirement: daily aggregates for a ticker over a
date range. One call covers 5 years of bars (limit=50000 >> ~1260 trading
days), so the whole S&P 500 is ~503 calls.

ADJUSTMENT. `adjusted=true` adjusts for SPLITS ONLY — the documented behaviour
is "whether or not the results are adjusted for splits"; dividends are not
mentioned and not applied. Every price in `daily_bars` is therefore a
split-adjusted PRICE and every return derived from it a PRICE return.

Two consequences for the write-up:
  1. The long leg is understated by roughly the dividend yield, ~1.2-1.5%/yr
     on the S&P 500.
  2. Ken French's UMD series, the ground truth for momentum, is built on TOTAL
     returns, so the correlation target must account for the mismatch.

A dollar-neutral long/short book cancels most of the effect, since both legs
lack dividends and yields are similar across momentum deciles. Not exact: value
and low-volatility deciles are systematically higher-yielding, so those
anomalies are affected more than momentum. Total returns need the dividend
endpoint (paid tier); deferred to Module 6 and disclosed until then.
"""
from __future__ import annotations

import time

import httpx

from falsify.config import settings

BASE = "https://api.polygon.io"


class PolygonClient:
    def __init__(self, api_key: str | None = None, rpm: int | None = None):
        self.api_key = api_key or settings.polygon_api_key
        self.min_interval = 60.0 / (rpm or settings.polygon_rpm)
        self._last_call = 0.0
        self._http = httpx.Client(timeout=30.0)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def _get(self, path: str, **params) -> dict:
        self._throttle()
        params["apiKey"] = self.api_key
        for attempt in range(3):
            resp = self._http.get(f"{BASE}{path}", params=params)
            if resp.status_code == 429:  # rate limited — back off and retry
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    def daily_bars(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """SPLIT-adjusted daily OHLCV bars for one ticker. Dates are 'YYYY-MM-DD'.

        Not dividend-adjusted: see the module docstring.
        """
        data = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
            adjusted="true",
            sort="asc",
            limit=50000,
        )
        return data.get("results", []) or []

    def close(self) -> None:
        self._http.close()
