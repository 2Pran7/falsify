"""Minimal, rate-limited Polygon.io client.

A single method covers the ingest requirement: daily aggregates for a ticker
over a date range, split/dividend adjusted. One API call covers 5 years of daily
bars (limit=50000 >> ~1260 trading days), so the whole S&P 500 is ~503 calls.
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
        """Adjusted daily OHLCV bars for one ticker. Dates are 'YYYY-MM-DD'."""
        data = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
            adjusted="true",
            sort="asc",
            limit=50000,
        )
        return data.get("results", []) or []

    def close(self) -> None:
        self._http.close()
