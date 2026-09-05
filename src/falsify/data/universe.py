"""S&P 500 universe.

Source: the `datasets/s-and-p-500-companies` CSV on GitHub — a maintained,
machine-readable constituents list. Chosen over scraping Wikipedia HTML
because scraping breaks whenever the page layout or served variant changes
(it did, on day one). Wikipedia scrape kept as a fallback.

HONEST LIMITATION (document, don't hide): this is the CURRENT constituent
list, not point-in-time historical membership. Backtests on today's members
overstate returns (survivorship bias) — dead/dropped companies are missing.
Module 3's audit quantifies this; the methodology post discloses it.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re

import httpx
import polars as pl

CSV_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "falsify/0.1 research"}


def _from_csv() -> list[str]:
    resp = httpx.get(CSV_URL, timeout=30.0, follow_redirects=True, headers=HEADERS)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    return [row["Symbol"].strip() for row in reader if row.get("Symbol", "").strip()]


def _from_wikipedia() -> list[str]:
    html = httpx.get(WIKI_URL, timeout=30.0, follow_redirects=True, headers=HEADERS).text
    if 'id="constituents"' not in html:
        raise RuntimeError("Wikipedia page variant lacks constituents anchor")
    table = html.split('id="constituents"')[1].split("</table>")[0]
    rows = re.findall(r"<tr>(.*?)</tr>", table, flags=re.S)
    tickers = []
    for row in rows:
        cells = re.findall(r"<td.*?>(.*?)</td>", row, flags=re.S)
        if cells:
            first = re.sub(r"<.*?>", "", cells[0]).strip()
            if first:
                tickers.append(first)
    return tickers


def fetch_sp500_tickers() -> list[str]:
    """Current S&P 500 tickers. CSV primary, Wikipedia fallback."""
    try:
        tickers = _from_csv()
    except Exception as csv_err:  # noqa: BLE001
        print(f"CSV source failed ({csv_err}); falling back to Wikipedia scrape")
        tickers = _from_wikipedia()
    tickers = sorted(set(tickers))
    if len(tickers) < 400:
        raise RuntimeError(f"Only {len(tickers)} tickers parsed — source is broken")
    return tickers


def snapshot_frame(tickers: list[str]) -> pl.DataFrame:
    """Universe snapshot rows for the universe_snapshot table (as_of = today)."""
    today = dt.date.today()
    return pl.DataFrame(
        {
            "ticker": tickers,
            "index_name": ["SP500"] * len(tickers),
            "as_of": [today] * len(tickers),
        }
    )