"""Session 0: prove the SDK works, one tool, one round trip, under a cent.

    python scripts/hello_tools.py

Nothing from Module 4 is built on top of this. It exists to answer three
questions before any agent code is written:

    1. Does the key work and is the model reachable?
    2. Does a tool_use -> tool_result -> final answer round trip complete?
    3. What does one call actually cost?

Question 3 is the point. Every session from here on reports its spend, and you
cannot report a spend you have never measured. The usage figures printed at the
bottom come from `response.usage`, which is the ground truth, not an estimate
from string length.

The tool is deliberately trivial and deliberately REAL: it counts rows in
daily_bars. The model is never told the number. It has to ask.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import psycopg
from anthropic import Anthropic
from dotenv import load_dotenv

from falsify.config import settings

load_dotenv()

MODEL = "claude-sonnet-5"

# Pricing per million tokens, claude-sonnet-5. Update if the model changes.
USD_PER_MTOK_IN = 2.0
USD_PER_MTOK_OUT = 10.0

TOOLS = [
    {
        "name": "count_bars",
        "description": (
            "Count the daily price bars stored for one ticker, and report the "
            "date range they cover. Use this whenever you need to know how much "
            "price history is available for a ticker before reasoning about it. "
            "Returns n_rows, first_date and last_date. A ticker that was never "
            "ingested returns n_rows 0."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Uppercase US equity ticker, e.g. AAPL.",
                }
            },
            "required": ["ticker"],
        },
    }
]


def count_bars(ticker: str) -> dict:
    """The deterministic side. The model never computes this; it asks for it."""
    sql = (
        "SELECT count(*), min(ts), max(ts) FROM daily_bars WHERE ticker = %s"
    )
    with psycopg.connect(settings.db_dsn) as conn:
        n, first, last = conn.execute(sql, (ticker,)).fetchone()
    return {
        "ticker": ticker,
        "n_rows": n,
        "first_date": str(first) if first else None,
        "last_date": str(last) if last else None,
    }


def _cost(usage) -> float:
    return (
        usage.input_tokens / 1e6 * USD_PER_MTOK_IN
        + usage.output_tokens / 1e6 * USD_PER_MTOK_OUT
    )


def main() -> None:
    client = Anthropic()
    messages = [
        {
            "role": "user",
            "content": "How much price history do we have for AAPL? Answer in one sentence.",
        }
    ]

    # --- turn 1: the model should ask for the tool ---------------------------
    r1 = client.messages.create(
        model=MODEL, max_tokens=512, tools=TOOLS, messages=messages
    )
    print(f"turn 1 stop_reason: {r1.stop_reason}")

    tool_uses = [b for b in r1.content if b.type == "tool_use"]
    if not tool_uses:
        text = "".join(b.text for b in r1.content if b.type == "text")
        print("\nThe model answered WITHOUT calling the tool:\n  " + text)
        print("\nThat is a failure for this test: it means it guessed.")
        raise SystemExit(1)

    call = tool_uses[0]
    print(f"  tool requested: {call.name}({call.input})")

    # --- execute it ourselves ------------------------------------------------
    result = count_bars(**call.input)
    print(f"  pipeline returned: {result}")

    # --- turn 2: hand the result back ---------------------------------------
    messages.append({"role": "assistant", "content": r1.content})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": str(result),
                }
            ],
        }
    )
    r2 = client.messages.create(
        model=MODEL, max_tokens=512, tools=TOOLS, messages=messages
    )
    answer = "".join(b.text for b in r2.content if b.type == "text")
    print(f"\nturn 2 stop_reason: {r2.stop_reason}")
    print(f"\nMODEL'S ANSWER:\n  {answer}")

    # --- what it cost --------------------------------------------------------
    tok_in = r1.usage.input_tokens + r2.usage.input_tokens
    tok_out = r1.usage.output_tokens + r2.usage.output_tokens
    total = _cost(r1.usage) + _cost(r2.usage)
    print("\nCOST")
    print(f"  input tokens   {tok_in:>7,}")
    print(f"  output tokens  {tok_out:>7,}")
    print(f"  this run       ${total:.5f}")
    print(f"  runs per $1    {1 / total:,.0f}" if total else "")


if __name__ == "__main__":
    main()
