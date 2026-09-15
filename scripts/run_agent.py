"""Session 4: the first real hypothesis, end to end, against the real API.

    python scripts/run_agent.py
    python scripts/run_agent.py "do low-volatility stocks outperform?"
    python scripts/run_agent.py --max-turns 6 --budget 50000

Everything before this ran on scripted responses and cost nothing. This spends
money, so it prints what it spent, every time, whether it succeeded or not.

WHAT TO LOOK FOR IN THE OUTPUT, in order of importance:

  1. The TOOL SEQUENCE. A correct run is fetch_data, compute_feature,
     run_backtest, analyze_results. If analyze_results is missing, the model
     reached a conclusion from a raw Sharpe and the answer is worthless
     regardless of how confident it sounds.

  2. The STOP REASON. Only "end_turn" means the model finished. Any other value
     means a guard fired and the answer, if there is one, is partial.

  3. Whether the VERDICT MATCHES THE STATISTICS. On this sample the deflated
     Sharpe will be poor and the minimum track record length will be measured
     in years. An agent that reports "momentum works" over that is failing, and
     an agent that reports "insufficient evidence" is doing exactly what the
     project is named for. The second outcome is the better demo.

The transcript is written to .cache/runs/ so a run can be read back later
without paying for it again. That directory is gitignored and is a debugging
aid, not an artifact: the DURABLE record is the row this script writes to
`research_note` (Module 5). Read it back with `scripts/show_notes.py`, which
needs no API key and costs nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, "src")

from anthropic import Anthropic
from dotenv import load_dotenv

from falsify.agent.loop import RunConfig, run
from falsify.agent.session import Session
from falsify.agent.provenance import check_run
from falsify.agent.tools import TRIAL_VARIANCE, ToolError, analyze_results
from falsify.notes import from_run
from falsify.notes import store as note_store

load_dotenv()

DEFAULT_HYPOTHESIS = "Do stocks that went up over the last year keep going up?"
RUNS_DIR = pathlib.Path(".cache/runs")


def _print_header(hypothesis: str, config: RunConfig) -> None:
    print("=" * 72)
    print(f"HYPOTHESIS: {hypothesis}")
    print(
        f"model {config.model} | max {config.max_turns} turns | "
        f"budget {config.max_total_tokens:,} tokens"
    )
    print("=" * 72)


def _print_tools(result) -> None:
    print("\nTOOL SEQUENCE")
    if not result.tool_calls:
        print("  (none — the model answered without touching the pipeline)")
        return
    for i, call in enumerate(result.tool_calls, 1):
        mark = "ok  " if call["ok"] else "FAIL"
        args = json.dumps(call["input"])
        if len(args) > 60:
            args = args[:57] + "..."
        print(f"  {i}. {mark} {call['name']:<16} {args}")
        if not call["ok"]:
            print(f"          -> {call['error'][:100]}")

    names = [c["name"] for c in result.tool_calls if c["ok"]]
    if "analyze_results" not in names:
        print(
            "\n  WARNING: analyze_results was never called successfully.\n"
            "  Any conclusion below rests on an unadjusted Sharpe ratio and\n"
            "  should not be believed."
        )


def _print_stats(result) -> None:
    """The honesty table, recomputed at the FINAL trial count.

    Why recompute rather than read what the agent was told: the deflation
    benchmark depends on how many strategies were tried, and the count is taken
    when analyze_results is called. An agent that analyses its first backtest
    and then runs two more has a first row deflated for one trial when the
    honest number turned out to be three. Every row here is recomputed at the
    count the run actually finished on, so the table cannot be more flattering
    than the run deserves.

    This is also the number to quote. The model's prose is the model's; this is
    the pipeline's.
    """
    session = result.session
    handles = session.handles("backtest")
    if not handles:
        return

    n = session.n_backtests
    print(f"\nWHAT THE PIPELINE ACTUALLY COMPUTED  (all rows deflated at N={n})")
    print(
        f"  {'backtest':<11}{'variant':<28}{'Sharpe':>8}"
        f"{'P(SR>0)':>10}{'P(beats N)':>12}{'MinTRL':>10}"
    )
    for h in handles:
        s = session.summary(h)
        variant = (
            f"{s['feature']}, {s['n_buckets']}b, "
            f"{'L/S' if s['long_short'] else 'long-only'}"
        )
        try:
            stats = analyze_results(session, h, n_trials=n)
        except ToolError as exc:
            print(f"  {h:<11}{variant:<28}  analysis failed: {exc}")
            continue
        print(
            f"  {h:<11}{variant:<28}{s['sharpe']:>8.2f}"
            f"{stats['prob_sharpe_above_zero']:>10.3f}"
            f"{stats['prob_beats_best_of_n_trials']:>12.3f}"
            f"{stats['min_track_record_length_days']:>10.0f}"
        )

    print(
        f"\n  P(SR>0) and P(beats N) are PROBABILITIES, not Sharpe ratios.\n"
        f"  P(beats N) is the deflated Sharpe: below 0.5 the evidence does not\n"
        f"  survive having tried {n} strateg{'y' if n == 1 else 'ies'}.\n"
        f"  MinTRL is days of data needed before the Sharpe is distinguishable\n"
        f"  from zero at 95%. Compare it to the invested days above."
    )
    print(
        f"\n  ASSUMPTION: P(beats N) uses a trial-Sharpe variance of "
        f"{TRIAL_VARIANCE}, which is\n"
        f"  an assumed value, not a measured one. It is measured from the six\n"
        f"  anomalies at Module 6. Every deflated figure above moves with it, so\n"
        f"  it is printed rather than buried in the tool output."
    )
    if any(not session.summary(h)["long_short"] for h in handles):
        print(
            "\n  NOTE: a long-only variant is present. Long-only holds the market\n"
            "  plus a tilt, so its Sharpe carries beta and is NOT a test of a\n"
            "  cross-sectional hypothesis. There is no benchmark tool yet, so the\n"
            "  beta cannot be separated out. Read the long/short rows."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hypothesis", nargs="?", default=DEFAULT_HYPOTHESIS)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--budget", type=int, default=200_000,
                    help="total token budget for the run")
    ap.add_argument("--eval-key", default=None,
                    help="Module 6 anomaly slug; re-running one overwrites its note")
    ap.add_argument("--no-store", action="store_true",
                    help="skip the database write (the run still costs money)")
    args = ap.parse_args()

    config = RunConfig(max_turns=args.max_turns, max_total_tokens=args.budget)
    _print_header(args.hypothesis, config)

    session = Session()
    result = run(args.hypothesis, Anthropic(), session=session, config=config)

    _print_tools(result)
    _print_stats(result)

    print("\nMODEL'S ANSWER")
    print("-" * 72)
    print(result.answer or "(no text returned)")
    print("-" * 72)

    prov = check_run(result)
    print("\nNUMERIC PROVENANCE")
    print("  " + prov.summary().replace("\n", "\n  "))
    if not prov.ok:
        print(
            "\n  Every figure in a research note must trace to a tool result.\n"
            "  The numbers above appear in the prose and in no tool output, so\n"
            "  either the model computed them itself or it recalled them. Either\n"
            "  way the note is not publishable as it stands."
        )

    print(f"\nRUN: {result.summary()}")
    if not result.completed:
        print(
            f"  NOT COMPLETE: a guard fired ({result.stop_reason}). The answer\n"
            "  above is partial and must not be quoted as a finding."
        )

    # -- Module 5: the durable record --------------------------------------
    #
    # Built whether or not it is publishable, and stored either way. A run that
    # failed provenance or was cut short is evidence, and an eval suite that
    # keeps only its successes is a highlight reel.
    note = from_run(
        result,
        hypothesis=args.hypothesis,
        provenance_report=prov,
        session=session,
        eval_key=args.eval_key,
    )
    print("\nRESEARCH NOTE")
    if note.publishable:
        print("  PUBLISHABLE: provenance passed, the run completed, statistics computed.")
    else:
        print("  NOT PUBLISHABLE — stored anyway, as a failure:")
        for reason in note.unpublishable_reasons:
            print(f"    - {reason}")

    if args.no_store:
        print("  --no-store: not written to the database.")
    else:
        try:
            note_store.apply_schema()
            note_id = note_store.save(note)
            counts = note_store.counts()
            print(f"  stored as {note_id}")
            print(
                f"  research_note now holds {counts['total']} note(s): "
                f"{counts['publishable']} publishable, "
                f"{counts['unpublishable']} not."
            )
            print(f"  read it back with:  python scripts/show_notes.py {note_id}")
        except Exception as exc:
            # The run has already been paid for. Losing the note to a database
            # that is down would be the one failure worth shouting about, so
            # say so loudly and point at the transcript that did survive.
            print(f"  DATABASE WRITE FAILED: {exc}")
            print(
                "  The run cost real money and its note is NOT durable. Start\n"
                "  Postgres (docker compose up -d) and re-store from the\n"
                "  transcript below, or re-run."
            )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "hypothesis": args.hypothesis,
                "stop_reason": result.stop_reason,
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "answer": result.answer,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cache_read_tokens": result.cache_read_tokens,
                "cache_write_tokens": result.cache_write_tokens,
                "cost_usd": round(result.cost_usd, 6),
                "provenance": {
                    "ok": prov.ok,
                    "checked": prov.checked,
                    "verified": prov.verified,
                    "unverified": [str(u) for u in prov.unverified],
                },
                "backtests": {
                    h: session.summary(h) for h in session.handles("backtest")
                },
                "messages": result.messages,
            },
            indent=2,
            default=str,
        )
    )
    print(f"  transcript saved to {path}")


if __name__ == "__main__":
    main()
