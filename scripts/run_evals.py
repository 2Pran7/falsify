"""The eval suite: six pre-registered anomalies, scored honestly.

Usage
-----
    python scripts/run_evals.py
    python scripts/run_evals.py --universe point_in_time
    python scripts/run_evals.py --compare              # both universes, per-anomaly gap
    python scripts/run_evals.py --compare --store      # and write the verdicts
    python scripts/run_evals.py --check                # CI gate: refuse stale rows
    python scripts/run_evals.py --registry             # print the pre-registration

NO API KEY, NO SPEND. The suite runs the pipeline, not the model: the question
is whether this PIPELINE rediscovers published effects, and a language model in
the loop would make a failing row unattributable between the data and the
model's choices.

WHAT THIS PRINTS IS NOT YET A RESULT. On roughly two years of one up market, of
which the first is momentum warmup, six anomalies on ~250 invested days of a
single regime cannot support a published claim. The suite is built to SAY so --
`insufficient_data` is a separate outcome from `fail` for exactly this reason --
rather than to fill the table. Nothing here is quotable until the Polygon
Starter backfill lengthens the panel.

Three things this script refuses to do, each of them a check that exits rather
than warns:

  --check exits non-zero when any stored row carries a `registry_sha` that no
  longer matches the current registry. A stale row was scored against a
  prediction that has since been edited, and quoting it beside a fresh one puts
  two different standards in one table with nothing to distinguish them.

  --store refuses when the panel is too short to support any verdict at all, so
  a table of `insufficient_data` rows cannot be mistaken for a run that failed.

  The point-in-time leg refuses to run on a single snapshot date, in
  `tools.fetch_data` rather than here. That is the failure mode that fails by
  looking healthy.
"""
from __future__ import annotations

import argparse
import json
import sys

from falsify.agent import tools as T
from falsify.eval import registry as R
from falsify.eval.runner import compare_universes, run_suite

# Below this, no verdict on this panel means anything and --store says so.
MIN_TRADING_DAYS_TO_STORE = 300

_MARK = {
    "pass": "PASS",
    "partial": "PART",
    "fail": "FAIL",
    "insufficient_data": "N/A ",
}


def _fmt(x: float | None, nd: int = 3) -> str:
    return "     -" if x is None else f"{x:>6.{nd}f}"


def print_registry() -> None:
    print(f"\nPRE-REGISTRATION  sha256 {R.registry_digest()}")
    print(f"hashed fields: {', '.join(R.PREREGISTERED_FIELDS)}")
    print("prose fields are deliberately NOT hashed, so fixing a citation typo")
    print("does not invalidate every verdict already stored against it.\n")
    print(f"{'key':26s} {'feature':14s} {'dir':>4s} {'pub SR':>7s} {'needs':>7s}  citation")
    print("-" * 110)
    for a in sorted(R.ANOMALIES, key=lambda x: x.key):
        print(
            f"{a.key:26s} {a.feature:14s} {a.direction:>+4d} "
            f"{a.published_sharpe:>7.2f} {a.min_history_days:>6d}d  {a.citation}"
        )
    print()
    for a in sorted(R.ANOMALIES, key=lambda x: x.key):
        print(f"  {a.key}\n    caveat: {a.caveat}\n")


def print_run(run) -> None:
    p = run.panel
    print(f"\n=== {run.universe.upper()} ===")
    print(
        f"panel: {p['n_rows']:,} rows, {p['n_tickers']} tickers, "
        f"{p['n_trading_days']} trading days, {p['first_date']} to {p['last_date']}"
    )
    if run.universe == "point_in_time":
        print(
            f"       {p.get('n_snapshots')} membership snapshots from "
            f"{p.get('first_snapshot')}, {p.get('n_member_days'):,} member-days"
        )
    print(f"trials counted for deflation: {run.n_trials}")
    print(f"registry sha: {run.registry_sha}")
    print()
    print(
        f"{'':5s}{'anomaly':26s} {'SR':>7s} {'orient':>7s} {'pub':>6s} "
        f"{'p':>7s} {'BH p':>7s} {'defl':>6s} {'days':>6s}"
    )
    print("-" * 96)
    for s in run.score:
        print(
            f"{_MARK[s.verdict]:5s}{s.key:26s} {_fmt(s.realised_sharpe)} "
            f"{_fmt(s.oriented_sharpe)} {s.published_sharpe:>6.2f} "
            f"{_fmt(s.p_value, 4)} {_fmt(s.p_value_adjusted, 4)} "
            f"{_fmt(s.deflated_psr)} {s.n_invested_days:>6d}"
        )
    print()
    for s in run.score:
        print(f"  {s.key}  [{s.verdict}]")
        for r in s.reasons:
            print(f"      - {r}")
        if s.caveat:
            print(f"      caveat: {s.caveat}")
        print()

    t = run.score.tally()
    print(
        f"tally: {t['pass']} pass, {t['partial']} partial, {t['fail']} fail, "
        f"{t['insufficient_data']} insufficient_data "
        f"(of {len(run.score)}; {run.score.n_tested} entered the BH correction)"
    )
    if run.errors:
        print("\ntool errors, recorded rather than swallowed:")
        for k, e in run.errors.items():
            print(f"  {k}: {e}")

    # THE ASSUMPTION AND THE MEASUREMENT, SIDE BY SIDE, NEITHER SUBSTITUTED.
    print(
        f"\ntrial variance: assumed {run.trial_variance_assumed} "
        f"(tools.TRIAL_VARIANCE), measured from this suite's own trials "
        f"{run.trial_variance_measured}"
    )
    print(
        "  NOT substituted. Six points is far too few for a variance, and replacing a\n"
        "  labelled assumption with an unlabelled estimate would be invisible in every\n"
        "  number downstream of it. Revisit after the backfill."
    )


def print_gaps(gaps: list[dict]) -> None:
    print("\n=== SURVIVORSHIP GAP, PER ANOMALY ===")
    print("The Module 3 measurement generalised from one strategy to six. A gap that")
    print("appears on momentum and nothing else is a different finding from one that")
    print("appears on all six, and only the second supports a general claim.\n")
    print(f"{'anomaly':26s} {'current':>9s} {'pit':>9s} {'gap':>9s}  verdicts")
    print("-" * 78)
    for g in gaps:
        print(
            f"{g['key']:26s} {_fmt(g['current_sharpe']):>9s} "
            f"{_fmt(g['pit_sharpe']):>9s} {_fmt(g['sharpe_gap']):>9s}  "
            f"{g['current_verdict']} -> {g['pit_verdict']}"
        )
    real = [g["sharpe_gap"] for g in gaps if g["sharpe_gap"] is not None]
    if real:
        print(f"\nmean gap across {len(real)} testable anomalies: {sum(real)/len(real):+.3f}")
    print(
        "\nLOWER BOUND, not a point estimate. Restoring a dropped name to the universe\n"
        "is not the same as capturing its delisting return: a cash acquisition simply\n"
        "stops having prices. Shumway (1997) covers the size of the missing piece."
    )


def do_check(dsn: str | None) -> int:
    from falsify.eval import store

    if not store.available(dsn):
        print("database unreachable; --check cannot verify anything.", file=sys.stderr)
        return 2
    store.apply_schema(dsn)

    key = store.primary_key_columns(dsn)
    if key != ["eval_key", "universe"]:
        print(
            f"SCHEMA DRIFT: {store.TABLE} primary key on the live server is {key}, "
            "expected ['eval_key', 'universe'].\n"
            "CREATE TABLE IF NOT EXISTS is a no-op against an existing table, so a "
            "schema edit after the first `docker compose up` is silently ignored on "
            "that database. Drop and recreate the table, or ALTER it.",
            file=sys.stderr,
        )
        return 1

    stale = store.stale_rows(R.registry_digest(), dsn)
    if stale:
        print(
            f"STALE VERDICTS: {len(stale)} row(s) were scored against a registry that "
            "has since changed. Re-run the suite before quoting this table.",
            file=sys.stderr,
        )
        for r in stale:
            print(f"  {r['eval_key']} / {r['universe']}: {r['registry_sha'][:12]}",
                  file=sys.stderr)
        return 1

    print(f"OK: primary key {key}, no stale verdicts, registry {R.registry_digest()[:12]}")
    print(f"counts: {store.counts(dsn)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=list(T.UNIVERSES), default="current")
    ap.add_argument("--compare", action="store_true",
                    help="run both universes and report the gap per anomaly")
    ap.add_argument("--store", action="store_true", help="write verdicts to Postgres")
    ap.add_argument("--check", action="store_true",
                    help="verify the stored table; exits non-zero on drift or stale rows")
    ap.add_argument("--registry", action="store_true",
                    help="print the pre-registration and its hash, then exit")
    ap.add_argument("--n-buckets", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if args.registry:
        print_registry()
        return 0
    if args.check:
        return do_check(args.dsn)

    kwargs = dict(
        n_buckets=args.n_buckets, cost_bps=args.cost_bps,
        start=args.start, end=args.end,
    )

    try:
        if args.compare:
            out = compare_universes(**kwargs)
            runs = [out["current"], out["point_in_time"]]
        else:
            runs = [run_suite(universe=args.universe, **kwargs)]
            out = None
    except T.ToolError as exc:
        print(f"\nthe suite could not run: {exc}\n", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(
            {
                r.universe: {
                    "registry_sha": r.registry_sha,
                    "n_trials": r.n_trials,
                    "panel": r.panel,
                    "tally": r.score.tally(),
                    "scores": [s.to_dict() for s in r.score],
                }
                for r in runs
            },
            indent=2, default=str,
        ))
    else:
        for r in runs:
            print_run(r)
        if out is not None:
            print_gaps(out["gaps"])

    if args.store:
        from falsify.eval import store

        short = [r for r in runs if r.panel["n_trading_days"] < MIN_TRADING_DAYS_TO_STORE]
        if short:
            print(
                f"\nREFUSING TO STORE: the panel has "
                f"{short[0].panel['n_trading_days']} trading days, under the "
                f"{MIN_TRADING_DAYS_TO_STORE} minimum. A table of insufficient_data rows "
                "is indistinguishable from a suite that ran and failed, and it is the "
                "table Module 7 renders. Backfill first.",
                file=sys.stderr,
            )
            return 1
        store.apply_schema(args.dsn)
        for r in runs:
            n = store.save_suite(r.score, r.universe, r.registry_sha, dsn=args.dsn)
            print(f"stored {n} verdicts for universe={r.universe}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
