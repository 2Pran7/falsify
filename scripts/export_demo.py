"""Freeze the stored evidence into web/data/demo.json for the static demo.

Usage
-----
    python scripts/export_demo.py
    python scripts/export_demo.py --survivorship web/data/survivorship.json
    python scripts/export_demo.py --allow-partial      # some verdicts missing
    python scripts/export_demo.py --allow-dirty        # uncommitted code (not for ship)

Order of operations for a ship, each step a separate command:

    python scripts/run_evals.py --compare --store
    python scripts/run_survivorship.py --save web/data/survivorship.json
    python scripts/export_demo.py --survivorship web/data/survivorship.json

NO API KEY, NO SPEND, NO RECOMPUTATION. This reads what Postgres holds and
writes it down. Every refusal lives in `falsify.demo.build_snapshot`, which is
tested without a database; this file only gathers inputs and writes the output.

One refusal lives here because it needs git: the snapshot records the commit it
was exported from, and a commit with uncommitted changes under src/ or
scripts/ does not identify the code that produced the rows. --allow-dirty
overrides, and the snapshot then says `-dirty` on the page.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

from falsify.agent import tools as T
from falsify.demo import ExportError, build_snapshot
from falsify.eval import store as eval_store
from falsify.notes import store as notes_store

DEFAULT_OUT = pathlib.Path("web/data/demo.json")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--survivorship", type=pathlib.Path,
                    help="JSON written by run_survivorship.py --save")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    commit = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain", "--", "src", "scripts", "db")
    if dirty:
        if not args.allow_dirty:
            print("Refusing: uncommitted changes under src/, scripts/ or db/:")
            print(dirty)
            print("Commit first, so the snapshot names the code that produced it.")
            return 2
        commit = f"{commit}-dirty"

    for name, store in (("eval_result", eval_store), ("research_note", notes_store)):
        if not store.available():
            print(f"Refusing: Postgres unreachable, cannot read {name}. docker compose up -d")
            return 2

    rows = eval_store.fetch_all()
    notes = notes_store.fetch_all()
    surv = None
    if args.survivorship:
        if not args.survivorship.exists():
            print(f"Refusing: {args.survivorship} does not exist. Produce it with")
            print(f"  python scripts/run_survivorship.py --save {args.survivorship.as_posix()}")
            return 2
        surv = json.loads(args.survivorship.read_text(encoding="utf-8"))

    try:
        snap = build_snapshot(
            rows,
            notes,
            survivorship=surv,
            code_commit=commit,
            trial_variance=T.TRIAL_VARIANCE,
            allow_partial=args.allow_partial,
        )
    except ExportError as e:
        print(f"Refusing: {e}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")

    ev = snap["evals"]
    print(f"wrote {args.out}  ({args.out.stat().st_size:,} bytes)")
    print(f"  commit        {commit}")
    print(f"  registry sha  {snap['registry_sha'][:16]}")
    for u in ev["universes"]:
        t = ev["tally"][u]
        print(f"  {u:<14} " + "  ".join(f"{k} {v}" for k, v in t.items()))
    if ev["missing"]:
        print(f"  MISSING       {', '.join(ev['missing'])}")
    c = snap["notes"]["counts"]
    print(f"  notes         {c['total']} total, {c['publishable']} publishable, "
          f"{c['unpublishable']} unpublishable (all exported)")
    print(f"  survivorship  {'included' if surv else 'NOT included (pass --survivorship)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
