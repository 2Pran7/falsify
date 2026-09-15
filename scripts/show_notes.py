"""Read stored research notes back. No API key, no Claude call, no spend.

    python scripts/show_notes.py                 list every note, newest first
    python scripts/show_notes.py --latest        render the most recent one
    python scripts/show_notes.py NOTE_ID         render one as markdown
    python scripts/show_notes.py --eval momentum_12_1
    python scripts/show_notes.py --latest --out note.md

THIS SCRIPT IS THE PROOF THAT MODULE 5 WORKS. It shares no memory with the
process that produced the note and it cannot make a model call: everything it
prints came out of the database. That is the whole claim — Module 7 will serve
a note the same way, from pre-computed rows, without a rerun.

It also prints the unpublishable count, deliberately. A demo that shows six
anomalies and says nothing about the runs that failed is a highlight reel, and
this project is pitched against exactly that.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, "src")

from falsify.notes import store
from falsify.notes.render import to_markdown, to_summary_line


def _list() -> int:
    notes = store.fetch_all()
    counts = store.counts()
    if not notes:
        print("No notes stored yet. Run scripts/run_agent.py.")
        return 0

    print(f"{counts['total']} note(s): {counts['publishable']} publishable, "
          f"{counts['unpublishable']} not.\n")
    for note in notes:
        print(f"  {note.note_id}  {to_summary_line(note)}")
    if counts["unpublishable"]:
        print(
            f"\n  {counts['unpublishable']} note(s) are not publishable. They are\n"
            "  kept on purpose: a suite that stores only its successes cannot be\n"
            "  used as evidence about anything."
        )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("note_id", nargs="?", help="render this note; omit to list all")
    ap.add_argument("--eval", dest="eval_key", help="render the note for this eval case")
    ap.add_argument("--latest", action="store_true",
                    help="render the most recent note, so no id has to be typed")
    ap.add_argument("--out", type=pathlib.Path, help="write the markdown to a file")
    args = ap.parse_args()

    if not store.available():
        print(
            "Postgres is unreachable. Start it with `docker compose up -d`, or\n"
            "set DATABASE_URL.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    store.apply_schema()

    if not args.note_id and not args.eval_key and not args.latest:
        raise SystemExit(_list())

    if args.eval_key:
        note = store.fetch_by_eval_key(args.eval_key)
    elif args.latest:
        # fetch_all is newest first, so the most recent run is index 0.
        recent = store.fetch_all(limit=1)
        note = recent[0] if recent else None
    else:
        note = store.fetch(args.note_id)
    if note is None:
        print("No such note.", file=sys.stderr)
        raise SystemExit(1)

    markdown = to_markdown(note)
    if args.out:
        args.out.write_text(markdown, encoding="utf-8")
        print(f"written to {args.out}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
