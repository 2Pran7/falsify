"""Module 5: the research note, as a durable record rather than a print.

schema   the Note record, how one is built from a run, and the publishable rule
store    read and write notes in Postgres
render   Note -> markdown

The module exists because of one gap: Modules 1 to 4 can produce a verified
research note and then lose it when the process exits. A JSON transcript lands
in `.cache/runs/`, which is gitignored and not a publishable artifact. Module 7
has to serve a note without making a Claude call and without a rerun, and it
cannot do that from a file nobody kept.
"""
from falsify.notes.schema import (
    BacktestRecord,
    Note,
    NoteError,
    ProvenanceVerdict,
    RunMetadata,
    from_run,
)

__all__ = [
    "BacktestRecord",
    "Note",
    "NoteError",
    "ProvenanceVerdict",
    "RunMetadata",
    "from_run",
]
