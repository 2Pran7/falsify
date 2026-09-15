"""Persistence. Skips cleanly when Postgres is not up, like the SPY smoke test.

Module 5 closes when a note survives a PROCESS RESTART, not when the code is
written, so the test that matters here spawns a second interpreter and reads
the note back in it. An in-process round trip would pass against a dictionary.

The other thing under test is that the derived `publishable` column cannot
become the source of truth. It exists so Module 7 can filter in SQL; the Note
recomputes the verdict on read. A row edited in psql to say true must not come
back publishable, and there is a test that does exactly that edit.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

import pytest

from falsify.notes import store
from falsify.notes.schema import Note
from tests.test_notes_schema import a_backtest, a_note, bad_provenance, good_run

DSN = os.getenv("DATABASE_URL")

needs_db = pytest.mark.skipif(
    not store.available(DSN),
    reason="Postgres unreachable (docker compose up, or set DATABASE_URL)",
)


@pytest.fixture
def db():
    """Schema applied, and every note this test writes removed afterwards."""
    store.apply_schema(DSN)
    written: list[str] = []

    class Handle:
        dsn = DSN

        def save(self, note: Note) -> str:
            note_id = store.save(note, DSN)
            written.append(note_id)
            written.append(note.note_id)
            return note_id

    yield Handle()

    for note_id in set(written):
        store.delete(note_id, DSN)


# --------------------------------------------------------------------------
# the point of the module
# --------------------------------------------------------------------------

@needs_db
def test_a_note_survives_a_process_restart(db):
    """The whole of Module 5 in one test.

    A second interpreter, a fresh connection, no shared memory with this one.
    Before Module 5 the note existed only in the model's final turn and in a
    gitignored transcript; after it, a note read back here was never in this
    process at all.
    """
    note = a_note(prose="The long/short spread returned 10.02% over 239 days.")
    db.save(note)

    script = (
        "import sys, json; sys.path.insert(0, 'src');"
        "from falsify.notes import store;"
        f"n = store.fetch({note.note_id!r}, {DSN!r});"
        "print(json.dumps({'prose': n.prose, 'publishable': n.publishable,"
        " 'hypothesis': n.hypothesis, 'sharpe': n.backtests[0].metrics['sharpe']}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["prose"] == note.prose
    assert out["hypothesis"] == note.hypothesis
    assert out["publishable"] is True
    assert out["sharpe"] == 0.45


@needs_db
def test_round_trip_through_postgres_is_exact(db):
    note = a_note()
    db.save(note)
    assert store.fetch(note.note_id, DSN) == note


@needs_db
def test_structured_statistics_survive_as_data_not_as_rendered_text(db):
    """Module 7 renders the table from these fields. If they came back as a
    string the page would be parsing prose, which is the thing this schema
    exists to avoid."""
    note = a_note()
    db.save(note)
    back = store.fetch(note.note_id, DSN)
    stats = back.backtests[0].statistics
    assert isinstance(stats, dict)
    assert stats["prob_beats_best_of_n_trials"] == pytest.approx(0.4218)
    assert isinstance(back.backtests[0].metrics["sharpe"], float)


# --------------------------------------------------------------------------
# failures are kept
# --------------------------------------------------------------------------

@needs_db
def test_a_note_that_failed_provenance_is_stored_as_failed_not_refused(db):
    """Deleting failures is how an eval suite becomes a highlight reel, and
    this project's entire pitch is that the failures are shown."""
    failed = a_note(provenance=bad_provenance())
    db.save(failed)
    back = store.fetch(failed.note_id, DSN)
    assert back is not None
    assert not back.publishable
    assert back.provenance.unverified == failed.provenance.unverified


@needs_db
def test_a_cut_short_run_is_stored_and_says_which_guard_fired(db):
    note = a_note(run=good_run(stop_reason="token_budget", turns=12))
    db.save(note)
    back = store.fetch(note.note_id, DSN)
    assert back.run.stop_reason == "token_budget"
    assert not back.publishable


@needs_db
def test_counts_reports_the_unpublishable_ones(db):
    before = store.counts(DSN)
    db.save(a_note())
    db.save(a_note(provenance=bad_provenance()))
    after = store.counts(DSN)
    assert after["total"] == before["total"] + 2
    assert after["publishable"] == before["publishable"] + 1
    assert after["unpublishable"] == before["unpublishable"] + 1


# --------------------------------------------------------------------------
# the derived column is an index, never the truth
# --------------------------------------------------------------------------

@needs_db
def test_editing_the_publishable_column_does_not_make_a_note_publishable(db):
    """The forgery test. `publishable` is stored for SQL filtering only; the
    Note recomputes it from the provenance verdict and the run metadata."""
    import psycopg

    failed = a_note(provenance=bad_provenance())
    db.save(failed)
    with psycopg.connect(DSN or "") as conn:
        conn.execute(
            f"UPDATE {store.TABLE} SET publishable = true, "
            "unpublishable_reasons = '[]'::jsonb WHERE note_id = %s",
            (failed.note_id,),
        )
        conn.commit()

    assert store.fetch(failed.note_id, DSN).publishable is False
    # And the tampered row cannot smuggle itself into the published list.
    ids = [n.note_id for n in store.fetch_all(DSN, publishable_only=True)]
    assert failed.note_id not in ids


@needs_db
def test_fetch_all_publishable_only_excludes_the_failures(db):
    good, bad = a_note(), a_note(provenance=bad_provenance())
    db.save(good)
    db.save(bad)
    ids = [n.note_id for n in store.fetch_all(DSN, publishable_only=True)]
    assert good.note_id in ids
    assert bad.note_id not in ids


# --------------------------------------------------------------------------
# Module 6 re-runs every anomaly and overwrites
# --------------------------------------------------------------------------

@needs_db
def test_the_same_eval_key_replaces_rather_than_accumulates(db):
    # Namespaced: this suite is run against a live database, and an eval_key
    # that looked like a real one ("momentum_12_1") would overwrite and then
    # delete a genuine Module 6 note on teardown.
    key = "__pytest__momentum_12_1"
    first = a_note(eval_key=key, prose="first attempt")
    second = a_note(eval_key=key, prose="second attempt")
    db.save(first)
    db.save(second)

    current = store.fetch_by_eval_key(key, DSN)
    assert current.prose == "second attempt"
    assert store.fetch(first.note_id, DSN) is None


@needs_db
def test_notes_without_an_eval_key_accumulate(db):
    """Ad-hoc runs are history, not a keyed slot: two runs of the same question
    are two rows."""
    before = store.counts(DSN)["total"]
    db.save(a_note(eval_key=None))
    db.save(a_note(eval_key=None))
    assert store.counts(DSN)["total"] == before + 2


@needs_db
def test_apply_schema_is_safe_to_reapply(db):
    store.apply_schema(DSN)
    store.apply_schema(DSN)
    note = a_note()
    db.save(note)
    assert store.fetch(note.note_id, DSN) is not None


# --------------------------------------------------------------------------
# odds and ends
# --------------------------------------------------------------------------

@needs_db
def test_fetching_a_missing_note_returns_none_rather_than_raising(db):
    assert store.fetch("00000000-0000-0000-0000-000000000000", DSN) is None


@needs_db
def test_timestamps_come_back_as_the_same_instant(db):
    note = a_note(created_at=dt.datetime(2026, 9, 15, 11, 30, tzinfo=dt.timezone.utc))
    db.save(note)
    assert store.fetch(note.note_id, DSN).created_at == note.created_at


@needs_db
def test_a_run_with_no_backtests_stores_and_is_unpublishable(db):
    note = a_note(backtests=(), run=good_run(tool_sequence=("fetch_data",)))
    db.save(note)
    back = store.fetch(note.note_id, DSN)
    assert back.backtests == ()
    assert not back.publishable


@needs_db
def test_prose_with_quotes_and_newlines_survives(db):
    nasty = "It said: \"don't\" -- and then\n\n* a bullet with 'quotes' & 100%\n"
    note = a_note(prose=nasty)
    db.save(note)
    assert store.fetch(note.note_id, DSN).prose == nasty


@needs_db
def test_a_backtest_whose_analysis_failed_keeps_its_error(db):
    note = a_note(
        backtests=(a_backtest(analysed=False, analysis_error="series too short"),)
    )
    db.save(note)
    back = store.fetch(note.note_id, DSN)
    assert back.backtests[0].analysis_error == "series too short"
    assert back.backtests[0].statistics == {}


def test_available_is_false_for_a_dead_dsn():
    """Runs without a database: the guard the whole module's skip depends on
    must not itself need one."""
    assert store.available("postgresql://nobody@127.0.0.1:1/none") is False
