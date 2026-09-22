"""Verdicts in Postgres. Skips cleanly when the database is not up.

THE TEST THAT JUSTIFIES THIS FILE'S EXISTENCE is at the bottom:

    test_the_primary_key_on_the_LIVE_SERVER_is_eval_key_and_universe

An idempotent schema statement is a no-op, and a no-op is indistinguishable
from a success. Every schema statement in this project is
`CREATE TABLE IF NOT EXISTS`, so any edit after the first `docker compose up`
is silently ignored on that database, forever, while every read and write keeps
succeeding. The injection audit dropped `universe` from the primary key: the
string changed, nothing else did, and the whole suite stayed green.

So the invariant is checked against `pg_index` on the live server, not against
the string that was supposed to create it. The string test below is still worth
having -- it catches the two SCHEMA_SQL copies drifting apart before anyone
touches a database -- but it is the weaker of the two and is labelled as such.

Test rows use keys prefixed `zz_test_`. The Module 5 lesson, paid for once
already: that file's store tests originally used `eval_key="momentum_12_1"`,
which against a live database overwrote a genuine note and deleted it on
teardown. A test that destroys production data while passing.
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib

import pytest

from falsify.eval import registry as R
from falsify.eval import store
from falsify.eval.score import Measurement, score_anomaly

DSN = os.getenv("DATABASE_URL")

needs_db = pytest.mark.skipif(
    not store.available(DSN),
    reason="Postgres unreachable (docker compose up, or set DATABASE_URL)",
)

# Never a real registry key. See the module docstring.
TEST_KEY = "zz_test_momentum"


def a_score(key: str = TEST_KEY, **over):
    anomaly = R.Anomaly(
        key=key, feature="mom_12_1", direction=1, published_sharpe=0.5,
        min_history_days=252,
        citation="test", hypothesis="a test hypothesis, long enough to pass",
        sharpe_source="a test source, long enough to pass the length check",
        caveat="a test caveat, long enough to pass the length check",
    )
    fields = dict(
        key=key, realised_sharpe=0.8, p_value=0.001, deflated_psr=0.99,
        history_days=800, n_invested_days=250, n_trials=6,
    )
    m = Measurement(**{**fields, **over})
    return score_anomaly(m, anomaly, 0.001)


@pytest.fixture
def db():
    """Schema applied, and every row this test writes removed afterwards."""
    store.apply_schema(DSN)
    written: list[tuple[str, str]] = []

    class Handle:
        dsn = DSN

        def save(self, score, universe="current", sha=None):
            store.save(score, universe, sha or R.registry_digest(), dsn=DSN)
            written.append((score.key, universe))

    yield Handle()
    for key, universe in written:
        store.delete(key, universe, dsn=DSN)


# --- the string check: the weaker of the two --------------------------------


def test_both_copies_of_the_schema_declare_the_same_primary_key():
    """Catches the two SCHEMA_SQL copies drifting apart before a database is
    involved at all. It does NOT catch a deployed table that never got the
    edit; only the pg_index test below does that."""
    sql_file = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    text = sql_file.read_text()
    assert "PRIMARY KEY (eval_key, universe)" in text
    assert "PRIMARY KEY (eval_key, universe)" in store.SCHEMA_SQL


def test_the_schema_file_documents_the_no_op_hazard():
    """The lesson belongs next to the statement that causes it, or the next
    person to edit the file reintroduces the same silent drift."""
    sql_file = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    text = sql_file.read_text()
    assert "NO-OP AGAINST AN EXISTING" in text


# --- everything below needs a database --------------------------------------


@needs_db
def test_the_primary_key_on_the_LIVE_SERVER_is_eval_key_and_universe(db):
    """THE TEST THAT CATCHES WHAT CREATE TABLE IF NOT EXISTS HIDES.

    Read from pg_index on the server, not from the string that was supposed to
    create it. This is the only check that fails when the code and the deployed
    table have drifted apart, which is the state every schema edit after the
    first `docker compose up` silently produces.
    """
    assert store.primary_key_columns(DSN) == ["eval_key", "universe"]


@needs_db
def test_a_verdict_round_trips(db):
    s = a_score()
    db.save(s)
    row = store.fetch(TEST_KEY, "current", dsn=DSN)
    assert row is not None
    assert row["verdict"] == s.verdict
    assert row["oriented_sharpe"] == pytest.approx(s.oriented_sharpe)
    assert row["direction"] == 1


@needs_db
def test_the_same_anomaly_on_two_universes_is_two_rows(db):
    """THE REASON THE KEY IS COMPOUND.

    A verdict is a property of an anomaly RUN ON A UNIVERSE. Momentum scores
    differently on today's constituent list than on point-in-time membership --
    that difference is the project's headline finding -- so both rows must
    coexist. Keyed on eval_key alone, the second run silently overwrites the
    first and the survivorship comparison comes back as a single row looking
    perfectly fine.
    """
    db.save(a_score(), universe="current")
    db.save(a_score(realised_sharpe=0.4), universe="point_in_time")

    cur = store.fetch(TEST_KEY, "current", dsn=DSN)
    pit = store.fetch(TEST_KEY, "point_in_time", dsn=DSN)
    assert cur is not None and pit is not None
    assert cur["realised_sharpe"] != pit["realised_sharpe"]


@needs_db
def test_re_running_the_same_anomaly_and_universe_overwrites_in_place(db):
    db.save(a_score(realised_sharpe=0.8))
    db.save(a_score(realised_sharpe=0.9))
    rows = [r for r in store.fetch_all(dsn=DSN) if r["eval_key"] == TEST_KEY]
    assert len(rows) == 1
    assert rows[0]["realised_sharpe"] == pytest.approx(0.9)


@needs_db
def test_failing_and_untestable_verdicts_are_stored_like_any_other(db):
    """A suite that only stored its passes would be a highlight reel, and the
    count of honest failures is a number the demo shows."""
    db.save(a_score(realised_sharpe=-0.8))                 # fails the sign gate
    db.save(a_score(tested=False), universe="point_in_time")
    cur = store.fetch(TEST_KEY, "current", dsn=DSN)
    pit = store.fetch(TEST_KEY, "point_in_time", dsn=DSN)
    assert cur["verdict"] == "fail"
    assert pit["verdict"] == "insufficient_data"


@needs_db
def test_the_reasons_are_stored_not_just_the_verdict(db):
    """Module 7 renders the reasons. A bare count is a number nobody can act on."""
    s = a_score()
    db.save(s)
    row = store.fetch(TEST_KEY, "current", dsn=DSN)
    assert row["reasons"] == list(s.reasons)
    assert len(row["reasons"]) >= 3


@needs_db
def test_every_row_carries_the_registry_sha_and_the_rule_version(db):
    """Together they say "this prediction, judged by this rule". Neither alone
    would reveal that two rows in one table were held to different standards."""
    db.save(a_score())
    row = store.fetch(TEST_KEY, "current", dsn=DSN)
    assert row["registry_sha"] == R.registry_digest()
    assert row["scoring_rule_version"] == a_score().scoring_rule_version


@needs_db
def test_a_row_scored_against_an_edited_registry_is_reported_as_stale(db):
    """`run_evals.py --check` exits non-zero on this rather than warning."""
    db.save(a_score(), sha="0" * 64)
    stale = store.stale_rows(R.registry_digest(), dsn=DSN)
    assert any(r["eval_key"] == TEST_KEY for r in stale)


@needs_db
def test_a_row_scored_against_the_current_registry_is_not_stale(db):
    db.save(a_score())
    stale = store.stale_rows(R.registry_digest(), dsn=DSN)
    assert not any(r["eval_key"] == TEST_KEY for r in stale)


@needs_db
def test_the_caveat_travels_with_the_verdict(db):
    """An anomaly that fails for a reason already known is a different finding
    from one that fails on its merits."""
    db.save(a_score())
    assert "test caveat" in store.fetch(TEST_KEY, "current", dsn=DSN)["caveat"]


@needs_db
def test_save_suite_writes_every_verdict(db):
    scores = [a_score(key=f"zz_test_{i}") for i in range(3)]
    n = store.save_suite(scores, "current", R.registry_digest(), dsn=DSN)
    assert n == 3
    for s in scores:
        assert store.fetch(s.key, "current", dsn=DSN) is not None
        store.delete(s.key, "current", dsn=DSN)


@needs_db
def test_fetch_all_can_filter_by_universe(db):
    db.save(a_score(), universe="point_in_time")
    rows = store.fetch_all(universe="point_in_time", dsn=DSN)
    assert all(r["universe"] == "point_in_time" for r in rows)
    assert any(r["eval_key"] == TEST_KEY for r in rows)


@needs_db
def test_delete_reports_whether_a_row_went(db):
    store.save(a_score(key="zz_test_gone"), "current", R.registry_digest(), dsn=DSN)
    assert store.delete("zz_test_gone", "current", dsn=DSN) is True
    assert store.delete("zz_test_gone", "current", dsn=DSN) is False


@needs_db
def test_counts_groups_by_verdict(db):
    db.save(a_score())
    assert sum(store.counts(dsn=DSN).values()) >= 1


@needs_db
def test_run_at_is_stored_and_comes_back(db):
    when = dt.datetime(2026, 9, 18, 12, 0, tzinfo=dt.timezone.utc)
    store.save(a_score(), "current", R.registry_digest(), run_at=when, dsn=DSN)
    row = store.fetch(TEST_KEY, "current", dsn=DSN)
    store.delete(TEST_KEY, "current", dsn=DSN)
    assert row["run_at"] == when
