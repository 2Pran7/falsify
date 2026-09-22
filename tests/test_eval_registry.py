"""The pre-registration must be a receipt, not a comment.

The two tests that carry this file are the pair at the bottom:

    test_every_predicted_field_moves_the_digest
    test_no_documentation_field_moves_the_digest

Together they are what makes "we did not edit the expectation after seeing the
result" a string comparison. The first fails if a prediction could be changed
without the hash noticing, which would let the registry be quietly rewritten
around an inconvenient result. The second fails if a typo fix in a citation
would invalidate every stored verdict, which is how documentation rots: nobody
fixes a comment that costs them a re-run.
"""
from __future__ import annotations

import dataclasses

import pytest

from falsify.agent import tools as T
from falsify.eval import registry as R


def _anom(**over) -> R.Anomaly:
    base = dict(
        key="x", feature="mom_12_1", direction=1, published_sharpe=0.5,
        min_history_days=252, citation="c", hypothesis="h", sharpe_source="s",
        caveat="cv",
    )
    return R.Anomaly(**{**base, **over})


# --- the suite is what it says it is ---------------------------------------


def test_six_anomalies_with_unique_keys():
    assert len(R.ANOMALIES) == 6
    assert len({a.key for a in R.ANOMALIES}) == 6


def test_every_feature_is_on_the_closed_tool_menu():
    """A registry entry naming a feature the agent cannot compute is unrunnable.

    The failure would otherwise surface as a ToolError mid-suite, recorded as
    `insufficient_data`, and look exactly like a short panel.
    """
    for a in R.ANOMALIES:
        assert a.feature in T.FEATURES, f"{a.key} names unknown feature {a.feature}"


def test_most_of_the_suite_predicts_a_negative_spread():
    """The reason `direction` exists at all.

    Four of the six -- both volatility anomalies and both reversals -- predict
    that the BOTTOM bucket wins. If this ever became six positives the field
    would look redundant, someone would remove it, and every reversal and
    low-vol result would then be scored as a failure precisely when it worked.
    """
    negatives = {a.key for a in R.ANOMALIES if a.direction == -1}
    assert negatives == {
        "short_term_reversal",
        "long_term_reversal",
        "low_volatility",
        "idiosyncratic_volatility",
    }


def test_direction_is_only_ever_plus_or_minus_one():
    with pytest.raises(ValueError, match="direction"):
        _anom(direction=0)


def test_published_sharpe_is_the_oriented_reference_and_is_positive():
    """The sign lives in `direction`, never in the reference level.

    A negative published_sharpe would make the 3x embarrassment ratio negative
    and silently disable the downgrade.
    """
    with pytest.raises(ValueError, match="published_sharpe"):
        _anom(published_sharpe=-0.5)
    assert all(a.published_sharpe > 0 for a in R.ANOMALIES)


def test_min_history_days_is_positive():
    with pytest.raises(ValueError, match="min_history_days"):
        _anom(min_history_days=0)


def test_long_term_reversal_needs_more_than_two_years():
    """Pre-registered anyway, and expected to report insufficient_data.

    A suite that quietly dropped the one anomaly its panel cannot test would be
    hiding a known limitation behind a full-looking table.
    """
    assert R.get("long_term_reversal").min_history_days > 504


def test_every_anomaly_documents_its_caveat_and_sharpe_source():
    """Without sharpe_source the reference numbers are folklore with a decimal
    point; without a caveat, a failure for a known reason is indistinguishable
    from a failure on the merits."""
    for a in R.ANOMALIES:
        assert len(a.sharpe_source) > 20, a.key
        assert len(a.caveat) > 20, a.key
        assert len(a.hypothesis) > 20, a.key
        assert a.citation.strip(), a.key


def test_anomalies_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        R.ANOMALIES[0].direction = -1


def test_get_names_the_valid_keys_when_it_fails():
    with pytest.raises(KeyError, match="momentum_12_1"):
        R.get("momentum")


# --- the digest -------------------------------------------------------------


def test_digest_is_stable_across_calls():
    assert R.registry_digest() == R.registry_digest()


def test_digest_is_a_sha256_hex_string():
    d = R.registry_digest()
    assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)


def test_digest_does_not_depend_on_tuple_order():
    """Reordering the registry must not report a changed pre-registration.

    A digest that moved when someone alphabetised the file would cry wolf, and
    a check that cries wolf gets switched off -- the Module 4 provenance lesson,
    applied to the hash.
    """
    a, b = R.ANOMALIES[:2]
    reordered = (b, a, *R.ANOMALIES[2:])
    assert R.registry_digest(reordered) == R.registry_digest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("key", "renamed"),
        ("feature", "vol_21d"),
        ("direction", -1),
        ("published_sharpe", 0.51),
        ("min_history_days", 253),
    ],
)
def test_every_predicted_field_moves_the_digest(field, value):
    """THE TEST THE PRE-REGISTRATION RESTS ON.

    Every field in PREREGISTERED_FIELDS must be load-bearing. If one could be
    edited without the hash noticing, the registry could be rewritten around an
    inconvenient result and every stored verdict would still carry a matching
    sha.
    """
    original = R.ANOMALIES[0]
    assert getattr(original, field) != value, "parametrised value must differ"
    mutated = (dataclasses.replace(original, **{field: value}), *R.ANOMALIES[1:])
    assert R.registry_digest(mutated) != R.registry_digest()


@pytest.mark.parametrize(
    "field", ["citation", "hypothesis", "sharpe_source", "caveat"]
)
def test_no_documentation_field_moves_the_digest(field):
    """The mirror image, and it matters just as much.

    A typo fixed in a citation must not invalidate every result already stored
    against it. If it did, nobody would ever fix one.
    """
    mutated = (
        dataclasses.replace(R.ANOMALIES[0], **{field: "rewritten entirely"}),
        *R.ANOMALIES[1:],
    )
    assert R.registry_digest(mutated) == R.registry_digest()


def test_preregistered_fields_are_exactly_the_predictions():
    """Guards against a documentation field drifting into the hashed set."""
    assert set(R.PREREGISTERED_FIELDS) == {
        "key", "feature", "direction", "published_sharpe", "min_history_days"
    }


def test_as_records_round_trips_every_field():
    recs = R.as_records()
    assert len(recs) == 6
    assert set(recs[0]) == {f.name for f in dataclasses.fields(R.Anomaly)}
