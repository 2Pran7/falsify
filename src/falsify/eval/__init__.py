"""The eval suite (Module 6): six published anomalies, pre-registered and scored.

registry  the pre-registration -- six anomalies, frozen and hashed
score     pass / partial / fail / insufficient_data, with reasons
runner    the suite, run through the same tools the agent uses
store     verdicts in Postgres, keyed by (anomaly, universe)

The project's central claim is that this pipeline rediscovers published effects
and says so honestly when it does not. This package is what makes that a
checkable table rather than a sentence in a README.
"""
from falsify.eval.registry import ANOMALIES, Anomaly, registry_digest
from falsify.eval.score import Measurement, Score, SuiteScore, score_suite

__all__ = [
    "ANOMALIES",
    "Anomaly",
    "Measurement",
    "Score",
    "SuiteScore",
    "registry_digest",
    "score_suite",
]
