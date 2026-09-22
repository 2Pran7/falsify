"""The pre-registration: six published anomalies, frozen before the runs and hashed.

AN EVAL SUITE CAN BE BUILT TWO WAYS, and nothing in the output reveals which
one you chose.

The wrong way is to run six strategies, look at the numbers, and write down
which ones "worked". That is not an eval suite, it is a narrative with a table
in it, and every row still looks like a result.

The way this is built: four things are fixed for each anomaly IN ADVANCE — the
feature, the expected DIRECTION, the published Sharpe it is compared against,
and the history it needs before it may be scored at all — and `registry_digest`
hashes exactly those fields. Every stored verdict carries the hash. "We did not
edit the expectation after seeing the result" therefore becomes a string
comparison rather than a promise.

WHAT IS DELIBERATELY NOT HASHED. `PREREGISTERED_FIELDS` excludes every prose
field: the citation, the hypothesis text, the provenance of the reference
Sharpe, the caveat. A typo fixed in a citation must not invalidate every result
already stored against it, because if it did, nobody would ever fix one and the
documentation would rot in place. There is a test for each direction: every
predicted field must move the hash, and no documentation field may.

THE PUBLISHED SHARPES ARE REFERENCE LEVELS, NOT TARGETS. Most of these papers
report monthly excess returns or t-statistics and no Sharpe at all, so every
entry records where its number came from in `sharpe_source`. Without that field
the values would be folklore with a decimal point.

EVERY ANOMALY RECORDS A CAVEAT: the known gap between this implementation and
the published one. An anomaly that fails for a reason already known is a
different finding from one that fails on its merits, and the results table
prints the caveat beside the verdict so the two cannot be confused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

# Only these fields enter the digest. Order here is irrelevant: the hash is
# taken over a canonical, key-sorted JSON encoding of a list sorted by `key`,
# so neither dict ordering nor the tuple order of ANOMALIES can move it.
PREREGISTERED_FIELDS: tuple[str, ...] = (
    "key",
    "feature",
    "direction",
    "published_sharpe",
    "min_history_days",
)


@dataclass(frozen=True)
class Anomaly:
    """One pre-registered prediction.

    key:              slug, and the primary key of every stored verdict.
    feature:          a key of `agent.tools.FEATURES`. Validated by test.
    direction:        +1 if the paper predicts the TOP bucket wins, -1 if the
                      BOTTOM does. See the module note below; this is the
                      load-bearing field.
    published_sharpe: annualised reference level, NOT a target.
    min_history_days: trading days of per-ticker history the feature needs
                      before it produces a value. Below this the anomaly scores
                      `insufficient_data`, which is not a failure.

    The rest is documentation and is excluded from the digest.
    """

    key: str
    feature: str
    direction: int
    published_sharpe: float
    min_history_days: int
    citation: str
    hypothesis: str
    sharpe_source: str
    caveat: str

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(
                f"{self.key}: direction must be +1 or -1, got {self.direction}. "
                "There is no such thing as a pre-registered prediction of no sign."
            )
        if self.published_sharpe <= 0:
            raise ValueError(
                f"{self.key}: published_sharpe is the ORIENTED reference level and is "
                f"always positive; the sign lives in `direction`. Got "
                f"{self.published_sharpe}."
            )
        if self.min_history_days < 1:
            raise ValueError(f"{self.key}: min_history_days must be >= 1")


# DIRECTION IS THE LOAD-BEARING FIELD, and it is worth being explicit about why.
#
# `run_backtest` always goes long the top bucket and short the bottom, so the
# Sharpe it returns is the sign of the top-minus-bottom spread. FOUR OF THESE
# SIX PREDICT A NEGATIVE SPREAD: low volatility, idiosyncratic volatility,
# short-term reversal and long-term reversal all say the BOTTOM bucket wins.
# The majority of the suite, in other words. Without a sign fixed in
# advance, "the spread was -0.6" is unscoreable, and the temptation is to look
# at the number and then decide which way the paper said it should run. That is
# exactly the self-deception this project is named after refusing.
ANOMALIES: tuple[Anomaly, ...] = (
    Anomaly(
        key="momentum_12_1",
        feature="mom_12_1",
        direction=1,
        published_sharpe=0.50,
        min_history_days=252,
        citation="Jegadeesh & Titman (1993), Journal of Finance 48(1)",
        hypothesis=(
            "Stocks with high returns over the past twelve months, skipping the most "
            "recent month, continue to outperform stocks with low past returns over "
            "the following month."
        ),
        sharpe_source=(
            "The paper reports monthly excess returns and t-statistics, not a Sharpe. "
            "0.50 is the widely quoted long-run annualised Sharpe of the UMD factor."
        ),
        caveat=(
            "Prices here are split-adjusted only, never dividend-adjusted, so every "
            "return is a price return. The Ken French UMD series is a total-return "
            "construction and is not like-for-like."
        ),
    ),
    Anomaly(
        key="short_term_reversal",
        feature="ret_21d",
        direction=-1,
        published_sharpe=0.35,
        min_history_days=21,
        citation="Jegadeesh (1990), Journal of Finance 45(3)",
        hypothesis=(
            "Stocks with the highest returns over the past month underperform those "
            "with the lowest over the following month: the one-month effect runs "
            "opposite to twelve-month momentum."
        ),
        sharpe_source=(
            "Reference level for the short-term reversal factor. The published effect "
            "is reported as a monthly return spread."
        ),
        caveat=(
            "The published effect is largely a bid-ask bounce and microstructure "
            "phenomenon, and is the one anomaly here most likely to be destroyed by "
            "realistic trading costs. It rebalances monthly at 10bps one-way; the "
            "literature suggests that is optimistic for this signal specifically."
        ),
    ),
    Anomaly(
        key="long_term_reversal",
        feature="rev_36_12",
        direction=-1,
        published_sharpe=0.20,
        min_history_days=756,
        citation="De Bondt & Thaler (1985), Journal of Finance 40(3)",
        hypothesis=(
            "Stocks with the highest returns over the past three years, skipping the "
            "most recent year, underperform over the following period: extreme past "
            "performance mean-reverts at long horizons."
        ),
        sharpe_source=(
            "The paper reports cumulative abnormal returns over 36-month holding "
            "periods. 0.20 is a conservative annualised reference level."
        ),
        caveat=(
            "The published test uses 36-month FORMATION and 36-month HOLDING periods. "
            "This holds for one month and rebalances monthly, which is a much higher "
            "turnover implementation of the same sort. A weaker result is expected "
            "even where the effect is real."
        ),
    ),
    Anomaly(
        key="low_volatility",
        feature="vol_63d",
        direction=-1,
        published_sharpe=0.78,
        min_history_days=63,
        citation="Ang, Hodrick, Xing & Zhang (2006), Journal of Finance 61(1)",
        hypothesis=(
            "Low-volatility stocks earn higher risk-adjusted returns than "
            "high-volatility stocks: the security market line is flatter than theory "
            "predicts, and at the extreme it inverts."
        ),
        sharpe_source=(
            "Reference level taken from the betting-against-beta literature "
            "(Frazzini & Pedersen 2014), which reports the highest Sharpe of any "
            "anomaly in this suite."
        ),
        caveat=(
            "THIS SORTS ON TOTAL VOLATILITY; Frazzini-Pedersen BAB sorts on BETA and "
            "is beta-neutralised by construction. A related effect, not the same one, "
            "and the 0.78 reference belongs to the version this does not implement."
        ),
    ),
    Anomaly(
        key="idiosyncratic_volatility",
        feature="ivol_63d",
        direction=-1,
        published_sharpe=0.60,
        min_history_days=63,
        citation="Ang, Hodrick, Xing & Zhang (2006), Journal of Finance 61(1)",
        hypothesis=(
            "Stocks with high idiosyncratic volatility relative to a market model "
            "earn abnormally LOW average returns: the puzzle is that the sign is "
            "backwards from what bearing unpriced risk should pay."
        ),
        sharpe_source=(
            "Reference level for the published IVOL spread, reported in the paper as "
            "a monthly alpha rather than a Sharpe."
        ),
        caveat=(
            "The residual here comes from a MARKET-ONLY regression on SPY with no "
            "intercept. The published residual is from three factors, which are not "
            "in this database. A stated simplification beats an invented factor "
            "series. If SPY is absent the feature is null everywhere rather than "
            "falling back to total volatility, which would score low_volatility twice "
            "under two names and call the second one independent evidence."
        ),
    ),
    Anomaly(
        key="fifty_two_week_high",
        feature="pct_52w_high",
        direction=1,
        published_sharpe=0.55,
        min_history_days=252,
        citation="George & Hwang (2004), Journal of Finance 59(5)",
        hypothesis=(
            "Stocks trading close to their 52-week high outperform those trading far "
            "below it, and this proximity predicts returns better than the past "
            "return itself does."
        ),
        sharpe_source=(
            "Reference level for the nearness-to-high strategy; the paper reports "
            "monthly return spreads."
        ),
        caveat=(
            "Strongly correlated with momentum by construction — a stock near its "
            "52-week high generally got there by rising — so this is NOT independent "
            "evidence of a separate effect. Both are in the suite, and both are "
            "counted in the trial count and the multiplicity correction, which is the "
            "honest way to carry a correlated pair."
        ),
    ),
)

ANOMALIES_BY_KEY: dict[str, Anomaly] = {a.key: a for a in ANOMALIES}


def registry_digest(anomalies: tuple[Anomaly, ...] = ANOMALIES) -> str:
    """SHA-256 over the PREDICTED fields only. The pre-registration receipt.

    Args:
        anomalies: the registry to hash. Defaults to the frozen suite.

    Returns:
        A 64-character hex digest.

    Sorted by `key` before encoding, and each record encoded with sorted keys,
    so the hash is a function of the predictions and nothing else. Reordering
    the tuple must not move it: a registry that hashed its own tuple order
    would report a changed pre-registration every time someone tidied the file,
    and a digest that cries wolf gets ignored — the same lesson the provenance
    checker taught at Module 4.
    """
    records = [
        {f: getattr(a, f) for f in PREREGISTERED_FIELDS}
        for a in sorted(anomalies, key=lambda x: x.key)
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def as_records(anomalies: tuple[Anomaly, ...] = ANOMALIES) -> list[dict]:
    """The registry as plain dicts, for printing and for JSONB storage."""
    return [asdict(a) for a in sorted(anomalies, key=lambda x: x.key)]


def get(key: str) -> Anomaly:
    """One anomaly by key.

    Raises:
        KeyError: with the valid keys in the message.
    """
    try:
        return ANOMALIES_BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown anomaly {key!r}. Registered: {', '.join(sorted(ANOMALIES_BY_KEY))}"
        ) from None


__all__ = [
    "ANOMALIES",
    "ANOMALIES_BY_KEY",
    "PREREGISTERED_FIELDS",
    "Anomaly",
    "as_records",
    "get",
    "registry_digest",
]
