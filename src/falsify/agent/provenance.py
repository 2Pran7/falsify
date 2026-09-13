"""Numeric provenance: every number in the note must trace to a tool result.

The project's claim is "the LLM decides, never computes". Three mechanisms
already make that hard to violate: the menu is closed, results cross as
summaries, and the payloads never reach the model. This module makes it
CHECKABLE, which is a different and stronger thing.

The rule, in one sentence:

    A number in the model's prose is verified if it appears in, or is a
    permitted transform of, a number the model was actually shown.

"Was actually shown" is the whole idea. Rather than guessing which figures
matter, the check harvests every numeric literal out of every tool_result the
run produced, and demands that each numeral in the note be reachable from that
set. A Sharpe the model invented, or half-remembered from its training data,
or averaged in its head, is not in the set and is reported.

PERMITTED TRANSFORMS, and why each exists. A strict equality check fails
immediately and uselessly, because prose is not JSON: the tool returns
`0.196938` and the note says `19.7%`. So a small, closed list of
presentation-level conversions is allowed:

    identity        0.65        -> 0.65
    percent         0.196938    -> 19.7    (x100, the common one)
    rounding        1654.6      -> 1655
    days to years   1655        -> 6.6     (/252, trading days)
    sign            -0.3187     -> 31.9    (drawdowns quoted as magnitudes)

Each is a way of WRITING a number the pipeline produced, not a way of computing
a new one. The list is deliberately short and deliberately closed: every
transform added widens what counts as verified, so adding one should feel like
a concession rather than a convenience. Ratios, differences and sums are NOT
permitted, because those are arithmetic, and arithmetic is what the model is
not supposed to be doing.

WHAT THIS CANNOT DO, stated plainly because the limitation is the interesting
part. Provenance checks numbers, not claims. A note that reports every figure
correctly and draws a wrong conclusion from them passes this check completely.
The first real run did exactly that: every number was traceable, and the model
still put a probability in a column labelled "Deflated Sharpe" and reasoned
about the comparison. Provenance is a floor, not a proof, and the honest way to
describe it is "no fabricated numbers", never "the note is correct".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Numbers as they appear in prose: optional sign, digits with optional commas as
# thousands separators, optional decimal part, optional exponent.
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")

# Relative tolerance for a match. Loose enough to absorb a figure quoted to one
# decimal place, tight enough that two genuinely different Sharpes cannot be
# confused: 0.65 and 0.66 are 1.5% apart and stay distinguishable.
TOLERANCE = 0.005

TRADING_DAYS = 252

# Numerals that carry no numeric claim and would otherwise generate noise: list
# markers, section numbers, and the small integers that appear in ordinary
# prose ("all 3 variants", "the top 10"). Restricting the exemption to integers
# at or below this value keeps it from ever covering a Sharpe, a probability or
# a return.
TRIVIAL_INTEGER_MAX = 12


@dataclass
class Unverified:
    """One numeral in the note with no source in any tool result."""

    literal: str
    value: float
    context: str

    def __str__(self) -> str:
        return f"{self.literal!r} in: ...{self.context}..."


@dataclass
class ProvenanceReport:
    """The verdict on one research note."""

    checked: int = 0
    verified: int = 0
    unverified: list[Unverified] = field(default_factory=list)
    exempt: int = 0

    @property
    def ok(self) -> bool:
        """True only if every non-trivial numeral traced to a tool result."""
        return not self.unverified

    @property
    def coverage(self) -> float:
        return 1.0 if self.checked == 0 else self.verified / self.checked

    def summary(self) -> str:
        head = (
            f"{self.verified}/{self.checked} numerals traced to tool output "
            f"({self.coverage:.0%}), {self.exempt} trivial exempt"
        )
        if self.ok:
            return f"PASS: {head}"
        lines = "\n".join(f"    {u}" for u in self.unverified)
        return f"FAIL: {head}\n  {len(self.unverified)} with no source:\n{lines}"


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """Every numeric literal in a string, as (literal, value).

    Commas are stripped before parsing so "1,655" reads as 1655. A literal that
    does not parse is skipped rather than raising: this runs over model prose,
    and prose contains things like "12-1" that are names, not numbers.
    """
    out: list[tuple[str, float]] = []
    for match in _NUMBER.finditer(text):
        literal = match.group()
        try:
            out.append((literal, float(literal.replace(",", ""))))
        except ValueError:
            continue
    return out


def _variants(value: float) -> set[float]:
    """Every permitted way of writing one pipeline number.

    Kept small on purpose. Each entry is a presentation convention, not a
    calculation: see the module docstring for why that distinction is the whole
    point of the check.
    """
    base = {value, -value}
    scaled: set[float] = set()
    for v in base:
        scaled.update({v, v * 100.0, v / 100.0, v / TRADING_DAYS})
    out: set[float] = set()
    for v in scaled:
        out.add(v)
        for places in range(5):
            out.add(round(v, places))
    return out


def allowed_values(tool_results: Iterable[str]) -> set[float]:
    """Every number the model was shown, plus its permitted transforms.

    Args:
        tool_results: the raw content string of each tool_result sent back to
            the model, successes and errors alike.

    Returns:
        The set a note's numerals are checked against.

    Errors are included deliberately: an error message may quote a handle or a
    count, the model may legitimately repeat it, and excluding them would
    produce false alarms on a run that recovered from a mistake.
    """
    allowed: set[float] = set()
    for blob in tool_results:
        for _, value in extract_numbers(blob):
            allowed |= _variants(value)
    return allowed


def _matches(value: float, allowed: set[float]) -> bool:
    if value in allowed:
        return True
    for candidate in allowed:
        if abs(value - candidate) <= TOLERANCE * max(1.0, abs(candidate)):
            return True
    return False


def check_note(
    note: str, tool_results: Iterable[str], context_chars: int = 40
) -> ProvenanceReport:
    """Verify every numeral in a research note against what the tools returned.

    Args:
        note: the model's final prose.
        tool_results: content strings of the tool_results it received.
        context_chars: how much surrounding text to quote for a failure.

    Returns:
        ProvenanceReport. `ok` is True only when nothing was fabricated.

    A failure is not necessarily dishonesty: the commonest cause is the model
    doing arithmetic it was told not to do, such as quoting a difference
    between two figures it was given. That is still a violation, because a
    number the pipeline never produced cannot be checked by anyone reading the
    note.
    """
    allowed = allowed_values(tool_results)
    report = ProvenanceReport()

    for match in _NUMBER.finditer(note):
        literal = match.group()
        try:
            value = float(literal.replace(",", ""))
        except ValueError:
            continue

        if value.is_integer() and abs(value) <= TRIVIAL_INTEGER_MAX:
            report.exempt += 1
            continue

        report.checked += 1
        if _matches(value, allowed):
            report.verified += 1
        else:
            start = max(0, match.start() - context_chars)
            end = min(len(note), match.end() + context_chars)
            report.unverified.append(
                Unverified(
                    literal=literal,
                    value=value,
                    context=note[start:end].replace("\n", " "),
                )
            )
    return report


def check_run(result: Any) -> ProvenanceReport:
    """Convenience wrapper: check a RunResult's answer against its own transcript.

    Sources are every tool_result the run produced PLUS the tool schemas.

    The schemas belong in the set because the model was shown them, and they
    contain real figures it is entitled to repeat: `compute_feature` tells it
    that mom_12_1 needs 252 days of history, and a note that mentions the
    252-day lookback is quoting the system, not inventing a number. Checking a
    real run without them flagged exactly that, and a checker that cries wolf
    on a correct note is one that gets switched off.
    """
    from falsify.agent.tools import tool_schemas  # local: avoids a cycle

    blobs: list[str] = [json.dumps(tool_schemas())]
    for message in result.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blobs.append(str(block.get("content", "")))
    return check_note(result.answer, blobs)
