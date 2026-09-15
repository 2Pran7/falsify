"""Note -> markdown.

The rendering rule, which is the same rule the storage schema exists to make
possible: THE TABLE COMES FROM THE DATA, THE PROSE COMES FROM THE MODEL, AND
THE PAGE SAYS WHICH IS WHICH. Nothing here parses a number out of the prose.
If the prose and the table disagree, the reader can see both and the table is
the one that was computed.

Two formatting rules are load-bearing rather than cosmetic:

  1. Probabilities and Sharpe ratios never share a column, and the probability
     columns are named for what they are. Run 1 of this agent tabulated 0.65
     raw beside 0.73 "deflated" — impossible for a ratio, ordinary for a
     probability — and the column heading is what caused it.
  2. The trial-variance assumption is printed under every table, not appended
     at the end of the document. Every deflated figure moves with it, and an
     assumption that is not next to the number it governs is not disclosed,
     it is filed.
"""
from __future__ import annotations

from falsify.notes.schema import Note

TRADING_DAYS = 252


def _pct(value: float | None, places: int = 2) -> str:
    return "n/a" if value is None else f"{value * 100:.{places}f}%"


def _num(value: float | None, places: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _banner(note: Note) -> str:
    """The verdict, first, before anyone reads a number."""
    if note.publishable:
        return (
            "> **PUBLISHABLE.** Provenance passed, the run completed on its own, "
            "and the statistics were computed.\n"
        )
    lines = ["> **NOT PUBLISHABLE.** This note is kept and shown as a failure.", ">"]
    lines += [f"> - {r}" for r in note.unpublishable_reasons]
    return "\n".join(lines) + "\n"


def _metrics_table(note: Note) -> str:
    """What the engine measured over the invested window. No statistics here."""
    if not note.backtests:
        return "_No backtest was run._\n"
    head = (
        "| backtest | variant | invested days | total return | CAGR | ann. vol "
        "| Sharpe | max DD |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for b in note.backtests:
        m = b.metrics
        rows.append(
            f"| `{b.handle}` | {b.variant} | {int(m.get('n_invested_days', 0))} "
            f"| {_pct(m.get('total_return'))} | {_pct(m.get('cagr'))} "
            f"| {_pct(m.get('ann_vol'))} | {_num(m.get('sharpe'))} "
            f"| {_pct(m.get('max_drawdown'))} |"
        )
    return head + "\n".join(rows) + "\n"


def _statistics_table(note: Note) -> str:
    """The Module 3 statistics, at the run's final trial count.

    Separate table from the metrics on purpose. Putting a probability in the
    row next to a Sharpe is how a reader ends up comparing them.
    """
    analysed = [b for b in note.backtests if b.analysed]
    if not analysed:
        failed = [b for b in note.backtests if b.analysis_error]
        if failed:
            lines = [f"- `{b.handle}`: {b.analysis_error}" for b in failed]
            return "_The statistics could not be computed._\n\n" + "\n".join(lines) + "\n"
        return "_No statistics were computed for this run._\n"

    n = note.assumptions.get("n_trials_for_deflation", note.run.n_backtests)
    head = (
        f"All rows deflated at **N = {n}** trials.\n\n"
        "| backtest | P(SR &gt; 0) | P(beats best of N) | MinTRL (days) "
        "| MinTRL (years) | observations |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for b in analysed:
        s = b.statistics
        trl = s.get("min_track_record_length_days")
        rows.append(
            f"| `{b.handle}` | {_num(s.get('prob_sharpe_above_zero'), 3)} "
            f"| {_num(s.get('prob_beats_best_of_n_trials'), 3)} "
            f"| {_num(trl, 0)} "
            f"| {'n/a' if trl is None else f'{trl / TRADING_DAYS:.1f}'} "
            f"| {int(s.get('n_observations', 0))} |"
        )

    tv = note.assumptions.get("trial_variance")
    footer = (
        "\n\nBoth `P(` columns are **probabilities in [0, 1], not Sharpe ratios**. "
        "`P(beats best of N)` is the deflated Sharpe result: below 0.50 the "
        "evidence does not survive the number of strategies tried. `MinTRL` is "
        "how much data would be needed before the Sharpe could be told apart "
        "from zero at 95% — compare it to the invested days above.\n"
    )
    if tv is not None:
        footer += (
            f"\n**Assumption:** the deflated figures use a trial-Sharpe variance "
            f"of `{tv}`, which is assumed, not measured. Every number in this "
            f"table moves with it.\n"
        )
    return head + "\n".join(rows) + footer


def _provenance_section(note: Note) -> str:
    p = note.provenance
    verdict = "PASS" if p.ok else "FAIL"
    out = (
        f"**{verdict}** — {p.verified}/{p.checked} numerals in the commentary "
        f"trace to a tool result ({p.coverage:.0%}); {p.exempt} trivial exempt.\n"
    )
    if p.unverified:
        out += "\nWith no source in any tool output:\n\n"
        out += "\n".join(f"- {u}" for u in p.unverified) + "\n"
    out += (
        "\nProvenance checks **numbers, not claims**. A note in which every "
        "figure is traceable and the conclusion is wrong passes this check "
        "completely. It is a floor, not a proof.\n"
    )
    return out


def _run_section(note: Note) -> str:
    r = note.run
    seq = " → ".join(r.tool_sequence) if r.tool_sequence else "(no tools called)"
    lines = [
        f"- **model:** `{r.model}`",
        f"- **stop reason:** `{r.stop_reason}`"
        + ("" if r.completed else "  ← a guard fired; the answer is partial"),
        f"- **turns:** {r.turns}",
        f"- **tool sequence:** {seq}",
        f"- **tool errors:** {r.n_tool_errors}",
        f"- **backtests run (the honest trial count):** {r.n_backtests}",
        f"- **tokens:** {r.total_tokens:,}",
        f"- **cost:** ${r.cost_usd:.4f}",
    ]
    return "\n".join(lines) + "\n"


def _assumptions_section(note: Note) -> str:
    if not note.assumptions:
        return "_None recorded._\n"
    return "\n".join(f"- `{k}`: {v}" for k, v in sorted(note.assumptions.items())) + "\n"


def to_markdown(note: Note) -> str:
    """The full note.

    Order is deliberate. Verdict, then what the pipeline computed, then what
    the model said about it, then the checks. A reader who stops after the
    first table has the numbers; a reader who stops after the prose has been
    told, twice already, that the prose is commentary.
    """
    created = note.created_at.strftime("%d %B %Y, %H:%M UTC")
    parts = [
        f"# {note.hypothesis}",
        "",
        _banner(note),
        f"_Run {note.note_id}"
        + (f" · eval case `{note.eval_key}`" if note.eval_key else "")
        + f" · {created}_",
        "",
        "## What the pipeline computed",
        "",
        _metrics_table(note),
        "",
        "### Statistics",
        "",
        _statistics_table(note),
        "",
        "## The agent's commentary",
        "",
        "_Written by the model. Every number in it is checked against the tool "
        "results below, but the reasoning is not checked by anything. The "
        "tables above are the pipeline's; this section is the model's._",
        "",
        note.prose.strip() or "_(the model returned no text)_",
        "",
        "## Numeric provenance",
        "",
        _provenance_section(note),
        "",
        "## Assumptions in force",
        "",
        _assumptions_section(note),
        "",
        "## The run",
        "",
        _run_section(note),
    ]
    return "\n".join(parts).rstrip() + "\n"


def to_summary_line(note: Note) -> str:
    """One line, for a list of notes or a terminal.

    Leads with the verdict because that is the field to read first.
    """
    mark = "PUBLISHABLE  " if note.publishable else "NOT PUBLISHABLE"
    best = max(
        (b.deflated_probability for b in note.backtests if b.analysed),
        default=None,
    )
    tail = "no statistics" if best is None else f"best P(beats N) {best:.3f}"
    return (
        f"{mark}  {note.hypothesis[:48]:<48}  "
        f"{note.run.n_backtests} backtest(s), {tail}, ${note.run.cost_usd:.4f}"
    )


__all__ = ["to_markdown", "to_summary_line"]
