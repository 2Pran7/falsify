import type { NoteView } from "@/lib/types";

/** A list row: one question, every run of it.
 *
 * "The same question" is decided by WHAT WAS TESTED, not by how it was worded:
 * "Do stocks that went up last year keep going up?" and "Does 12-month
 * momentum predict returns?" both test mom_12_1, so they are one row. The
 * tested feature comes from the pipeline's verdict card, never from the text.
 * A run that tested nothing (refused before a backtest) falls back to its
 * normalised wording. Every run stays reachable from its row: a record that
 * hides its repeats hides how many tries a result took. */
export interface Question {
  key: string;
  /** The run the row links to: the newest research run that passed, else the
   * newest live run that passed, else the newest run. */
  lead: NoteView;
  /** Every run, newest first. */
  runs: NoteView[];
  first: string;
  last: string;
}

const FAMILIES: [RegExp, string][] = [
  [/^mom_/, "momentum"],
  [/^ivol_/, "idiosyncratic volatility"],
  [/^vol_/, "volatility"],
  [/^rev_36_12/, "long-term reversal"],
  [/^pct_52w_high/, "52-week high"],
  [/^ret_/, "short-term returns"],
  [/^sma_/, "moving average"],
];

/** The feature the run's headline backtest tested, e.g. "vol_63d". */
export function testedFeature(n: NoteView): string | null {
  const v =
    n.verdict?.variant ??
    n.backtests.find((b) => b.statistics && Object.keys(b.statistics).length > 0)?.variant ??
    n.backtests[0]?.variant;
  return v ? v.split(",")[0].trim() : null;
}

/** Plain-English name of what a row tests, or null. */
export function family(n: NoteView): string | null {
  const f = testedFeature(n);
  if (!f) return null;
  return FAMILIES.find(([re]) => re.test(f))?.[1] ?? f;
}

export function questionKey(n: NoteView): string {
  const fam = family(n);
  return fam ? `effect:${fam}` : `text:${n.hypothesis.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()}`;
}

function leadOf(runs: NoteView[]): NoteView {
  return (
    runs.find((r) => r.publishable && r.source !== "live") ??
    runs.find((r) => r.publishable) ??
    runs[0]
  );
}

/** Group, de-duplicating by note id so a run is never counted twice. */
export function groupNotes(items: NoteView[]): Question[] {
  const seen = new Set<string>();
  const byKey = new Map<string, NoteView[]>();
  for (const n of items) {
    if (seen.has(n.note_id)) continue;
    seen.add(n.note_id);
    const k = questionKey(n);
    byKey.set(k, [...(byKey.get(k) ?? []), n]);
  }
  const out: Question[] = [];
  for (const [key, runs] of byKey) {
    runs.sort((a, b) => b.created_at.localeCompare(a.created_at));
    out.push({ key, runs, lead: leadOf(runs), first: runs[runs.length - 1].created_at, last: runs[0].created_at });
  }
  return out.sort((a, b) => b.last.localeCompare(a.last));
}

/** Where a run's page lives: frozen research notes are static pages, live
 * runs are read from the server. */
export function noteHref(n: NoteView): string {
  return n.source === "live" ? `/notes/live/?id=${n.note_id}` : `/notes/${n.note_id}/`;
}

export function otherRuns(items: NoteView[], n: NoteView): NoteView[] {
  const k = questionKey(n);
  return items
    .filter((x) => x.note_id !== n.note_id && questionKey(x) === k)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
}
