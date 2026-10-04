import type { NoteView } from "@/lib/types";

/** One question, every run of it. The list shows each question once; every
 * run keeps its own page, reachable from the question's page, because a
 * record that hides its repeats hides how many tries a result took. */
export interface Question {
  key: string;
  /** The run the list links to: the newest that passed every check, else the newest. */
  lead: NoteView;
  /** Every run, newest first, the lead included. */
  runs: NoteView[];
}

/** Same question asked twice reads the same after this: case, spacing and
 * trailing punctuation do not make a new question. */
export function questionKey(h: string): string {
  return h.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

export function groupNotes(items: NoteView[]): Question[] {
  const byKey = new Map<string, NoteView[]>();
  for (const n of items) {
    const k = questionKey(n.hypothesis);
    byKey.set(k, [...(byKey.get(k) ?? []), n]);
  }
  const out: Question[] = [];
  for (const [key, runs] of byKey) {
    runs.sort((a, b) => b.created_at.localeCompare(a.created_at));
    out.push({ key, runs, lead: runs.find((r) => r.publishable) ?? runs[0] });
  }
  return out.sort((a, b) => b.lead.created_at.localeCompare(a.lead.created_at));
}

export function otherRuns(items: NoteView[], n: NoteView): NoteView[] {
  const k = questionKey(n.hypothesis);
  return items
    .filter((x) => x.note_id !== n.note_id && questionKey(x.hypothesis) === k)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
}
