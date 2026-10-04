"use client";

import Link from "next/link";
import { Fragment, useEffect, useState } from "react";
import { Chip } from "@/components/Chip";
import { API_URL } from "@/lib/format";
import { family, groupNotes, noteHref } from "@/lib/notes";
import type { NoteView } from "@/lib/types";

const day = (iso: string) => iso.slice(0, 10);

/** The notes list. Frozen research notes render at build time; questions asked
 * on the live page are fetched from the server and merged in, one row per
 * tested effect, so a repeat adds a run to its row instead of a new row. */
export function NotesTable({ items, limit }: { items: NoteView[]; limit?: number }) {
  const [live, setLive] = useState<NoteView[]>([]);
  const [state, setState] = useState<"idle" | "loading" | "done" | "offline">(API_URL ? "loading" : "idle");
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    if (!API_URL) return;
    let stop = false;
    (async () => {
      // The free server sleeps; a first request can take about a minute.
      for (let i = 0; i < 4 && !stop; i++) {
        try {
          const ctl = new AbortController();
          const t = setTimeout(() => ctl.abort(), 30_000);
          const r = await fetch(`${API_URL}/notes/live`, { signal: ctl.signal });
          clearTimeout(t);
          if (r.ok) {
            const body = await r.json();
            if (!stop) { setLive(body.notes ?? []); setState("done"); }
            return;
          }
        } catch {}
        await new Promise((res) => setTimeout(res, 5_000));
      }
      if (!stop) setState("offline");
    })();
    return () => { stop = true; };
  }, []);

  const questions = groupNotes([...items, ...live]);
  const shown = limit ? questions.slice(0, limit) : questions;
  const runs = questions.reduce((a, q) => a + q.runs.length, 0);

  return (
    <>
      <p className="small muted" style={{ margin: "0 0 10px" }}>
        {questions.length} questions from {runs} runs.{" "}
        {state === "loading" && "Loading questions asked on the live page…"}
        {state === "done" && `${live.length} of the runs were asked on the live page.`}
        {state === "offline" && "The live server didn't answer, so questions asked there aren't shown right now."}
      </p>
      <div className="table-card scroll">
        <table>
          <thead><tr><th>Question</th><th>Verdict</th><th>Runs</th><th>First asked</th><th>Last asked</th><th>Status</th></tr></thead>
          <tbody>
            {shown.map((q) => {
              const { key, lead } = q;
              const fam = family(lead);
              const isOpen = open === key;
              return (
                <Fragment key={key}>
                  <tr>
                    <td>
                      <Link href={noteHref(lead)} className="cell-title">{lead.hypothesis}</Link>
                      <div className="cell-sub">
                        {fam && <>Tests {fam}</>}
                        {lead.source === "live" && <>{fam ? " · " : ""}asked on the live page, latest ~3 years</>}
                        {!lead.publishable && <>{fam || lead.source === "live" ? " · " : ""}{lead.unpublishable_reasons[0]}</>}
                      </div>
                    </td>
                    <td>{lead.verdict ? <Chip kind={lead.verdict.outcome} /> : <span className="muted small">n/a</span>}</td>
                    <td className="num">
                      {q.runs.length > 1 ? (
                        <button className="linkish" onClick={() => setOpen(isOpen ? null : key)} aria-expanded={isOpen}>
                          {q.runs.length} {isOpen ? "▴" : "▾"}
                        </button>
                      ) : 1}
                    </td>
                    <td className="num">{day(q.first)}</td>
                    <td className="num">{day(q.last)}</td>
                    <td><Chip kind={lead.publishable ? "publishable" : "refused"} /></td>
                  </tr>
                  {isOpen && q.runs.map((r) => (
                    <tr key={r.note_id} className="subrow">
                      <td>
                        <Link href={noteHref(r)}>{r.hypothesis}</Link>
                        <span className="muted small"> · {r.source === "live" ? "live" : "research"}{r.note_id === lead.note_id ? " · shown above" : ""}</span>
                      </td>
                      <td>{r.verdict ? <Chip kind={r.verdict.outcome} /> : <span className="muted small">n/a</span>}</td>
                      <td />
                      <td className="num" colSpan={2}>{r.created_at.slice(0, 16).replace("T", " ")} UTC</td>
                      <td><Chip kind={r.publishable ? "publishable" : "refused"} /></td>
                    </tr>
                  ))}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
