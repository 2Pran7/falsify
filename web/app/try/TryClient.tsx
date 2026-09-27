"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Chip } from "@/components/Chip";
import { NoteBody } from "@/components/NoteBody";
import { API_URL } from "@/lib/format";
import type { NoteView } from "@/lib/types";

type Run = { run_id: string; status: "queued" | "running" | "done" | "failed"; hypothesis: string; note: NoteView | null; error: string | null };

const EXAMPLES = [
  "Do stocks that went up over the last year keep going up?",
  "Do low-volatility stocks outperform high-volatility stocks?",
  "Do last month's losers bounce back this month?",
  "Do stocks near their 52-week high keep outperforming?",
];
const STEPS: [Run["status"], string][] = [["queued", "Queued"], ["running", "Running the agent"], ["done", "Note written"]];

export function TryClient() {
  const [text, setText] = useState("");
  const [name, setName] = useState("");
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [left, setLeft] = useState<number | null>(null);
  const [waking, setWaking] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!API_URL) return;
    const t = setTimeout(() => setWaking(true), 2500);
    fetch(`${API_URL}/health`).then((r) => r.json()).then((d) => setLeft(d.runs_left_for_you_today)).catch(() => {}).finally(() => { clearTimeout(t); setWaking(false); });
    return () => { clearTimeout(t); if (timer.current) clearTimeout(timer.current); };
  }, []);

  async function poll(id: string) {
    try {
      const r = await fetch(`${API_URL}/runs/${id}`);
      const d: Run = await r.json();
      setRun(d);
      if (d.status === "queued" || d.status === "running") timer.current = setTimeout(() => poll(id), 2500);
      else { setBusy(false); setLeft((x) => (x === null ? x : Math.max(0, x - 1))); }
    } catch {
      timer.current = setTimeout(() => poll(id), 4000);
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null); setRun(null); setBusy(true);
    try {
      const r = await fetch(`${API_URL}/runs`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hypothesis: text, name: name || null }),
      });
      const d = await r.json();
      if (!r.ok) { setError(d.detail ?? "Something went wrong."); setBusy(false); return; }
      setRun({ run_id: d.run_id, status: "queued", hypothesis: text, note: null, error: null });
      poll(d.run_id);
    } catch {
      setError("Couldn't reach the server. It sleeps when idle and takes about a minute to wake; try again shortly.");
      setBusy(false);
    }
  }

  const stepIndex = run ? STEPS.findIndex(([s]) => s === run.status) : -1;

  return (
    <>
      <div className="eyebrow">Live</div>
      <h1 className="display" style={{ fontSize: "clamp(1.9rem,4vw,2.6rem)" }}>Ask it a question.</h1>
      <p className="lede">
        Type a market hypothesis. The agent picks a feature, runs the backtests on S&amp;P 500 data, deflates the result
        for the number of things it tried, and writes a note. Expect 30 to 90 seconds. Your result is shown only to
        you.
      </p>

      {!API_URL ? (
        <div className="notice"><span className="dot" /><span>Live runs are offline right now. The frozen results on the <Link href="/">home page</Link> are always available.</span></div>
      ) : (
        <form onSubmit={submit} className="card" style={{ marginTop: 28 }}>
          <label className="f" htmlFor="h">Hypothesis</label>
          <textarea id="h" value={text} maxLength={300} onChange={(e) => setText(e.target.value)} placeholder="e.g. Do low-volatility stocks outperform?" disabled={busy} />
          <div className="examples">
            {EXAMPLES.map((x) => <button type="button" key={x} onClick={() => setText(x)} disabled={busy}>{x}</button>)}
          </div>
          <div style={{ marginTop: 18 }}>
            <label className="f" htmlFor="n">Your name <span className="muted" style={{ fontWeight: 400 }}>(optional, seen only by the author)</span></label>
            <input id="n" type="text" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} placeholder="e.g. Jane, Acme Capital" disabled={busy} />
          </div>
          <div style={{ display: "flex", gap: 14, alignItems: "center", marginTop: 20, flexWrap: "wrap" }}>
            <button className="btn primary" type="submit" disabled={busy || text.trim().length < 12 || left === 0}>
              {busy ? <><span className="spinner" /> Running</> : "Run the agent"}
            </button>
            <span className="small muted">
              {waking ? "Waking the server (about a minute after it has been idle)…" : left === null ? "" : left === 0 ? "No runs left today; resets at midnight UTC." : `${left} run${left === 1 ? "" : "s"} left for you today.`}
            </span>
          </div>
          {error && <div className="error">{error}</div>}
        </form>
      )}

      {run && (
        <section style={{ marginTop: 36 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <div className="steps">
              {STEPS.map(([s, l], i) => (
                <span key={s} className={`step ${i === stepIndex ? "on" : i < stepIndex ? "done" : ""}`}>{l}</span>
              ))}
            </div>
            <Chip kind={run.status} />
          </div>
          {run.status === "failed" && <div className="error">{run.error ?? "The run failed."}</div>}
          {run.note && (
            <div style={{ marginTop: 24 }}>
              <h2 className="display" style={{ fontSize: "1.6rem", marginBottom: 18 }}>{run.hypothesis}</h2>
              <NoteBody n={run.note} />
            </div>
          )}
        </section>
      )}
    </>
  );
}
