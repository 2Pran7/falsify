"use client";

import { useEffect, useState } from "react";
import { Chip } from "@/components/Chip";
import { NoteBody } from "@/components/NoteBody";
import { API_URL } from "@/lib/format";
import type { NoteView } from "@/lib/types";

type Row = {
  run_id: string; created_at: string; finished_at: string | null; visitor: string; display_name: string | null;
  hypothesis: string; status: string; note: NoteView | null; error: string | null; cost_usd: number;
};
type Payload = { runs: Row[]; spent_today_usd: number; limits: { daily_usd_cap: number; daily_run_cap: number; per_visitor_cap: number } };

const KEY = "falsify-admin-token";

export function AdminClient() {
  const [token, setToken] = useState("");
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load(t: string) {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API_URL}/admin/runs`, { headers: { Authorization: `Bearer ${t}` } });
      if (r.status === 401) { setError("Wrong token."); setData(null); return; }
      if (!r.ok) throw new Error(String(r.status));
      setData(await r.json());
      try { sessionStorage.setItem(KEY, t); } catch {}
    } catch {
      setError("Couldn't reach the server. If it was idle, it takes about a minute to wake.");
    } finally { setLoading(false); }
  }

  useEffect(() => {
    let t = "";
    try { t = sessionStorage.getItem(KEY) ?? ""; } catch {}
    if (t && API_URL) { setToken(t); load(t); }
  }, []);

  if (!API_URL) return <p className="muted">The live API is not configured for this deployment.</p>;

  return (
    <>
      <div className="eyebrow">Private</div>
      <h1 className="display" style={{ fontSize: "clamp(1.8rem,4vw,2.4rem)" }}>Live runs</h1>
      {!data ? (
        <form className="card" style={{ marginTop: 24, maxWidth: 520 }} onSubmit={(e) => { e.preventDefault(); load(token); }}>
          <label className="f" htmlFor="t">Admin token</label>
          <input id="t" type="password" value={token} onChange={(e) => setToken(e.target.value)} autoComplete="off" />
          <p className="hint">From the ADMIN_TOKEN setting on Render. Kept for this browser tab only.</p>
          <button className="btn primary" style={{ marginTop: 14 }} disabled={!token || loading}>{loading ? "Loading…" : "Open"}</button>
          {error && <div className="error">{error}</div>}
        </form>
      ) : (
        <>
          <div className="stats" style={{ marginTop: 24 }}>
            <div className="stat"><div className="v">{data.runs.length}</div><div className="k">Runs recorded (latest 200)</div></div>
            <div className="stat"><div className="v">${data.spent_today_usd.toFixed(2)}<small> / ${data.limits.daily_usd_cap.toFixed(2)}</small></div><div className="k">Spent today, incl. reservations for unfinished runs</div></div>
            <div className="stat"><div className="v">{new Set(data.runs.map((r) => r.visitor)).size}</div><div className="k">Distinct visitors (hashed, no IPs stored)</div></div>
          </div>
          <div style={{ display: "flex", gap: 10, margin: "20px 0 12px" }}>
            <button className="btn" onClick={() => load(token)} disabled={loading}>{loading ? "Refreshing…" : "Refresh"}</button>
            <button className="btn" onClick={() => { try { sessionStorage.removeItem(KEY); } catch {} setData(null); setToken(""); }}>Lock</button>
          </div>
          <div className="table-card scroll">
            <table>
              <thead><tr><th>Hypothesis</th><th>Who</th><th>When (UTC)</th><th>Cost</th><th>Status</th></tr></thead>
              <tbody>
                {data.runs.map((r) => (
                  <tr key={r.run_id} className="link" onClick={() => setOpen(open === r.run_id ? null : r.run_id)}>
                    <td><span className="cell-title">{r.hypothesis}</span>{r.error && <div className="cell-sub" style={{ color: "var(--fail)" }}>{r.error}</div>}</td>
                    <td className="small">{r.display_name ?? <span className="muted">anonymous</span>}<div className="cell-sub mono">{r.visitor}</div></td>
                    <td className="num">{r.created_at.slice(0, 16).replace("T", " ")}</td>
                    <td className="num">${r.cost_usd.toFixed(3)}</td>
                    <td>
                      <Chip kind={r.status} />
                      {r.note && <div style={{ marginTop: 4 }}><Chip kind={r.note.publishable ? "publishable" : "refused"} /></div>}
                    </td>
                  </tr>
                ))}
                {data.runs.length === 0 && <tr><td colSpan={5} className="muted">No live runs yet.</td></tr>}
              </tbody>
            </table>
          </div>
          {open && (() => {
            const r = data.runs.find((x) => x.run_id === open);
            return r?.note ? (
              <section style={{ marginTop: 28 }}>
                <h2 className="display" style={{ fontSize: "1.5rem", marginBottom: 16 }}>{r.hypothesis}</h2>
                <NoteBody n={r.note} />
              </section>
            ) : null;
          })()}
        </>
      )}
    </>
  );
}
