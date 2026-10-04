import Link from "next/link";
import { notFound } from "next/navigation";
import { NoteBody } from "@/components/NoteBody";
import { Chip } from "@/components/Chip";
import { day, snapshot } from "@/lib/data";
import { noteHref, otherRuns } from "@/lib/notes";

export const dynamicParams = false;

export function generateStaticParams() {
  // Static export refuses a dynamic route with no params, so an empty snapshot
  // still emits one page, which 404s by design.
  const ids = snapshot.notes.items.map((n) => ({ id: n.note_id }));
  return ids.length ? ids : [{ id: "none" }];
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const n = snapshot.notes.items.find((x) => x.note_id === id);
  return { title: n ? `${n.hypothesis} · falsify` : "Note not found · falsify" };
}

export default async function NotePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const n = snapshot.notes.items.find((x) => x.note_id === id);
  if (!n) notFound();
  const others = otherRuns(snapshot.notes.items, n);
  return (
    <>
      <p className="small"><Link href="/notes/">← All notes</Link></p>
      <div className="eyebrow" style={{ marginTop: 20 }}>{n.source === "live" ? "Live run, published by the author" : "Research note"}</div>
      <h1 className="display" style={{ fontSize: "clamp(1.7rem,3.6vw,2.4rem)" }}>{n.hypothesis}</h1>
      <div className="meta">
        <span>{day(n.created_at)}</span><span>{n.run.model}</span><span>{n.run.turns} turns</span><span>${n.run.cost_usd.toFixed(3)}</span>
      </div>
      {n.source === "live" && (
        <p className="small muted" style={{ marginTop: 10 }}>
          Asked by a visitor on the live page. Live runs use the latest ~3 years of prices so they fit a free server.
        </p>
      )}
      <div style={{ marginTop: 28 }}><NoteBody n={n} /></div>
      {others.length > 0 && (
        <>
          <h3 className="sub">Other research runs testing the same effect</h3>
          <div className="table-card scroll">
            <table>
              <thead><tr><th>Run</th><th>Verdict</th><th>Cost</th><th>Status</th></tr></thead>
              <tbody>
                {others.map((o) => (
                  <tr key={o.note_id}>
                    <td><Link href={noteHref(o)}>{day(o.created_at)}</Link>{o.source === "live" && <span className="muted small"> · live</span>}</td>
                    <td>{o.verdict ? <Chip kind={o.verdict.outcome} /> : <span className="muted small">n/a</span>}</td>
                    <td className="num">${o.run.cost_usd.toFixed(3)}</td>
                    <td><Chip kind={o.publishable ? "publishable" : "refused"} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
