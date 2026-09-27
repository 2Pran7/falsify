import Link from "next/link";
import { notFound } from "next/navigation";
import { NoteBody } from "@/components/NoteBody";
import { day, snapshot } from "@/lib/data";

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
  return (
    <>
      <p className="small"><Link href="/notes/">← All notes</Link></p>
      <div className="eyebrow" style={{ marginTop: 20 }}>Research note</div>
      <h1 className="display" style={{ fontSize: "clamp(1.7rem,3.6vw,2.4rem)" }}>{n.hypothesis}</h1>
      <div className="meta">
        <span>{day(n.created_at)}</span><span>{n.run.model}</span><span>{n.run.turns} turns</span><span>${n.run.cost_usd.toFixed(3)}</span>
      </div>
      <div style={{ marginTop: 28 }}><NoteBody n={n} /></div>
    </>
  );
}
