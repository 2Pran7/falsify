import Link from "next/link";
import { Chip } from "@/components/Chip";
import { day, snapshot } from "@/lib/data";

export const metadata = { title: "Research notes · falsify" };

export default function Notes() {
  const { counts, items } = snapshot.notes;
  return (
    <>
      <div className="eyebrow">Research notes</div>
      <h1 className="display" style={{ fontSize: "clamp(1.9rem,4vw,2.6rem)" }}>Every run, including the refused ones.</h1>
      <p className="lede">
        {counts.publishable} of {counts.total} stored runs passed every publication check. The other{" "}
        {counts.unpublishable} are shown with the reason they were stopped, because a record that keeps only its
        successes is a highlight reel.
      </p>
      <div className="table-card scroll" style={{ marginTop: 32 }}>
        <table>
          <thead><tr><th>Hypothesis</th><th>Date</th><th>Model</th><th>Cost</th><th>Status</th></tr></thead>
          <tbody>
            {items.map((n) => (
              <tr key={n.note_id}>
                <td>
                  <Link href={`/notes/${n.note_id}/`} className="cell-title">{n.hypothesis}</Link>
                  {!n.publishable && <div className="cell-sub">{n.unpublishable_reasons[0]}</div>}
                </td>
                <td className="num">{day(n.created_at)}</td>
                <td className="small">{n.run.model}</td>
                <td className="num">${n.run.cost_usd.toFixed(3)}</td>
                <td><Chip kind={n.publishable ? "publishable" : "refused"} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
