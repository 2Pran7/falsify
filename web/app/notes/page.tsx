import Link from "next/link";
import { day, snapshot } from "@/lib/data";

export const metadata = { title: "Research notes · falsify" };

export default function Notes() {
  const { counts, items } = snapshot.notes;
  return (
    <>
      <h1>Research notes</h1>
      <p className="lede">
        Every stored agent run, including the ones the system refused to publish. {counts.publishable} of{" "}
        {counts.total} passed; the other {counts.unpublishable} are shown with the reason they were
        stopped.
      </p>
      {items.length === 0 && <p className="muted">No notes were stored when this snapshot was taken.</p>}
      {items.map((n) => (
        <article className="card" key={n.note_id}>
          <header>
            <h3>
              <Link href={`/notes/${n.note_id}/`}>{n.hypothesis}</Link>
            </h3>
            <span className={`chip ${n.publishable ? "pass" : "fail"}`}>
              {n.publishable ? "Publishable" : "Refused"}
            </span>
          </header>
          <p className="small muted" style={{ marginTop: 6 }}>
            {day(n.created_at)} · {n.run.model} · {n.run.turns} turns · {n.run.n_backtests} backtest
            {n.run.n_backtests === 1 ? "" : "s"} · ${n.run.cost_usd.toFixed(3)}
          </p>
          {!n.publishable && (
            <ul className="reasons small">
              {n.unpublishable_reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
        </article>
      ))}
    </>
  );
}
