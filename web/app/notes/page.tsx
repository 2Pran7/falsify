import Link from "next/link";
import { Chip } from "@/components/Chip";
import { day, snapshot } from "@/lib/data";
import { groupNotes } from "@/lib/notes";

export const metadata = { title: "Research notes · falsify" };

export default function Notes() {
  const { counts, items } = snapshot.notes;
  const questions = groupNotes(items);
  return (
    <>
      <div className="eyebrow">Research notes</div>
      <h1 className="display" style={{ fontSize: "clamp(1.9rem,4vw,2.6rem)" }}>Every question, every run.</h1>
      <p className="lede">
        {questions.length} questions, asked {counts.total} times. {counts.publishable} runs passed every publication
        check; the {counts.unpublishable === 1 ? "other one is" : `other ${counts.unpublishable} are`} kept with the reason they were stopped, because a record that
        keeps only its successes is a highlight reel. Each question is listed once, linked to its latest run that
        passed; earlier runs are on its page.
      </p>
      <div className="table-card scroll" style={{ marginTop: 32 }}>
        <table>
          <thead><tr><th>Question</th><th>Verdict</th><th>Runs</th><th>Latest</th><th>Status</th></tr></thead>
          <tbody>
            {questions.map(({ key, lead, runs }) => (
              <tr key={key}>
                <td>
                  <Link href={`/notes/${lead.note_id}/`} className="cell-title">{lead.hypothesis}</Link>
                  {lead.source === "live" && <div className="cell-sub">Asked on the live page · latest ~3 years of prices</div>}
                  {!lead.publishable && <div className="cell-sub">{lead.unpublishable_reasons[0]}</div>}
                </td>
                <td>{lead.verdict ? <Chip kind={lead.verdict.outcome} /> : <span className="muted small">n/a</span>}</td>
                <td className="num">{runs.length}</td>
                <td className="num">{day(lead.created_at)}</td>
                <td><Chip kind={lead.publishable ? "publishable" : "refused"} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
