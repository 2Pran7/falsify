import { NotesTable } from "@/components/NotesTable";
import { snapshot } from "@/lib/data";

export const metadata = { title: "Research notes · falsify" };

export default function Notes() {
  const { counts, items } = snapshot.notes;
  return (
    <>
      <div className="eyebrow">Research notes</div>
      <h1 className="display" style={{ fontSize: "clamp(1.9rem,4vw,2.6rem)" }}>Every question, once.</h1>
      <p className="lede" style={{ marginBottom: 28 }}>
        One row per effect tested, however the question was worded. Asking again adds a run to the row, not a new row;
        open the run count to see every run and when it was asked. Questions from the live page appear here as soon as
        their note passes every check. Of the {counts.total} research runs,{" "}
        {counts.unpublishable === 1 ? "1 failed a check and stays" : `${counts.unpublishable} failed a check and stay`} listed with the reason, because a record that keeps only its successes is a highlight reel.
      </p>
      <NotesTable items={items} />
    </>
  );
}
