import Link from "next/link";
import { notFound } from "next/navigation";
import { day, num, pct, snapshot, type BacktestView } from "@/lib/data";

export const dynamicParams = false;

export function generateStaticParams() {
  // Next's static export refuses a dynamic route with no params, so an empty
  // snapshot still emits one page, which 404s by design.
  const ids = snapshot.notes.items.map((n) => ({ id: n.note_id }));
  return ids.length ? ids : [{ id: "none" }];
}

const PCT = new Set(["total_return", "cagr", "ann_vol", "max_drawdown"]);

function Metric({ k, v }: { k: string; v: unknown }) {
  if (typeof v !== "number") return <dd>{String(v)}</dd>;
  return <dd>{PCT.has(k) ? pct(v) : Number.isInteger(v) ? v : num(v, 4)}</dd>;
}

function Backtest({ b }: { b: BacktestView }) {
  const stats = Object.entries(b.statistics);
  return (
    <div className="leg">
      <div className="label">
        <span>{b.variant}</span>
      </div>
      <div className="twocol" style={{ marginTop: 0 }}>
        <dl className="stats">
          {Object.entries(b.metrics)
            .filter(([k]) => !["feature", "n_buckets", "long_short"].includes(k))
            .map(([k, v]) => (
              <FragmentPair key={k} k={k} v={v} />
            ))}
        </dl>
        <dl className="stats">
          {stats.length === 0 ? (
            <>
              <dt>statistics</dt>
              <dd>{b.analysis_error ?? "not computed"}</dd>
            </>
          ) : (
            stats.map(([k, v]) => <FragmentPair key={k} k={k} v={v} />)
          )}
        </dl>
      </div>
    </div>
  );
}

function FragmentPair({ k, v }: { k: string; v: unknown }) {
  return (
    <>
      <dt>{k.replaceAll("_", " ")}</dt>
      <Metric k={k} v={v} />
    </>
  );
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
  const p = n.provenance;
  return (
    <>
      <p className="small">
        <Link href="/notes/">&larr; All notes</Link>
      </p>
      <h1>{n.hypothesis}</h1>
      <p className="small muted">
        {day(n.created_at)} · {n.run.model} · stopped on <code>{n.run.stop_reason}</code> ·{" "}
        {n.run.turns} turns · {n.run.total_tokens.toLocaleString("en-GB")} tokens · $
        {n.run.cost_usd.toFixed(3)}
      </p>

      <div className="card">
        <header>
          <h3>{n.publishable ? "Publishable" : "Refused publication"}</h3>
          <span className={`chip ${n.publishable ? "pass" : "fail"}`}>
            {n.publishable ? "Publishable" : "Refused"}
          </span>
        </header>
        {n.publishable ? (
          <p className="small" style={{ marginTop: 8 }}>
            Every numeral checked traces to a number the pipeline produced ({p.verified} of {p.checked},{" "}
            {p.exempt} exempt), the run completed, and the deflation statistics were computed.
            Provenance checks numbers, not claims: the conclusion is still the model&apos;s.
          </p>
        ) : (
          <ul className="reasons small">
            {n.unpublishable_reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        )}
        {p.unverified.length > 0 && (
          <details>
            <summary>
              {p.unverified.length} numeral{p.unverified.length === 1 ? "" : "s"} with no source
            </summary>
            <ul className="small mono">
              {p.unverified.map((u, i) => (
                <li key={i}>{u}</li>
              ))}
            </ul>
          </details>
        )}
      </div>

      <h2>The note, as the model wrote it</h2>
      <div className="prose">{n.prose}</div>

      <h2>Backtests</h2>
      <p className="small muted">
        Statistics are recomputed at the run&apos;s final trial count, so a stored record can never be
        more flattering than the run deserves.
      </p>
      {n.backtests.map((b, i) => (
        <Backtest key={i} b={b} />
      ))}

      <h2>Run</h2>
      <p className="small">
        Tool sequence: <code>{n.run.tool_sequence.join(" → ")}</code>
      </p>
      <dl className="stats" style={{ maxWidth: 420 }}>
        {Object.entries(n.assumptions).map(([k, v]) => (
          <FragmentPair key={k} k={k} v={v} />
        ))}
      </dl>
    </>
  );
}
