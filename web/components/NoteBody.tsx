import type { BacktestView, NoteView } from "@/lib/types";
import { num, pct } from "@/lib/format";
import { Chip } from "@/components/Chip";
import { Markdown } from "@/components/Markdown";
import { BucketBars, EquityCurve, Robustness, Verdict } from "@/components/Charts";

const PCT = new Set(["total_return", "cagr", "ann_vol", "max_drawdown"]);
// What a reader needs, in reading order. The stored record holds more (help
// text for the model, intermediate counts); it stays in the JSON, not the page.
const METRICS = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "n_invested_days"];
const STATS = ["prob_sharpe_above_zero", "prob_beats_best_of_n_trials", "min_track_record_length_days", "n_trials_used", "trial_variance_assumption"];
const LABEL: Record<string, string> = {
  total_return: "Total return", cagr: "CAGR", ann_vol: "Annualised volatility", sharpe: "Sharpe",
  max_drawdown: "Max drawdown", n_invested_days: "Invested days",
  prob_sharpe_above_zero: "P(Sharpe > 0)", prob_beats_best_of_n_trials: "P(beats best of N trials)",
  min_track_record_length_days: "Min. track record (days)", n_observations: "Observations",
  n_trials_used: "Trials used", trial_variance_assumption: "Trial variance (assumed)",
};

const PROB = new Set(["prob_sharpe_above_zero", "prob_beats_best_of_n_trials"]);

// Standard reading precision: Sharpe 2 dp, percentages 1 dp, probabilities 3 dp.
function value(k: string, v: unknown): string {
  if (typeof v !== "number") return v === null || v === undefined ? "n/a" : String(v);
  if (PCT.has(k)) return pct(v, 1);
  if (k === "sharpe") return num(v, 2);
  if (PROB.has(k)) return num(v, 3);
  if (Number.isInteger(v)) return v.toLocaleString("en-GB");
  // Small values keep their significant digits: 0.0009 must not print as 0.001,
  // because that number IS the labelled assumption.
  if (Math.abs(v) < 0.01) return v.toPrecision(2);
  return num(v, Math.abs(v) >= 100 ? 0 : 2);
}

function Pairs({ obj, keys }: { obj: Record<string, unknown>; keys: string[] }) {
  return (
    <dl className="kv">
      {keys
        .filter((k) => k in obj)
        .map((k) => [k, obj[k]] as const)
        .map(([k, v]) => (
          <div key={k} style={{ display: "contents" }}>
            <dt>{LABEL[k] ?? k.replaceAll("_", " ")}</dt>
            <dd>{value(k, v)}</dd>
          </div>
        ))}
    </dl>
  );
}

function Backtest({ b }: { b: BacktestView }) {
  const hasStats = Object.keys(b.statistics).length > 0;
  const d = b.diagnostics ?? {};
  const longLow = b.metrics.prediction === "bottom_beats_top";
  const longShort = b.metrics.long_short !== false;
  return (
    <div className="card bt">
      <h4>{b.variant}</h4>
      {typeof b.metrics.legs === "string" && <p className="small muted" style={{ margin: "-4px 0 10px" }}>{b.metrics.legs}</p>}
      <div className="grid2">
        <Pairs obj={b.metrics} keys={METRICS} />
        {hasStats ? <Pairs obj={b.statistics} keys={STATS} /> : <p className="small muted">{b.analysis_error ?? "No statistics computed."}</p>}
      </div>
      {(d.buckets?.length || d.equity?.length) ? (
        <div className="charts">
          <BucketBars d={d} longLow={longLow} longShort={longShort} />
          <EquityCurve d={d} />
          <Robustness d={d} />
        </div>
      ) : (
        <p className="small muted" style={{ marginTop: 12 }}>Bucket, sub-period and cost diagnostics were added after this run was recorded.</p>
      )}
    </div>
  );
}

/** One research note. Used by the frozen notes pages, the live Try page and admin. */
export function NoteBody({ n }: { n: NoteView }) {
  const p = n.provenance;
  return (
    <>
      {n.verdict && <Verdict v={n.verdict} />}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <strong>{n.publishable ? "Passed every publication check" : "Refused publication"}</strong>
          <Chip kind={n.publishable ? "publishable" : "refused"} />
        </div>
        {n.publishable ? (
          <p className="small muted" style={{ margin: "8px 0 0" }}>
            {p.verified} of {p.checked} numerals trace to a number the pipeline produced; the run completed; deflation
            statistics were computed. Provenance checks numbers, not claims: the conclusion is still the model&apos;s.
          </p>
        ) : (
          <ul className="reasons" style={{ marginTop: 8 }}>
            {n.unpublishable_reasons.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        )}
        {p.unverified.length > 0 && (
          <details style={{ marginTop: 8 }}>
            <summary className="small" style={{ cursor: "pointer", color: "var(--accent)" }}>
              {p.unverified.length} numeral{p.unverified.length === 1 ? "" : "s"} with no source
            </summary>
            <ul className="small mono">{p.unverified.map((u, i) => <li key={i}>{u}</li>)}</ul>
          </details>
        )}
      </div>

      <h3 className="sub">The note, as the model wrote it</h3>
      <div className="prose">{n.prose ? <Markdown text={n.prose} /> : "(no text returned)"}</div>

      {n.backtests.length > 0 && (
        <>
          <h3 className="sub">What the pipeline computed</h3>
          <p className="small muted">Statistics are recomputed at the run&apos;s final trial count, so a stored record is never more flattering than the run deserves.</p>
          {n.backtests.map((b, i) => <Backtest key={i} b={b} />)}
        </>
      )}

      <h3 className="sub">Run</h3>
      <div className="card">
        <p className="small" style={{ marginBottom: 10 }}>
          Tool sequence: <code>{n.run.tool_sequence.join(" → ") || "none"}</code>
        </p>
        <dl className="kv">
          <dt>Model</dt><dd>{n.run.model}</dd>
          <dt>Stopped on</dt><dd>{n.run.stop_reason}</dd>
          <dt>Turns</dt><dd>{n.run.turns}</dd>
          <dt>Tokens</dt><dd>{n.run.total_tokens.toLocaleString("en-GB")}</dd>
          <dt>Cost</dt><dd>${n.run.cost_usd.toFixed(3)}</dd>
        </dl>
      </div>
    </>
  );
}
