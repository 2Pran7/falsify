import type { Diagnostics, VerdictCard } from "@/lib/types";
import { num, pct } from "@/lib/format";

/* Charts drawn as inline SVG from numbers the pipeline stored. No chart
 * library: three small pictures do not justify a dependency, and inline SVG
 * renders in the static export, on the live page and in dark mode alike. */

const W = 640;

function fmtPct1(v: number | null | undefined) {
  return pct(v, 1);
}

/** Bucket returns, low signal to high. A real ranking effect is a staircase. */
export function BucketBars({ d, longLow, longShort }: { d: Diagnostics; longLow: boolean; longShort: boolean }) {
  const b = (d.buckets ?? []).filter((x) => x.ann_return !== null);
  if (b.length < 2) return null;
  const H = 230, top = 22, bottom = 44, left = 8, right = 8;
  const vals = b.map((x) => x.ann_return as number);
  const hi = Math.max(0, ...vals), lo = Math.min(0, ...vals);
  const span = hi - lo || 1;
  const y = (v: number) => top + ((hi - v) / span) * (H - top - bottom);
  const slot = (W - left - right) / b.length;
  const bw = Math.min(46, slot * 0.68);
  const zero = y(0);
  const first = 0, last = b.length - 1;
  const longIdx = longLow ? first : last, shortIdx = longLow ? last : first;
  return (
    <figure className="chart">
      <figcaption>
        Return by signal bucket <span className="muted">annualised, gross of costs, 1 = lowest signal
        {typeof d.monotonicity === "number" ? <> · staircase (Spearman) <b>{num(d.monotonicity, 2)}</b></> : null}</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Bar chart of annualised return by signal bucket">
        <line x1={left} x2={W - right} y1={zero} y2={zero} className="axis" />
        {b.map((x, i) => {
          const v = x.ann_return as number;
          const cx = left + slot * i + slot / 2;
          const y0 = Math.min(y(v), zero), h = Math.max(1, Math.abs(y(v) - zero));
          const tag = i === longIdx ? "long" : longShort && i === shortIdx ? "short" : "";
          return (
            <g key={x.bucket}>
              <rect x={cx - bw / 2} y={y0} width={bw} height={h} rx={3} className={v >= 0 ? "bar pos" : "bar neg"} />
              <text x={cx} y={v >= 0 ? y0 - 6 : y0 + h + 13} className="val" textAnchor="middle">{fmtPct1(v)}</text>
              <text x={cx} y={H - bottom + 18} className="tick" textAnchor="middle">{x.bucket}</text>
              {tag && <text x={cx} y={H - bottom + 34} className={`leg ${tag}`} textAnchor="middle">{tag}</text>}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

/** Growth of 1 over the invested window. */
export function EquityCurve({ d }: { d: Diagnostics }) {
  const e = d.equity ?? [];
  if (e.length < 3) return null;
  const H = 220, top = 16, bottom = 28, left = 44, right = 12;
  const v = e.map((p) => p.equity);
  const hi = Math.max(1, ...v), lo = Math.min(1, ...v);
  const span = hi - lo || 1;
  const x = (i: number) => left + (i / (e.length - 1)) * (W - left - right);
  const y = (q: number) => top + ((hi - q) / span) * (H - top - bottom);
  const path = e.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join(" ");
  const area = `${path} L${x(e.length - 1).toFixed(1)},${y(1).toFixed(1)} L${x(0).toFixed(1)},${y(1).toFixed(1)} Z`;
  const end = v[v.length - 1];
  return (
    <figure className="chart">
      <figcaption>
        Growth of 1, net of costs <span className="muted">{e[0].ts.slice(0, 10)} to {e[e.length - 1].ts.slice(0, 10)} · ends at <b>{num(end, 2)}</b></span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Equity curve">
        <line x1={left} x2={W - right} y1={y(1)} y2={y(1)} className="axis dash" />
        {y(1) - y(hi) > 14 && <text x={left - 6} y={y(hi) + 4} className="tick" textAnchor="end">{num(hi, 2)}</text>}
        <text x={left - 6} y={y(1) + 4} className="tick" textAnchor="end">1.00</text>
        {lo < 1 && y(lo) - y(1) > 14 && <text x={left - 6} y={y(lo) + 4} className="tick" textAnchor="end">{num(lo, 2)}</text>}
        <path d={area} className={end >= 1 ? "area pos" : "area neg"} />
        <path d={path} className={end >= 1 ? "line pos" : "line neg"} />
        <text x={left} y={H - 8} className="tick">{e[0].ts.slice(0, 7)}</text>
        <text x={W - right} y={H - 8} className="tick" textAnchor="end">{e[e.length - 1].ts.slice(0, 7)}</text>
      </svg>
    </figure>
  );
}

/** Two halves and three cost levels, side by side. */
export function Robustness({ d }: { d: Diagnostics }) {
  const h = d.halves ?? [], c = d.costs ?? [];
  if (!h.length && !c.length) return null;
  return (
    <div className="grid2 robust">
      {h.length > 0 && (
        <table className="mini">
          <caption>Two halves of the sample</caption>
          <thead><tr><th>Period</th><th>Days</th><th>Sharpe</th></tr></thead>
          <tbody>
            {h.map((x, i) => (
              <tr key={i}>
                <td>{x.first_date.slice(0, 7)} to {x.last_date.slice(0, 7)}</td>
                <td>{x.n_days}</td>
                <td className={(x.sharpe ?? 0) > 0 ? "pos" : "neg"}>{num(x.sharpe, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {c.length > 0 && (
        <table className="mini">
          <caption>Cost sensitivity, one-way</caption>
          <thead><tr><th>Cost</th><th>Sharpe</th><th>CAGR</th></tr></thead>
          <tbody>
            {c.map((x) => (
              <tr key={x.cost_bps}>
                <td>{x.cost_bps} bps</td>
                <td className={(x.sharpe ?? 0) > 0 ? "pos" : "neg"}>{num(x.sharpe, 2)}</td>
                <td>{pct(x.cagr, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const OUTCOME_CLASS: Record<string, string> = {
  supported: "pass", not_confirmed: "partial", contradicted: "fail", no_verdict: "na",
};
const PRED: Record<string, string> = {
  top_beats_bottom: "high signal beats low",
  bottom_beats_top: "low signal beats high",
};

function yesNo(v: boolean | null | undefined) {
  return v === null || v === undefined ? null : v ? "Yes" : "No";
}

/** The card at the top of a note. Every value is the pipeline's. */
export function Verdict({ v }: { v: VerdictCard }) {
  const facts: [string, string | null][] = [
    ["Sharpe (annualised)", typeof v.sharpe === "number" ? num(v.sharpe, 2) : null],
    ["P(beats best of N trials)", typeof v.prob_beats_best_of_n_trials === "number" ? num(v.prob_beats_best_of_n_trials, 3) : null],
    ["Clears 0.95 gate", yesNo(v.clears_confirmation_gate)],
    ["Trials run", v.n_trials != null ? String(v.n_trials) : null],
    ["Invested days", v.n_invested_days != null ? String(v.n_invested_days) : null],
    ["Staircase (Spearman)", typeof v.monotonicity === "number" ? num(v.monotonicity, 2) : null],
    ["Both halves as predicted", yesNo(v.both_halves_as_predicted)],
    ["Survives 25 bps", yesNo(v.survives_25bps)],
  ];
  const shown = facts.filter(([, x]) => x !== null);
  const cls = OUTCOME_CLASS[v.outcome] ?? "na";
  return (
    <div className={`verdict ${cls}`}>
      <div className="verdict-head">
        <span className="verdict-label">{v.label}</span>
        <span className="verdict-reason">{v.reason}</span>
      </div>
      {v.variant && (
        <p className="verdict-sub">
          <code>{v.variant}</code>
          {v.prediction && <> · predicted: {PRED[v.prediction] ?? v.prediction}{v.prediction_declared === false ? " (not declared)" : ""}</>}
          {v.first_date && v.last_date && <> · {v.first_date.slice(0, 10)} to {v.last_date.slice(0, 10)}</>}
          {v.universe && <> · {v.universe === "point_in_time" ? "point-in-time universe" : "current constituents"}</>}
        </p>
      )}
      {shown.length > 0 && (
        <dl className="verdict-facts">
          {shown.map(([k, x]) => (
            <div key={k}><dt>{k}</dt><dd>{x}</dd></div>
          ))}
        </dl>
      )}
      <p className="verdict-foot">Computed by the pipeline from the stored numbers. The model did not write this card.</p>
    </div>
  );
}
