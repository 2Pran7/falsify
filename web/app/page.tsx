import Link from "next/link";
import { Chip } from "@/components/Chip";
import { NotesTable } from "@/components/NotesTable";
import {
  type AnomalyRow,
  type Result,
  type Universe,
  day,
  num,
  pct,
  signed,
  snapshot,
  title,
  UNIVERSE_LABEL,
} from "@/lib/data";

const SHORT: Record<Universe, string> = { current: "Today's list", point_in_time: "Point-in-time" };

function Leg({ u, r }: { u: Universe; r: Result | null }) {
  const s = snapshot.scoring;
  return (
    <div>
      <div className="leg-label">
        <span>{UNIVERSE_LABEL[u]}</span>
        <Chip kind={r?.verdict ?? "none"} />
      </div>
      {r === null ? (
        <p className="small muted">No stored verdict for this universe.</p>
      ) : (
        <>
          <ul className="reasons">{r.reasons.map((x, i) => <li key={i}>{x}</li>)}</ul>
          <dl className="kv">
            <dt>Sharpe, oriented to the prediction</dt><dd>{signed(r.oriented_sharpe)}</dd>
            <dt>P(beats best of {r.n_trials} trials) · gate {s.deflation_threshold}</dt><dd>{num(r.deflated_psr, 3)}</dd>
            <dt>BH-adjusted p · gate {s.fdr_alpha}</dt><dd>{num(r.p_value_adjusted, 3)}</dd>
            <dt>Invested days / history</dt><dd>{r.n_invested_days} / {r.history_days}</dd>
          </dl>
        </>
      )}
    </div>
  );
}

function Anomaly({ a }: { a: AnomalyRow }) {
  return (
    <details className="anomaly" id={a.key}>
      <summary>
        <div>
          <div className="t">{title(a.key)}</div>
          <div className="c">{a.citation}</div>
        </div>
        <div className="chips" style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {snapshot.evals.universes.map((u) => <Chip key={u} kind={a.results[u]?.verdict ?? "none"} />)}
        </div>
        <span className="chev" aria-hidden>›</span>
      </summary>
      <div className="body">
        <p className="small" style={{ marginTop: 16 }}>{a.hypothesis}</p>
        <p className="small muted">
          Pre-registered: the <b>{a.direction === 1 ? "top" : "bottom"}</b> bucket of <code>{a.feature}</code> wins.
          Published reference Sharpe {num(a.published_sharpe)}, reported but never a target. Needs {a.min_history_days}{" "}
          trading days of history per stock.
          {a.survivorship_sharpe_gap !== null && <> Survivorship gap on this anomaly: <span className="mono">{signed(a.survivorship_sharpe_gap)}</span> Sharpe.</>}
        </p>
        <div className="grid2">
          {snapshot.evals.universes.map((u) => <Leg key={u} u={u} r={a.results[u]} />)}
        </div>
        <p className="caveat"><b>Known gap from the paper.</b> {a.caveat}</p>
      </div>
    </details>
  );
}

export default function Home() {
  const ev = snapshot.evals;
  const sv = snapshot.survivorship;
  const nc = snapshot.notes.counts;
  const tested = (u: Universe) =>
    ev.anomalies.filter((a) => a.results[u] && a.results[u]!.verdict !== "insufficient_data").length;
  const passed = (u: Universe) => ev.tally[u].pass;
  const s = snapshot.scoring;

  return (
    <>
      <section className="hero">
        <div className="eyebrow">Quantitative research agent</div>
        <h1 className="display">An agent built to disprove<br />its own hypotheses.</h1>
        <p className="lede">
          falsify takes a market hypothesis in plain English, tests it with walk-forward backtests, deflated Sharpe
          ratios and multiple-testing correction, and writes a research note that reports the failures. A language
          model plans the experiments. It never computes a number.
        </p>
        <div className="actions">
          <Link href="/try/" className="btn primary">Ask it a question →</Link>
          <Link href="/#evals" className="btn">See the eval suite</Link>
        </div>

        <div className="stats">
          {sv && (
            <div className="stat">
              <div className="v">{signed(sv.metrics.sharpe_gap)}</div>
              <div className="k">Sharpe inflation from <b>survivorship bias</b>: {pct(sv.metrics.total_return_current, 1)} vs {pct(sv.metrics.total_return_pit, 1)} return</div>
            </div>
          )}
          <div className="stat">
            <div className="v">{passed("point_in_time")}<small> / {tested("point_in_time")}</small></div>
            <div className="k">Published anomalies that <b>survived</b> deflation and FDR on this sample</div>
          </div>
          <div className="stat">
            <div className="v">{nc.publishable}<small> / {nc.total}</small></div>
            <div className="k">Agent notes that passed <b>numeric provenance</b>; the rest are shown as refused</div>
          </div>
        </div>
        <div className="notice">
          <span className="dot" />
          <span>
            Frozen evidence, last scored {day(ev.last_run_at)}. Predictions were fixed and hashed before any result
            existed, so &ldquo;we did not move the goalposts&rdquo; is a string comparison, not a promise.
          </span>
        </div>
      </section>

      {sv && (
        <section className="section" id="survivorship">
          <div className="section-head">
            <div>
              <h2>Survivorship bias, measured</h2>
              <p>
                One 12-1 momentum strategy, run on today&apos;s S&amp;P 500 list and on membership as it stood each day,
                reconstructed from the git history of a public constituents file. {sv.window.invested_days} invested days,{" "}
                {sv.window.first} to {sv.window.last}.
              </p>
            </div>
          </div>
          <div className="table-card scroll">
            <table>
              <thead><tr><th>Metric</th><th>Today&apos;s list</th><th>Point-in-time</th><th>Gap</th></tr></thead>
              <tbody>
                {([["Total return", "total_return", true], ["CAGR", "cagr", true], ["Annualised volatility", "ann_vol", true],
                   ["Sharpe", "sharpe", false], ["Max drawdown", "max_drawdown", true]] as [string, string, boolean][]).map(([l, k, p]) => (
                  <tr key={k}>
                    <td>{l}</td>
                    {(["current", "pit", "gap"] as const).map((x) => {
                      const v = sv.metrics[`${k}_${x}`];
                      return <td key={x} className="num">{v === undefined ? "n/a" : p ? pct(v) : num(v)}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 12 }}>
            Volatility barely moves: the companies a today&apos;s-list backtest cannot see were not adding risk, they were
            removing return that was never earned. A <b>{sv.bound} bound</b>: a cash acquisition stops having prices, so
            its final move is missing from both runs. Splice gate: point-in-time, {sv.n_tickers_gated} ticker(s) truncated.
          </p>
        </section>
      )}

      <section className="section" id="evals">
        <div className="section-head">
          <div>
            <h2>The eval suite</h2>
            <p>Six published anomalies, each judged on both universes by four rules in code. Open any row for the reasons.</p>
          </div>
        </div>
        <div className="table-card scroll" style={{ marginBottom: 18 }}>
          <table>
            <thead>
              <tr><th>Anomaly</th><th>Predicts</th>{ev.universes.map((u) => <th key={u}>{SHORT[u]}</th>)}<th>Oriented Sharpe</th></tr>
            </thead>
            <tbody>
              {ev.anomalies.map((a) => (
                <tr key={a.key}>
                  <td><a href={`#${a.key}`} className="cell-title" style={{ color: "var(--ink)", textDecoration: "none" }}>{title(a.key)}</a><div className="cell-sub">{a.citation}</div></td>
                  <td className="small">{a.direction === 1 ? "Top wins" : "Bottom wins"}</td>
                  {ev.universes.map((u) => <td key={u}><Chip kind={a.results[u]?.verdict ?? "none"} /></td>)}
                  <td className="num">{ev.universes.map((u) => signed(a.results[u]?.oriented_sharpe)).join(" / ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {ev.missing.length > 0 && <p className="small"><b>Not yet scored:</b> {ev.missing.join(", ")}.</p>}
        {ev.anomalies.map((a) => <Anomaly key={a.key} a={a} />)}
      </section>

      <section className="section" id="notes">
        <div className="section-head">
          <div>
            <h2>Research notes</h2>
            <p>
              A note is published only if every numeral traces to a number the pipeline produced, the run completed, and
              deflation statistics were computed. Refused notes are kept and shown with their reasons.
            </p>
          </div>
          <Link href="/notes/" className="btn">All questions</Link>
        </div>
        <NotesTable items={snapshot.notes.items} limit={5} />
      </section>

      <section className="section" id="method">
        <div className="section-head">
          <div>
            <h2>Method</h2>
            <p>How a verdict is decided, in the order the rules are applied. Insufficient data is not a failure: an anomaly that needs three years of history on a two-year panel has not been refuted, it has not been tested.</p>
          </div>
        </div>
        <ol className="rules">
          <li><b>Sign</b><span>Did the long/short spread run the way the paper predicted? Four of six predict the bottom bucket wins, so the direction is fixed in advance and every later test uses the oriented Sharpe.</span></li>
          <li><b>Deflation</b><span>The probability that the true Sharpe beats the best of N worthless strategies must reach {s.deflation_threshold}, with N the number the suite actually tried. A missing statistic fails the gate.</span></li>
          <li><b>Multiplicity</b><span>Benjamini-Hochberg across the tested anomalies at α = {s.fdr_alpha}. Untestable anomalies leave the denominator rather than padding it.</span></li>
          <li><b>Embarrassment</b><span>A spread more than {s.embarrassment_multiple}× the published reference is downgraded to partial. On a short sample a huge number is likelier a defect than a discovery.</span></li>
        </ol>
        <p className="small muted" style={{ marginTop: 16 }}>
          Assumed variance of Sharpe across trials: {num(s.trial_variance_assumed, 4)}, labelled and stored with every
          result. Prices are split-adjusted only, so every return is a price return.
        </p>
      </section>
    </>
  );
}
