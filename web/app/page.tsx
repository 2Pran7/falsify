import Link from "next/link";
import {
  type AnomalyRow,
  type Result,
  type Universe,
  type Verdict,
  day,
  num,
  pct,
  signed,
  snapshot,
  title,
  UNIVERSE_LABEL,
  VERDICT_LABEL,
} from "@/lib/data";

function Chip({ v }: { v: Verdict | null }) {
  return <span className={`chip ${v ?? "none"}`}>{v ? VERDICT_LABEL[v] : "Not run"}</span>;
}

function Leg({ u, r, gate, alpha }: { u: Universe; r: Result | null; gate: number; alpha: number }) {
  return (
    <div className="leg">
      <div className="label">
        <span>{UNIVERSE_LABEL[u]}</span>
        <Chip v={r?.verdict ?? null} />
      </div>
      {r === null ? (
        <p className="small muted">No stored verdict for this universe.</p>
      ) : (
        <>
          <ul className="reasons small">
            {r.reasons.map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
          <dl className="stats">
            <dt>Sharpe, oriented to the prediction</dt>
            <dd>{signed(r.oriented_sharpe)}</dd>
            <dt>P(beats best of {r.n_trials} trials), gate {gate}</dt>
            <dd>{num(r.deflated_psr, 3)}</dd>
            <dt>BH-adjusted p, gate {alpha}</dt>
            <dd>{num(r.p_value_adjusted, 3)}</dd>
            <dt>Invested days / history</dt>
            <dd>
              {r.n_invested_days} / {r.history_days}
            </dd>
          </dl>
        </>
      )}
    </div>
  );
}

function AnomalyCard({ a }: { a: AnomalyRow }) {
  const s = snapshot.scoring;
  return (
    <article className="card" id={a.key}>
      <header>
        <h3>{title(a.key)}</h3>
        <span className="cite">{a.citation}</span>
      </header>
      <p className="small" style={{ marginTop: 8 }}>
        {a.hypothesis}
      </p>
      <p className="small muted">
        Pre-registered: the <strong>{a.direction === 1 ? "top" : "bottom"}</strong> bucket of{" "}
        <code>{a.feature}</code> wins. Published reference Sharpe {num(a.published_sharpe)}, reported
        and never used as a target. Needs {a.min_history_days} trading days of history per stock.
        {a.survivorship_sharpe_gap !== null && (
          <>
            {" "}
            Survivorship gap on this anomaly: <span className="mono">{signed(a.survivorship_sharpe_gap)}</span>{" "}
            Sharpe.
          </>
        )}
      </p>
      <div className="twocol">
        {snapshot.evals.universes.map((u) => (
          <Leg key={u} u={u} r={a.results[u]} gate={s.deflation_threshold} alpha={s.fdr_alpha} />
        ))}
      </div>
      <p className="caveat">
        <strong>Known gap from the paper:</strong> {a.caveat}
      </p>
    </article>
  );
}

function SurvivorshipSection() {
  const sv = snapshot.survivorship;
  if (!sv) return null;
  const m = sv.metrics;
  const rows: [string, string, boolean][] = [
    ["Total return", "total_return", true],
    ["CAGR", "cagr", true],
    ["Annualised volatility", "ann_vol", true],
    ["Sharpe", "sharpe", false],
    ["Max drawdown", "max_drawdown", true],
  ];
  const f = (x: number | undefined, isPct: boolean) =>
    x === undefined ? "n/a" : isPct ? pct(x) : num(x);
  return (
    <section>
      <h2 id="survivorship">Survivorship bias, measured</h2>
      <p>
        The same 12-1 momentum strategy, run twice over a common {sv.window.invested_days}-day invested
        window ({sv.window.first} to {sv.window.last}). Once on today&apos;s S&amp;P 500 list, the way
        most backtests are run, and once on membership as it stood on each date, reconstructed from
        the git history of a public constituents file.
      </p>
      <div className="headline">
        <div className="stat">
          <div className="v">
            {pct(m.total_return_current, 1)} vs {pct(m.total_return_pit, 1)}
          </div>
          <div className="k">Total return: today&apos;s list vs point-in-time</div>
        </div>
        <div className="stat">
          <div className="v">
            {num(m.sharpe_current)} vs {num(m.sharpe_pit)}
          </div>
          <div className="k">Sharpe ratio</div>
        </div>
        <div className="stat">
          <div className="v">{pct(m.ann_vol_gap, 1)}</div>
          <div className="k">Volatility difference: the missing companies were not adding risk</div>
        </div>
      </div>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Today&apos;s list</th>
              <th>Point-in-time</th>
              <th>Gap</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, k, isPct]) => (
              <tr key={k}>
                <td>{label}</td>
                <td className="num">{f(m[`${k}_current`], isPct)}</td>
                <td className="num">{f(m[`${k}_pit`], isPct)}</td>
                <td className="num">{f(m[`${k}_gap`], isPct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 12 }}>
        A <strong>{sv.bound} bound</strong>. A company acquired for cash stops having prices, so its
        final move is missing from the point-in-time run too. Splice gate: {sv.gate.replaceAll("_", "-")},{" "}
        {sv.n_tickers_gated} ticker(s) affected. Diagnostic on this window, not a publishable
        estimate.
      </p>
    </section>
  );
}

export default function Home() {
  const ev = snapshot.evals;
  const s = snapshot.scoring;
  const nTested = (u: Universe) =>
    ev.anomalies.filter((a) => a.results[u] && a.results[u]!.verdict !== "insufficient_data").length;
  const nc = snapshot.notes.counts;
  return (
    <>
      <h1>An agent built to disprove its own hypotheses.</h1>
      <p className="lede">
        falsify takes a market hypothesis in plain English, runs it through walk-forward backtests,
        deflated Sharpe and multiple-testing correction, and writes a research note that reports the
        failures. A language model plans the experiments. It never computes a number.
      </p>
      <p className="frozen">
        Everything below is frozen evidence from the pipeline&apos;s database, last scored{" "}
        {day(ev.last_run_at)}. Predictions were fixed and hashed before any result existed
        (sha256 <span className="mono">{snapshot.registry_sha.slice(0, 12)}</span>), so &ldquo;we did
        not move the goalposts&rdquo; is a string comparison, not a promise.
      </p>

      <SurvivorshipSection />

      <h2 id="evals">The eval suite: six published anomalies</h2>
      <p>
        Each anomaly is run on both universes and judged by four rules in code. The reasons are shown
        for every verdict, because a count of passes is a number nobody can act on.
      </p>
      {ev.universes.map((u) => (
        <div key={u} className="small" style={{ margin: "8px 0" }}>
          <strong>{UNIVERSE_LABEL[u]}:</strong> {ev.tally[u].pass} of {nTested(u)} testable anomalies
          pass
          <div className="tally">
            {(Object.keys(ev.tally[u]) as Verdict[]).map((v) => (
              <span key={v} className={`chip ${v}`}>
                {VERDICT_LABEL[v]} {ev.tally[u][v]}
              </span>
            ))}
          </div>
        </div>
      ))}
      {ev.missing.length > 0 && (
        <p className="small">
          <strong>Not yet scored:</strong> {ev.missing.join(", ")}.
        </p>
      )}
      {ev.anomalies.map((a) => (
        <AnomalyCard key={a.key} a={a} />
      ))}

      <h2 id="notes">Research notes</h2>
      <p>
        {nc.total} agent run{nc.total === 1 ? "" : "s"} stored: {nc.publishable} publishable,{" "}
        {nc.unpublishable} refused. A note is publishable only if every numeral in it traces to a
        number the pipeline produced, the run completed, and the deflation statistics were
        computed. The refused ones are kept and shown with their reasons.{" "}
        <Link href="/notes/">Read all {nc.total} &rarr;</Link>
      </p>

      <h2 id="method">Method</h2>
      <p>How a verdict is decided, in the order the rules are applied:</p>
      <ol className="rules">
        <li>
          <strong>Sign.</strong> Did the long/short spread run the way the paper predicted? Four of
          the six predict the bottom bucket wins, so the direction is fixed in advance and every
          later test uses the oriented Sharpe.
        </li>
        <li>
          <strong>Deflation.</strong> The probability that the true Sharpe beats the best of{" "}
          <em>N</em> worthless strategies must reach {s.deflation_threshold}, with <em>N</em> the
          number of anomalies the suite actually tried. A missing statistic fails the gate; it does
          not skip it.
        </li>
        <li>
          <strong>Multiplicity.</strong> Benjamini-Hochberg across the tested anomalies at &alpha; ={" "}
          {s.fdr_alpha}. Untestable anomalies leave the denominator rather than padding it.
        </li>
        <li>
          <strong>Embarrassment.</strong> A spread more than {s.embarrassment_multiple}&times; the
          published reference is downgraded to partial. On a short sample a huge number is likelier a
          defect than a discovery. This rule only ever moves a verdict down.
        </li>
      </ol>
      <p>
        <strong>Insufficient data is not a failure.</strong> An anomaly that needs three years of
        history on a two-year panel has not been refuted; it has not been tested.
      </p>
      <p className="small muted">
        Assumed variance of Sharpe across trials: {num(s.trial_variance_assumed, 4)}, labelled and
        stored with every result. Scoring rule version {s.rule_version}. Prices are split-adjusted
        only, so every return is a price return.
      </p>
    </>
  );
}
