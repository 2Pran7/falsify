/**
 * The one input to the whole site: data/demo.json, written by
 * scripts/export_demo.py from what Postgres holds.
 *
 * If the file is absent the build FAILS rather than rendering a sample. A demo
 * that falls back to placeholder numbers is a demo that can ship them.
 */
import raw from "@/data/demo.json";

export type Verdict = "pass" | "partial" | "fail" | "insufficient_data";
export type Universe = "current" | "point_in_time";

export interface Result {
  verdict: Verdict;
  reasons: string[];
  realised_sharpe: number | null;
  oriented_sharpe: number | null;
  published_sharpe: number;
  p_value: number | null;
  p_value_adjusted: number | null;
  deflated_psr: number | null;
  n_invested_days: number;
  history_days: number;
  min_history_days: number;
  n_trials: number;
  run_at: string;
}

export interface AnomalyRow {
  key: string;
  feature: string;
  direction: 1 | -1;
  published_sharpe: number;
  min_history_days: number;
  citation: string;
  hypothesis: string;
  sharpe_source: string;
  caveat: string;
  results: Record<Universe, Result | null>;
  survivorship_sharpe_gap: number | null;
}

export interface BacktestView {
  variant: string;
  metrics: Record<string, unknown>;
  statistics: Record<string, unknown>;
  analysis_error: string | null;
}

export interface NoteView {
  note_id: string;
  eval_key: string | null;
  created_at: string;
  hypothesis: string;
  prose: string;
  publishable: boolean;
  unpublishable_reasons: string[];
  backtests: BacktestView[];
  provenance: {
    ok: boolean;
    checked: number;
    verified: number;
    exempt: number;
    unverified: string[];
  };
  run: {
    model: string;
    stop_reason: string;
    turns: number;
    tool_sequence: string[];
    n_backtests: number;
    total_tokens: number;
    cost_usd: number;
  };
  assumptions: Record<string, unknown>;
}

export interface Survivorship {
  generated_at: string;
  window: { first: string; last: string; invested_days: number };
  gate: string;
  n_tickers_gated: number;
  bound: string;
  metrics: Record<string, number>;
}

export interface Snapshot {
  snapshot_version: number;
  generated_at: string;
  code_commit: string | null;
  registry_sha: string;
  scoring: {
    rule_version: number;
    deflation_threshold: number;
    fdr_alpha: number;
    embarrassment_multiple: number;
    trial_variance_assumed: number | null;
  };
  evals: {
    universes: Universe[];
    last_run_at: string | null;
    tally: Record<Universe, Record<Verdict, number>>;
    missing: string[];
    anomalies: AnomalyRow[];
  };
  notes: {
    counts: { total: number; publishable: number; unpublishable: number };
    items: NoteView[];
  };
  survivorship: Survivorship | null;
}

export const snapshot = raw as unknown as Snapshot;

if (snapshot.snapshot_version !== 1) {
  throw new Error(
    `data/demo.json is snapshot_version ${snapshot.snapshot_version}; this site renders version 1. ` +
      "Re-export with scripts/export_demo.py or update web/lib/data.ts.",
  );
}

export const REPO = "https://github.com/2Pran7/falsify";

export const TITLES: Record<string, string> = {
  momentum_12_1: "Momentum (12-1)",
  short_term_reversal: "Short-term reversal",
  long_term_reversal: "Long-term reversal",
  low_volatility: "Low volatility",
  idiosyncratic_volatility: "Idiosyncratic volatility",
  fifty_two_week_high: "52-week high",
};

export const title = (key: string) => TITLES[key] ?? key.replaceAll("_", " ");

export const UNIVERSE_LABEL: Record<Universe, string> = {
  current: "Today's constituents",
  point_in_time: "Point-in-time membership",
};

export const VERDICT_LABEL: Record<Verdict, string> = {
  pass: "Pass",
  partial: "Partial",
  fail: "Fail",
  insufficient_data: "Insufficient data",
};

export function num(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "n/a";
  return x.toFixed(digits);
}

export function signed(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "n/a";
  return (x > 0 ? "+" : "") + x.toFixed(digits);
}

export function pct(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "n/a";
  return (x * 100).toFixed(digits) + "%";
}

export function day(iso: string | null | undefined): string {
  if (!iso) return "n/a";
  return iso.slice(0, 10);
}
