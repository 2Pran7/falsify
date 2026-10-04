/** Shapes shared by the frozen snapshot and the live API. No imports, so
 * client components can use them without bundling data/demo.json. */
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

export interface Diagnostics {
  buckets?: { bucket: number; ann_return: number | null; n_days: number }[];
  monotonicity?: number | null;
  halves?: { first_date: string; last_date: string; sharpe: number | null; n_days: number }[];
  costs?: { cost_bps: number; sharpe: number | null; cagr: number | null }[];
  equity?: { ts: string; equity: number }[];
}

export interface BacktestView {
  variant: string;
  metrics: Record<string, unknown>;
  statistics: Record<string, unknown>;
  analysis_error: string | null;
  /** Absent on notes recorded before diagnostics existed. */
  diagnostics?: Diagnostics;
}

/** Computed by the pipeline from stored numbers, never written by the model. */
export interface VerdictCard {
  outcome: "supported" | "not_confirmed" | "contradicted" | "no_verdict";
  label: string;
  reason: string;
  variant?: string;
  prediction?: string | null;
  prediction_declared?: boolean;
  long_short?: boolean;
  sharpe?: number | null;
  prob_beats_best_of_n_trials?: number | null;
  clears_confirmation_gate?: boolean;
  n_trials?: number | null;
  n_invested_days?: number | null;
  first_date?: string | null;
  last_date?: string | null;
  universe?: string | null;
  monotonicity?: number | null;
  both_halves_as_predicted?: boolean | null;
  survives_25bps?: boolean | null;
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
  /** Absent on snapshots exported before the verdict card existed. */
  verdict?: VerdictCard;
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

