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

