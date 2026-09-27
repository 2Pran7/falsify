/**
 * The one input to the whole site: data/demo.json, written by
 * scripts/export_demo.py from what Postgres holds.
 *
 * If the file is absent the build FAILS rather than rendering a sample. A demo
 * that falls back to placeholder numbers is a demo that can ship them.
 */
import raw from "@/data/demo.json";

import type { AnomalyRow, NoteView, Snapshot, Universe, Verdict } from "@/lib/types";
export type * from "@/lib/types";

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

export { day, num, pct, signed } from "@/lib/format";
