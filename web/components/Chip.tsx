const LABEL: Record<string, string> = {
  pass: "Pass", partial: "Partial", fail: "Fail", insufficient_data: "Insufficient data",
  none: "Not run", queued: "Queued", running: "Running", done: "Done", failed: "Failed",
  publishable: "Publishable", refused: "Refused",
};
const CLASS: Record<string, string> = { publishable: "pass", refused: "fail" };

export function Chip({ kind }: { kind: string }) {
  return <span className={`chip ${CLASS[kind] ?? kind}`}>{LABEL[kind] ?? kind}</span>;
}
