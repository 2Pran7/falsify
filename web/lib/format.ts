/** Formatting shared by server and client components. */
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
  return iso ? iso.slice(0, 10) : "n/a";
}
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
