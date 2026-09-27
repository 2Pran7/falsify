import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { REPO, snapshot } from "@/lib/data";

export const metadata: Metadata = {
  title: "falsify: an agent that tries to disprove market hypotheses",
  description:
    "Six published equity anomalies, pre-registered and scored honestly, with the failures shown.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main>
          <nav className="top">
            <Link href="/" className="brand">falsify</Link>
            <Link href="/#evals">Eval suite</Link>
            <Link href="/notes/">Research notes</Link>
            <Link href="/#method">Method</Link>
            <a href={REPO}>Source</a>
          </nav>
          {children}
          <footer>
            Frozen {snapshot.generated_at.slice(0, 10)} from commit{" "}
            <a className="mono" href={`${REPO}/tree/${(snapshot.code_commit ?? "main").replace("-dirty", "")}`}>
              {snapshot.code_commit ?? "unknown"}
            </a>
            . Registry sha256 <span className="mono">{snapshot.registry_sha.slice(0, 16)}</span>.
            Nothing on this page is computed in your browser: every figure was produced by the
            pipeline and exported by <code>scripts/export_demo.py</code>.
          </footer>
        </main>
      </body>
    </html>
  );
}
