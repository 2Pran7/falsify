import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { REPO, snapshot } from "@/lib/data";

export const metadata: Metadata = {
  title: "falsify · a research agent built to disprove its own hypotheses",
  description:
    "Six published equity anomalies, pre-registered and scored with deflated Sharpe and multiple-testing correction. The failures are shown.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const commit = snapshot.code_commit ?? "main";
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <div className="wrap">
            <Link href="/" className="brand">
              <span className="brand-mark">f</span>falsify
            </Link>
            <nav className="nav">
              <Link href="/#evals">Eval suite</Link>
              <Link href="/notes/">Notes</Link>
              <Link href="/#method">Method</Link>
              <a href={REPO}>GitHub</a>
              <Link href="/try/" className="cta">Try it live</Link>
            </nav>
          </div>
        </header>
        <main>
          <div className="wrap">{children}</div>
        </main>
        <footer className="site">
          <div className="wrap">
            <span>
              Evidence frozen {snapshot.generated_at.slice(0, 10)} from commit{" "}
              <a className="mono" href={`${REPO}/tree/${commit.replace("-dirty", "")}`}>{commit}</a> · pre-registration sha256{" "}
              <span className="mono">{snapshot.registry_sha.slice(0, 12)}</span>
            </span>
            <span>Nothing on the published pages is computed in your browser. Built by Pranshu Sheel.</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
