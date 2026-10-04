"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { NoteBody } from "@/components/NoteBody";
import { API_URL } from "@/lib/format";
import type { NoteView } from "@/lib/types";

/** One question asked on the live page. Read from the server, because the
 * site is static and a visitor's run is newer than the last build. */
export function LiveNote() {
  const [n, setN] = useState<NoteView | null>(null);
  const [state, setState] = useState<"loading" | "missing" | "offline">("loading");

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("id") ?? "";
    if (!API_URL || !id) { setState("missing"); return; }
    let stop = false;
    (async () => {
      for (let i = 0; i < 4 && !stop; i++) {
        try {
          const r = await fetch(`${API_URL}/notes/live/${encodeURIComponent(id)}`);
          if (r.status === 404) { setState("missing"); return; }
          if (r.ok) { const body = await r.json(); if (!stop) setN(body); return; }
        } catch {}
        await new Promise((res) => setTimeout(res, 8_000));
      }
      if (!stop) setState("offline");
    })();
    return () => { stop = true; };
  }, []);

  return (
    <>
      <p className="small"><Link href="/notes/">← All notes</Link></p>
      {!n ? (
        <p className="muted" style={{ marginTop: 24 }}>
          {state === "loading" && "Loading the run. If the server was idle this takes up to a minute…"}
          {state === "missing" && "This run isn't on the notes page. It may have been removed."}
          {state === "offline" && "The live server didn't answer. Try again in a minute."}
        </p>
      ) : (
        <>
          <div className="eyebrow" style={{ marginTop: 20 }}>Asked on the live page</div>
          <h1 className="display" style={{ fontSize: "clamp(1.7rem,3.6vw,2.4rem)" }}>{n.hypothesis}</h1>
          <div className="meta">
            <span>{n.created_at.slice(0, 16).replace("T", " ")} UTC</span><span>{n.run.model}</span><span>${n.run.cost_usd.toFixed(3)}</span>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            Live runs use the latest ~3 years of prices so they fit a free server. The research notes use all five.
          </p>
          <div style={{ marginTop: 28 }}><NoteBody n={n} /></div>
        </>
      )}
    </>
  );
}
