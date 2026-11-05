"""End-to-end smoke run against a running API: readiness, search with every method, one agent
run (submitted as a background job and polled) and its trace. Exits non-zero if any step fails.

Usage: python scripts/e2e_smoke.py [api_url]     (default http://localhost:8000)
The query is the title and abstract of a real patent from the corpus, so it needs the corpus
file locally; the API is called over HTTP exactly as the UI calls it.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx

from patentrag.config import ROOT, load_config
from patentrag.ingest.corpus import load_corpus


def main(url: str) -> int:
    seed = next(p for p in load_corpus(load_config(), "full") if p.origin == "seed" and p.abstract)
    query = seed.text_for_query()
    out: dict[str, object] = {"api": url, "query_patent": str(seed.pub_id)}
    searches: dict[str, bool] = {}
    with httpx.Client(base_url=url, timeout=1800) as c:
        ready = c.get("/v1/ready")
        out["ready"] = {"status": ready.status_code, **ready.json()}
        for method in ("auto", "bm25", "vector", "hybrid", "rerank", "graphrag"):
            r = c.post("/v1/search", json={"query": query, "k": 5, "method": method})
            body = r.json()
            ok = r.status_code == 200 and len(body.get("hits", [])) == 5
            searches[method] = ok
            out[f"search_{method}"] = {
                "ok": ok,
                "method_used": body.get("method"),
                "ms": body.get("took_ms"),
                "top": [h["pub_id"] for h in body.get("hits", [])][:3],
            }
        t0 = time.perf_counter()
        job = c.post("/v1/report/jobs", json={"query": query}).json()["job_id"]
        state: dict[str, Any] = {"status": "queued"}
        while state["status"] not in ("done", "error") and time.perf_counter() - t0 < 3600:
            time.sleep(5)
            state = c.get(f"/v1/report/jobs/{job}").json()
        body = state.get("result", {})
        agent_status = 200 if state["status"] == "done" else 500
        out["agent"] = {
            "status": state["status"],
            "seconds": round(time.perf_counter() - t0, 1),
            "steps": [(s["tool"], s["source"]) for s in body.get("agent_steps", [])],
            "verified_items": len(body.get("items", [])),
            "removed_rows": body.get("removed_rows"),
            "warning": body.get("warning"),
        }
        tid = body.get("trace_id", "")
        trace = c.get(f"/v1/trace/{tid}") if tid else None
        trace_ok = trace is not None and trace.status_code == 200
        out["trace_ok"] = trace_ok
    dest = ROOT / "results" / "e2e_smoke_full.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    searches_ok = all(searches.values())
    return 0 if searches_ok and agent_status == 200 and trace_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
