"""Pick one full agent run from the test reports as a worked example.

Chooses the run with the most verified chart rows (ties: fewest removed rows, then query id)
and writes results/example_agent_run.json (gitignored: it contains patent text) with every
step, the features and the verified chart. Observations are shortened to keep it readable.
Usage: python scripts/example_run.py [scope]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(scope: str) -> None:
    path = RESULTS / f"agent_report_test_{scope}.jsonl"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def rows(r: dict) -> int:
        return sum(len(i.get("chart", [])) for i in (r.get("report") or {}).get("items", []))

    best = sorted(
        recs, key=lambda r: (-rows(r), (r.get("report") or {}).get("removed_rows", 0), r["qid"])
    )[0]
    rep = best["report"] or {}
    out = {
        "qid": best["qid"],
        "idea": " ".join(best["query"].split()[:120]),
        "steps": [
            {
                "n": s["n"],
                "tool": s["tool"],
                "source": s["source"],
                "thought": s["thought"],
                "args": s["args"],
                "error": s["error"],
                "observation": s["observation"][:220],
            }
            for s in best["steps"]
        ],
        "features": rep.get("features", []),
        "chart": [
            {"pub_id": i["pub_id"], "title": i["title"], "rows": i["chart"]}
            for i in rep.get("items", [])
        ],
        "removed_rows": rep.get("removed_rows", 0),
        "stats": best["stats"],
    }
    (RESULTS / "example_agent_run.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"example: {best['qid']} with {rows(best)} verified rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
