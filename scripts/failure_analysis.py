"""Failure analysis from results/eval_<scope>_test.json (per-query scores); no model needed.

Compares the main system (the agent if it was evaluated, else the re-ranked hybrid that the
agent searches with) with BM25 per query and writes results/failure_analysis_<scope>.json
and .md. Every sentence in the .md is computed from the scores; nothing is hand-written.
Usage: python scripts/failure_analysis.py [scope]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(scope: str) -> None:
    ev = json.loads((ROOT / "results" / f"eval_{scope}_test.json").read_text(encoding="utf-8"))
    ts = json.loads(
        (ROOT / "results" / f"testset_manifest_{scope}.json").read_text(encoding="utf-8")
    )
    queries = {q["qid"]: q for q in ts["test"]}
    pq = ev["per_query"]
    main_m = next(m for m in ("agent", "rerank", "hybrid") if m in pq)
    base_m = "bm25"
    rows = []
    for qid, q in queries.items():
        a, b = pq[main_m][qid]["ndcg@10"], pq[base_m][qid]["ndcg@10"]
        rows.append(
            {
                "qid": qid,
                "year": q["year"],
                "subclass": q["subclass"],
                "n_relevant": len(q["relevant_keys"]),
                main_m: a,
                base_m: b,
                "diff": a - b,
            }
        )
    n = len(rows)
    zero = sum(r[main_m] == 0 for r in rows)
    zero_both = sum(r[main_m] == 0 and r[base_m] == 0 for r in rows)
    wins = sum(r["diff"] > 0 for r in rows)
    losses = sum(r["diff"] < 0 for r in rows)
    by_n: dict[str, list[float]] = {}
    by_sub: dict[str, list[float]] = {}
    for r in rows:
        k = "3" if r["n_relevant"] == 3 else "4-5" if r["n_relevant"] <= 5 else "6+"
        by_n.setdefault(k, []).append(r[main_m])
        by_sub.setdefault(r["subclass"], []).append(r[main_m])
    worst = sorted(rows, key=lambda r: (r[main_m], r["qid"]))[:10]
    out = {
        "n_queries": n,
        "main_method": main_m,
        "baseline": base_m,
        "main_ndcg_zero": zero,
        "both_zero": zero_both,
        "main_beats_baseline": wins,
        "main_loses_to_baseline": losses,
        "mean_main_ndcg_by_num_relevant": {
            k: {"mean": sum(x) / len(x), "n": len(x)} for k, x in sorted(by_n.items())
        },
        "mean_main_ndcg_by_subclass": {
            k: {"mean": sum(x) / len(x), "n": len(x)}
            for k, x in sorted(by_sub.items())
            if len(x) >= 10
        },
        "worst_10": worst,
    }
    (ROOT / "results" / f"failure_analysis_{scope}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    lines = [
        f"Queries: {n}. {main_m} nDCG@10 is zero on {zero}; {main_m} and {base_m} are both zero "
        f"on {zero_both}.",
        f"{main_m} scores higher than {base_m} on {wins} queries, lower on {losses}, equal on "
        f"{n - wins - losses}.",
        "",
        f"Mean {main_m} nDCG@10 by number of relevant documents: "
        + ", ".join(
            f"{k}: {v['mean']:.3f} (n={v['n']})"
            for k, v in out["mean_main_ndcg_by_num_relevant"].items()
        ),
        "",
        f"Mean {main_m} nDCG@10 by CPC subclass (subclasses with at least 10 queries): "
        + ", ".join(
            f"{k}: {v['mean']:.3f} (n={v['n']})"
            for k, v in out["mean_main_ndcg_by_subclass"].items()
        ),
        "",
        f"| 10 lowest {main_m} queries | year | subclass | relevant | {main_m} | {base_m} |",
        "|---|---|---|---|---|---|",
    ]
    for r in worst:
        lines.append(
            f"| {r['qid']} | {r['year']} | {r['subclass']} | {r['n_relevant']} | "
            f"{r[main_m]:.3f} | {r[base_m]:.3f} |"
        )
    (ROOT / "results" / f"failure_analysis_{scope}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
