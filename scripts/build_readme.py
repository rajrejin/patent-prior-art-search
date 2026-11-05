"""Fill docs/README.template.md from results/*.json and write README.md.

Every number and every comparison word ("higher", "no significant difference") in the headline
is computed here from the result files. Run scripts/render_tables.py first.
Usage: python scripts/build_readme.py [scope]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> Any:
    p = ROOT / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def diff_text(paired: dict[str, Any], a: str, b: str) -> str:
    """'a minus b' on nDCG@10 with interval, and whether the interval excludes zero."""
    key, sign = (f"{a}-{b}", 1) if f"{a}-{b}" in paired else (f"{b}-{a}", -1)
    if key not in paired:
        return ""
    r = paired[key]["ndcg@10"]
    d, lo, hi = sign * r["diff"], sign * r["low"], sign * r["high"]
    lo, hi = min(lo, hi), max(lo, hi)
    verdict = "significant" if lo > 0 or hi < 0 else "not significant"
    return f"{a} minus {b}: {d:+.3f} (95% interval {lo:+.3f} to {hi:+.3f}, {verdict})"


def headline(scope: str) -> str:
    ev = load(f"eval_{scope}_test.json")
    if not ev:
        return "_The test-set comparison has not been run._"
    s = ev["summary"]
    order = sorted(s, key=lambda m: -s[m]["ndcg@10"]["mean"])
    lines = [
        f"* **Retrieval** ({ev['n_queries']} held-out test queries, leakage controls on), "
        "nDCG@10: " + ", ".join(f"{m} {s[m]['ndcg@10']['mean']:.3f}" for m in order) + "."
    ]
    paired = ev.get("paired", {})
    for a, b in (
        ("hybrid", "bm25"),
        ("hybrid", "vector"),
        ("agent", "hybrid"),
        ("rerank", "hybrid"),
        ("graphrag", "hybrid"),
    ):
        t = diff_text(paired, a, b)
        if t and a in s and b in s:
            lines.append(f"  * {t}.")
    beh, mode = ev.get("agent_behaviour"), "retrieval mode"
    if not beh:
        beh, mode = (load(f"faithfulness_{scope}.json") or {}).get("agent_behaviour"), "full runs"
    if beh:
        lines.append(
            f"* **Agent** ({mode}, {beh['n_runs']} test queries): {beh['mean_steps']:.1f} steps per "
            f"run; the LLM's own action was used in {beh['planner_action_accept_rate']:.1%} of "
            f"planner calls; {beh['mean_feature_searches']:.2f} feature searches per run."
        )
    fa = load(f"faithfulness_{scope}.json")
    if fa:
        sh = fa["shipped"]
        lines.append(
            f"* **Verified claim charts** ({fa['n_test_queries']} full agent runs): "
            f"{sh['reports_with_verified_rows']:.0%} of reports contain at least one verified row, "
            f"with {sh['mean_verified_rows']:.1f} verified rows per report on average; "
            f"{sh['pair_fill_rate']:.0%} of feature and patent pairs got a sentence that both "
            "judges accept (the rest are left out and counted)."
        )
    au = load(f"faithfulness_audit_{scope}.json")
    if au:
        lines.append(
            f"* **Independent audit** by {au['judge_model']} (never used by the agent): verified rows "
            f"{au['support_rate_kept_rows']:.0%} supported, rejected rows "
            f"{au['support_rate_removed_rows']:.1%}, unrelated control "
            f"{au['support_rate_mismatched_control']:.1%}."
        )
    return "\n".join(lines)


def corpus(scope: str) -> str:
    c = load(f"corpus_stats_{scope}.json")
    if not c:
        return "not built."
    cit = c["citations"]
    return (
        f"{c['n_patents']} documents ({c['by_origin'].get('seed', 0)} seeds, "
        f"{c['by_origin'].get('cited', 0)} cited). Seeds cite {cit['total_cited_documents']} "
        f"distinct documents, {cit['coverage']:.1%} of them inside the corpus. "
        f"{c['eligible_query_patents']['count']} seeds have at least "
        f"{c['eligible_query_patents']['min_relevant']} in-corpus cited documents and are "
        "eligible queries."
    )


def main(scope: str) -> None:
    tpl = (ROOT / "docs" / "README.template.md").read_text(encoding="utf-8")
    tables = (ROOT / "results" / f"tables_{scope}.md").read_text(encoding="utf-8")
    fa = ROOT / "results" / f"failure_analysis_{scope}.md"
    if fa.exists():
        tables += "\n### Failure analysis (test)\n\n" + fa.read_text(encoding="utf-8") + "\n"
    out = (
        tpl.replace("{{HEADLINE}}", headline(scope))
        .replace("{{CORPUS}}", corpus(scope))
        .replace("{{TABLES}}", tables)
    )
    (ROOT / "README.md").write_text(out, encoding="utf-8")
    print("README.md written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
