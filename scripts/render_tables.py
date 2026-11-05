"""Render markdown tables from results/*.json into results/tables_<scope>.md.

Every number in the README and the guide is copied from this output; nothing is typed by hand.
Works on the full result files and on the compact copies (it reads summaries only).
Usage: python scripts/render_tables.py [scope]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RES = Path(__file__).resolve().parents[1] / "results"
KS = [5, 10, 20]


def cell(s: dict[str, float]) -> str:
    return f"{s['mean']:.3f} [{s['low']:.3f}, {s['high']:.3f}]"


def load(name: str) -> Any:
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def comparison(scope: str, split: str = "test") -> list[str]:
    d = load(f"eval_{scope}_{split}.json")
    if not d:
        return ["_No test-set evaluation has been run for this scope._"]
    cols = ["ndcg@10", "mrr"] + [f"r@{k}" for k in KS] + [f"p@{k}" for k in (5, 10)]
    note = (
        ""
        if "agent" in d["summary"]
        else " The agent has no row of its own: with the validation-chosen feature weight of 0 "
        "its ranking is its whole-idea search, the re-ranked hybrid (`rerank`)."
    )
    out = [
        f"Test queries: {d['n_queries']}. Cells: mean [95% bootstrap interval]. Latency is per "
        "query on the evaluation VM (results/environment_colab.json); this comparison ran "
        "without a GPU, which makes the cross-encoder slow." + note,
        "",
        "| method | " + " | ".join(cols) + " | latency p50 (ms) |",
        "|" + "---|" * (len(cols) + 2),
    ]
    for m, s in d["summary"].items():
        lat = d["latency_ms"][m]
        out.append(f"| {m} | " + " | ".join(cell(s[c]) for c in cols) + f" | {lat['p50']:.0f} |")
    out += ["", "Paired difference (first minus second), 95% interval, bootstrap p-value:", ""]
    out += ["| pair | nDCG@10 difference | interval | p |", "|---|---|---|---|"]
    for pair, ms in d.get("paired", {}).items():
        r = ms["ndcg@10"]
        out.append(
            f"| {pair} | {r['diff']:+.3f} | [{r['low']:+.3f}, {r['high']:+.3f}] | {r['p']:.3f} |"
        )
    return out


def agent_behaviour(scope: str) -> list[str]:
    """The agent's decisions: full test runs (with claim charts) if available, else the
    retrieval-mode runs on the validation queries that chose its fusion weight."""
    b = (load(f"faithfulness_{scope}.json") or {}).get("agent_behaviour")
    label = "Full agent runs (with claim charts) on test queries"
    if not b:
        b = ((load(f"tuned_params_{scope}.json") or {}).get("agent") or {}).get("behaviour_val")
        label = "Retrieval-mode agent runs on the validation queries"
    if not b:
        return ["_The agent was not evaluated._"]
    tools = ", ".join(f"{t} {n}" for t, n in b["tool_use"].items())
    return [
        f"{label}: {b['n_runs']}.",
        "",
        "| measure | value |",
        "|---|---|",
        f"| mean steps per run | {b['mean_steps']:.1f} |",
        f"| planner calls | {b['planner_calls']} |",
        f"| planner actions used as chosen | {b['planner_action_accept_rate']:.3f} |",
        f"| actions replaced by the fallback policy | {b['replaced_actions']} |",
        f"| steps forced by guardrails | {b['guardrail_steps']} |",
        f"| planner outputs that were not valid JSON | {b['planner_output_errors']} |",
        f"| mean feature searches per run | {b['mean_feature_searches']:.2f} |",
        f"| mean seconds per run (wall time, runs sharing one GPU) | {b['mean_seconds']:.1f} |",
        "",
        f"Tool calls in total: {tools}.",
    ]


def leakage(scope: str) -> list[str]:
    out: list[str] = []
    for name, label in (
        (f"eval_{scope}_leak_demo.json", "tuned parameters"),
        (f"eval_{scope}_leak_demo_default.json", "default parameters (citation weight 0.2)"),
    ):
        d = load(name)
        if not d:
            continue
        out += [
            f"GraphRAG with {label}, validation queries: {d['n_queries']}.",
            "",
            "| run | nDCG@10 | recall@20 |",
            "|---|---|---|",
        ]
        for m in ("graphrag", "graphrag_leaky"):
            s = d["summary"][m]
            out.append(f"| {m} | {cell(s['ndcg@10'])} | {cell(s['r@20'])} |")
        out.append("")
    return out or ["_Leakage demonstration not run._"]


def tuning(scope: str) -> list[str]:
    d = load(f"tuned_params_{scope}.json")
    if not d:
        return ["_Tuning not run._"]
    out: list[str] = []
    h = d.get("hybrid")
    if h:
        out += [
            f"**Hybrid** ({h['n_val_queries']} validation queries; vector weight 1, BM25 weight "
            f"varied). Chosen BM25 weight: {h['params']['w_bm25']}.",
            "",
            "| BM25 weight | validation nDCG@10 |",
            "|---|---|",
        ]
        out += [f"| {r['w_bm25']} | {r['ndcg@10']:.3f} |" for r in h["search_log"]]
        out.append("")
    g = d.get("graph")
    if g:
        out += [
            "**Graph signals on top of the hybrid** (greedy; a signal is kept only if it raises "
            "validation nDCG@10).",
            "",
            "| step | value | validation nDCG@10 | kept |",
            "|---|---|---|---|",
        ]
        out += [
            f"| {r['param']} | {r['value']} | {r['ndcg@10']:.3f} | "
            f"{'yes' if r['chosen'] else 'no'} |"
            for r in g["search_log"]
        ]
        out.append("")
    r = d.get("rerank")
    if r:
        p = r["params"]
        verdict = "enabled" if p["enabled"] else "not enabled (no gain over the hybrid)"
        out += [
            f"**Cross-encoder re-ranking** (hybrid alone: {r['hybrid_val_ndcg@10']:.3f}). Best: "
            f"{p['model']}, top {p['depth']}, alpha {p['alpha']}: {r['val_ndcg@10']:.3f}; "
            f"{verdict}.",
            "",
            "| model | depth | alpha | validation nDCG@10 |",
            "|---|---|---|---|",
        ]
        out += [
            f"| {x['model'].split('/')[-1]} | {x['depth']} | {x['alpha']} | {x['ndcg@10']:.3f} |"
            for x in r["search_log"]
        ]
        out.append("")
    a = d.get("agent")
    if a:
        out += [
            f"**Agent feature-search weight** ({a['n_val_queries']} validation queries; 0 means "
            f"the ranking of the whole-idea search alone). Chosen: "
            f"{a['params']['feature_weight']}.",
            "",
            "| feature weight | validation nDCG@10 | recall@10 |",
            "|---|---|---|",
        ]
        out += [
            f"| {x['feature_weight']} | {x['ndcg@10']:.3f} | {x['r@10']:.3f} |"
            for x in a["search_log"]
        ]
    return out


def ablations(scope: str, split: str) -> list[str]:
    d = load(f"ablations_{scope}_{split}.json")
    if not d:
        return ["_Ablations not run._"]
    out = [f"Split: {split}, queries: {d['n_queries']}.", ""]
    for group, rows in d["ablations"].items():
        out += [f"**{group}**", "", "| setting | nDCG@10 | recall@10 | MRR |", "|---|---|---|---|"]
        for label, r in rows.items():
            s = r["summary"]
            out.append(f"| {label} | {cell(s['ndcg@10'])} | {cell(s['r@10'])} | {cell(s['mrr'])} |")
        out.append("")
    return out


def faithfulness(scope: str) -> list[str]:
    d = load(f"faithfulness_{scope}.json")
    if not d:
        return [
            "_Pending: the claim-chart reports with retrieved evidence (docs/DECISIONS.md, D28) "
            "and their audit run on a GPU and are added here when they finish._"
        ]
    b, a, ft, sh = d["first_choice"], d["after_retries"], d["features"], d["shipped"]
    out = [
        f"Full agent runs on {d['n_test_queries']} test queries; model {d['model']}, NLI "
        f'threshold {d["nli_threshold"]} (chosen on validation reports). A row says "this '
        'sentence of the patent states this feature of the idea"; for each feature and charted '
        "patent the sentences are tried in embedding-similarity order, up to three.",
        "",
        "| chart rows (feature and patent pairs) | first-ranked sentence | after up to 3 tries |",
        "|---|---|---|",
        f"| rows judged | {b['n_claims']} | {a['n_claims']} |",
        f"| supported, LLM judge | {b['support_rate_llm']:.3f} | {a['support_rate_llm']:.3f} |",
        f"| supported, NLI judge | {b['support_rate_nli']:.3f} | {a['support_rate_nli']:.3f} |",
        f"| supported, both judges | {b['support_rate_combined']:.3f} | "
        f"{a['support_rate_combined']:.3f} |",
        f"| agreement LLM vs NLI (Cohen's kappa) | {b['kappa_llm_vs_nli'] or 0:.3f} | "
        f"{a['kappa_llm_vs_nli'] or 0:.3f} |",
        "",
        f"Evidence sentences found verbatim in the cited patent: "
        f"{b['evidence_from_cited_patent']:.3f}. Features supported by the idea (both judges): "
        f"{ft['support_rate_combined']:.3f} of {ft['n_claims']}.",
        "",
        "| returned report (verified rows only) | value |",
        "|---|---|",
        f"| reports with at least one verified row | {sh['reports_with_verified_rows']:.3f} |",
        f"| mean verified patents per report | {sh['mean_verified_items']:.2f} |",
        f"| mean verified rows per report | {sh['mean_verified_rows']:.2f} |",
        f"| feature and patent pairs with a verified row | {sh['pair_fill_rate']:.3f} |",
        f"| mean features per idea | {sh['mean_features']:.2f} |",
    ]
    nc = d.get("negative_control", {})
    if "n_claims" in nc:
        out += [
            "",
            "Negative control (each row judged against a random sentence of another query):",
            "",
            "| judge | matched evidence | mismatched sentence |",
            "|---|---|---|",
            f"| LLM | {nc['matched_support_rate_llm']:.3f} | "
            f"{nc['mismatched_support_rate_llm']:.3f} |",
            f"| NLI | {nc['matched_support_rate_nli']:.3f} | "
            f"{nc['mismatched_support_rate_nli']:.3f} |",
            "",
            f"Rows in the control: {nc['n_claims']}. A judge that tells support from non-support "
            "accepts far fewer rows against the unrelated sentence than against the row's own "
            "evidence; compare the two columns.",
        ]
    au = load(f"faithfulness_audit_{scope}.json")
    if au:
        out += [
            "",
            f"Independent audit by {au['judge_model']} (not used by the agent):",
            "",
            "| rows | audited support rate | rows |",
            "|---|---|---|",
            f"| verified (shown to the user) | {au['support_rate_kept_rows']:.3f} | "
            f"{au['n_rows_kept']} |",
            f"| rejected after every try | {au['support_rate_removed_rows']:.3f} | "
            f"{au['n_rows_removed']} |",
            f"| mismatched control | {au['support_rate_mismatched_control']:.3f} | "
            f"{au['n_control']} |",
            "",
            f"Audited support of the extracted features against the idea: "
            f"{au['feature_support_rate']:.3f}.",
        ]
    beh = d.get("agent_behaviour")
    if beh:
        out += [
            "",
            f"Agent in report mode: {beh['mean_steps']:.1f} steps and {beh['mean_seconds']:.0f} s "
            f"per run on average (wall time, runs sharing one GPU); "
            f"planner actions used as chosen: "
            f"{beh['planner_action_accept_rate']:.3f}.",
        ]
    return out


def robustness(scope: str) -> list[str]:
    d = load(f"robustness_{scope}.json")
    if not d:
        return ["_Robustness study not run._"]
    out = [f"**Query form** ({d['query_form']['n_queries']} queries), nDCG@10", ""]
    out += ["| form / method | nDCG@10 |", "|---|---|"]
    for k, v in d["query_form"]["results"].items():
        out.append(f"| {k} | {cell(v['summary']['ndcg@10'])} |")
    out += ["", f"**Language** ({d['language']['n_queries']} queries with both titles)", ""]
    out += ["| title / method | nDCG@10 |", "|---|---|"]
    for k, v in d["language"]["results"].items():
        out.append(f"| {k} | {cell(v['summary']['ndcg@10'])} |")
    out += ["", "**Corpus size** (documents kept at query time), nDCG@10", ""]
    methods = list(next(iter(d["corpus_size"].values()))["results"])
    out += ["| fraction | documents | queries | " + " | ".join(methods) + " |"]
    out += ["|" + "---|" * (3 + len(methods))]
    for frac, v in d["corpus_size"].items():
        vals = " | ".join(f"{v['results'][m]['summary']['ndcg@10']['mean']:.3f}" for m in methods)
        out.append(f"| {frac} | {v['n_documents']} | {v['n_queries']} | {vals} |")
    return out


def main(scope: str) -> None:
    sections = [
        ("Comparison (test)", comparison(scope)),
        ("Agent behaviour", agent_behaviour(scope)),
        ("Tuning (validation)", tuning(scope)),
        ("Leakage demonstration (validation)", leakage(scope)),
        ("Ablations (test)", ablations(scope, "test")),
        ("Faithfulness of the agent's claim charts (test)", faithfulness(scope)),
        ("Robustness (test)", robustness(scope)),
    ]
    text = "\n\n".join(f"### {t}\n\n" + "\n".join(lines) for t, lines in sections) + "\n"
    out = RES / f"tables_{scope}.md"
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
