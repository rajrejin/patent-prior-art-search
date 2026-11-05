"""Ablation study: what each component contributes, with the tuned setting as reference.

Run once on the test split after tuning (or on validation for exploration). Groups:
  hybrid_weight   BM25 weight sweep (0 = vector only)
  graph_signals   each graph signal alone on top of the tuned hybrid
  rerank          cross-encoder model and depth on top of the hybrid
  vector          chunk aggregation and indexed text sections of the vector retriever
"""

from __future__ import annotations

import json
from typing import Any

from patentrag.config import Config
from patentrag.evaluation.harness import ensure_testset, subsample
from patentrag.evaluation.params import RESULTS, load_tuned_hybrid, load_tuned_rerank
from patentrag.evaluation.testset import build_indexes
from patentrag.evaluation.tuning import RERANK_DEPTHS, Scorer, hybrid_start
from patentrag.retrieval.base import Retriever
from patentrag.retrieval.vector import VectorRetriever

GRAPH_SIGNALS: dict[str, dict[str, Any]] = {
    "citation vote (w 0.2)": {"use_cite": True, "w_cite": 0.2},
    "CPC overlap (w 0.1)": {"w_cpc": 0.1},
    "CPC subtree candidates + overlap": {"use_cpc": True, "w_cpc": 0.1},
    "family (w 0.1)": {"use_family": True, "w_family": 0.1},
    "all four": {
        "use_cite": True,
        "w_cite": 0.2,
        "use_cpc": True,
        "w_cpc": 0.1,
        "use_family": True,
        "w_family": 0.1,
    },
}


def run_ablations(cfg: Config, scope: str, stack: Any, split: str) -> dict[str, Any]:
    patents = list(stack.patents.values())
    val, test, _ = ensure_testset(cfg, scope, patents)
    queries = subsample(val if split == "val" else test, cfg.seed)
    sc = Scorer(cfg, build_indexes(patents).key_of, queries)
    h = load_tuned_hybrid(cfg, scope)
    rows: dict[str, Any] = {}

    def add(group: str, label: str, retriever: Retriever) -> None:
        r = sc.score(f"{group}/{label}", retriever)
        rows.setdefault(group, {})[label] = {"summary": r["summary"], "latency_ms": r["latency_ms"]}

    for w in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        tag = f"w_bm25={w}" + (" (tuned)" if w == h.w_bm25 else "")
        add("hybrid_weight", tag, stack.hybrid(h.model_copy(update={"w_bm25": w})))
    add("hybrid_weight", "BM25 only", stack.bm25())

    start = hybrid_start(cfg.graph, h)
    add("graph_signals", "none (= hybrid)", stack.graphrag(start))
    for label, upd in GRAPH_SIGNALS.items():
        add("graph_signals", label, stack.graphrag(start.model_copy(update=upd)))

    tuned_r = load_tuned_rerank(cfg, scope)
    for model in cfg.rerank.candidates:
        for depth in RERANK_DEPTHS:
            p = tuned_r.model_copy(update={"model": model, "depth": depth, "enabled": True})
            short = model.split("/")[-1]
            add("rerank", f"{short}, top {depth}, alpha {p.alpha}", stack.rerank(p))

    vec = stack.vector
    for agg in ("max", "top3sum"):
        add("vector", f"aggregation {agg}", VectorRetriever(vec.embedder, vec.store, agg))
    add(
        "vector",
        "abstract-only chunks",
        VectorRetriever(vec.embedder, vec.store, "max", {"abstract"}),
    )
    out = {
        "split": split,
        "n_queries": len(queries),
        "hybrid": h.model_dump(),
        "rerank": tuned_r.model_dump(),
        "ablations": rows,
    }
    (RESULTS / f"ablations_{scope}_{split}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )
    return out
