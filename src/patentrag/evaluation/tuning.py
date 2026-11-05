"""Tuning on validation queries only. Test queries are never loaded here.

Stage 1  hybrid:  BM25 weight of the hybrid retriever (vector weight fixed at 1).
Stage 2  graph:   graph signals added on top of the chosen hybrid, one at a time (greedy). The
                  start setting ranks exactly like the hybrid, so a signal is kept only if it
                  raises validation nDCG@10.
Stage 3  rerank:  cross-encoder model, depth and blend weight on top of the hybrid; enabled only
                  if it beats the hybrid on validation.
The agent's own parameters are tuned in `evaluation/agent_eval.py`.
"""

from __future__ import annotations

import logging
from typing import Any

from patentrag.config import Config, GraphCfg, HybridCfg, RerankCfg
from patentrag.evaluation import runner
from patentrag.evaluation.harness import ensure_testset, subsample
from patentrag.evaluation.metrics import query_metrics
from patentrag.evaluation.params import load_tuned_hybrid, save_section
from patentrag.evaluation.testset import Query, build_indexes
from patentrag.retrieval.base import Retriever
from patentrag.retrieval.reranker import blend

log = logging.getLogger(__name__)
METRICS = ["ndcg@10", "r@10", "mrr"]


class Scorer:
    """Scores retrievers on a fixed query list; results are cached by label."""

    def __init__(self, cfg: Config, key_of: dict[str, str], queries: list[Query]):
        self.cfg = cfg
        self.queries = queries
        self.key_of = key_of
        self.cache: dict[str, dict[str, Any]] = {}

    def score_ranked(self, label: str, ranked: dict[str, list[str]], latency: list[float]) -> dict:
        per_q = {
            q.qid: query_metrics(
                ranked.get(q.qid, []), set(q.relevant_keys), self.cfg.evaluation.ks
            )
            for q in self.queries
        }
        summ = runner.summarise(per_q, self.cfg.evaluation.bootstrap_resamples, self.cfg.seed)
        out = {
            "label": label,
            "summary": {m: summ[m] for m in METRICS},
            "latency_ms": runner.latency_summary(latency),
            "per_query": per_q,
        }
        self.cache[label] = out
        log.info("%s: ndcg@10=%.4f", label, summ["ndcg@10"]["mean"])
        return out

    def score(self, label: str, retriever: Retriever) -> dict[str, Any]:
        if label in self.cache:
            return self.cache[label]
        run = runner.run_method(retriever, self.queries, self.key_of, label=label)
        return self.score_ranked(label, run.ranked, run.latency_ms)


def mean(res: dict[str, Any], metric: str = "ndcg@10") -> float:
    return float(res["summary"][metric]["mean"])


HYBRID_GRID: list[float] = [0.25, 0.5, 1.0, 2.0, 4.0]

# Each option is a set of parameter updates; `needs` names a parameter that must already be on.
GRAPH_GRID: list[tuple[str, list[dict[str, Any]], str | None]] = [
    ("w_cite", [{"use_cite": True, "w_cite": w} for w in (0.1, 0.2, 0.4, 0.8)], None),
    ("citation_hops", [{"citation_hops": 2}], "use_cite"),
    ("w_cpc", [{"w_cpc": w} for w in (0.05, 0.1, 0.2)], None),
    ("use_cpc", [{"use_cpc": True}], "w_cpc"),
    ("cpc_level", [{"cpc_level": "group", "cpc_depth": 1}], "w_cpc"),
    ("cpc_top", [{"cpc_top": 3}], "w_cpc"),
    ("seeds_n", [{"seeds_n": 5}, {"seeds_n": 20}], None),
    ("w_family", [{"use_family": True, "w_family": 0.1}], None),
    ("w_applicant", [{"w_applicant": 0.1}], None),
]

RERANK_DEPTHS = [20, 50]
RERANK_ALPHAS = [1.0, 0.7, 0.4]


def hybrid_start(base: GraphCfg, h: HybridCfg) -> GraphCfg:
    """GraphRAG settings that rank exactly like the hybrid retriever (graph switched off)."""
    return base.model_copy(
        update={
            "seed_source": "hybrid",
            "w_sim": h.w_sim,
            "w_bm25": h.w_bm25,
            "pool_n": h.pool_n,
            "bm25_pool": h.pool_n,
            "seeds_n": 10,
            "cpc_level": "subclass",
            "cpc_depth": 2,
            "cpc_top": 1,
            "citation_hops": 1,
            "use_cpc": False,
            "use_cite": False,
            "use_family": False,
            "use_rerank": True,
            "w_cpc": 0.0,
            "w_cite": 0.0,
            "w_family": 0.0,
            "w_applicant": 0.0,
        }
    )


def validation(cfg: Config, scope: str, stack: Any) -> tuple[list[Query], Scorer]:
    patents = list(stack.patents.values())
    val, _, _ = ensure_testset(cfg, scope, patents)
    val = subsample(val, cfg.seed)
    return val, Scorer(cfg, build_indexes(patents).key_of, val)


def tune_hybrid(cfg: Config, scope: str, stack: Any, sc: Scorer) -> HybridCfg:
    best = cfg.hybrid.model_copy()
    best_res = sc.score(f"hybrid|{best.w_bm25}", stack.hybrid(best))
    rows = [{"w_bm25": best.w_bm25, "ndcg@10": mean(best_res)}]
    for w in HYBRID_GRID:
        if any(r["w_bm25"] == w for r in rows):
            continue
        cand = best.model_copy(update={"w_bm25": w})
        res = sc.score(f"hybrid|{w}", stack.hybrid(cand))
        rows.append({"w_bm25": w, "ndcg@10": mean(res)})
        if mean(res) > mean(best_res) + 1e-9:
            best, best_res = cand, res
    for r in rows:
        r["chosen"] = r["w_bm25"] == best.w_bm25
    save_section(
        scope,
        "hybrid",
        {
            "selected_on": "validation",
            "n_val_queries": len(sc.queries),
            "params": best.model_dump(),
            "val_ndcg@10": best_res["summary"]["ndcg@10"],
            "search_log": sorted(rows, key=lambda r: r["w_bm25"]),
        },
    )
    return best


def tune_graph(cfg: Config, scope: str, stack: Any, sc: Scorer, h: HybridCfg) -> GraphCfg:
    best = hybrid_start(cfg.graph, h)
    best_res = sc.score(f"graph|{best.model_dump_json()}", stack.graphrag(best))
    rows: list[dict[str, Any]] = [
        {"param": "start (= hybrid)", "value": None, "ndcg@10": mean(best_res), "chosen": True}
    ]
    for name, options, needs in GRAPH_GRID:
        for upd in options:
            if needs and not getattr(best, needs):
                continue
            if all(getattr(best, k) == v for k, v in upd.items()):
                continue
            cand = best.model_copy(update=upd)
            res = sc.score(f"graph|{cand.model_dump_json()}", stack.graphrag(cand))
            gain = mean(res) - mean(best_res)
            chosen = gain > 1e-9
            value = upd.get(name, next(iter(upd.values())))
            rows.append(
                {
                    "param": name,
                    "value": value,
                    "ndcg@10": mean(res),
                    "gain": gain,
                    "chosen": chosen,
                }
            )
            if chosen:
                best, best_res = cand, res
    save_section(
        scope,
        "graph",
        {
            "selected_on": "validation",
            "n_val_queries": len(sc.queries),
            "params": best.model_dump(),
            "val_ndcg@10": best_res["summary"]["ndcg@10"],
            "search_log": rows,
        },
    )
    return best


def tune_rerank(cfg: Config, scope: str, stack: Any, sc: Scorer) -> RerankCfg:
    """Score each candidate cross-encoder once per query (top 50), then evaluate depth and
    blend weight offline. The best setting is stored; it is enabled for serving and for the
    agent only if it beats the plain hybrid on validation."""
    base = stack.hybrid()
    base_res = sc.score("rerank|none", base)
    pools: dict[str, Any] = {}
    for q in sc.queries:
        pools[q.qid] = base.search(q.text, 100, q.exclusion())
    rows: list[dict[str, Any]] = []
    best: RerankCfg = cfg.rerank.model_copy(update={"enabled": False})
    best_score = -1.0
    max_depth = max(RERANK_DEPTHS)
    for model in cfg.rerank.candidates:
        scorer = stack.scorer(model)
        ce: dict[str, list[float]] = {}
        for q in sc.queries:
            top = pools[q.qid][:max_depth]
            ce[q.qid] = scorer(q.text, [stack.texts.get(h.pub_id, "") for h in top])
        for depth in RERANK_DEPTHS:
            for alpha in RERANK_ALPHAS:
                ranked = {}
                for q in sc.queries:
                    hits = blend(pools[q.qid], ce[q.qid][:depth], depth, alpha)
                    ranked[q.qid] = runner.collapse_to_documents(
                        [h.pub_id for h in hits], sc.key_of
                    )[: runner.K_MAX]
                label = f"rerank|{model}|{depth}|{alpha}"
                res = sc.score_ranked(label, ranked, [])
                rows.append({"model": model, "depth": depth, "alpha": alpha, "ndcg@10": mean(res)})
                if mean(res) > best_score + 1e-9:
                    best_score = mean(res)
                    best = RerankCfg(
                        enabled=mean(res) > mean(base_res) + 1e-9,
                        model=model,
                        candidates=cfg.rerank.candidates,
                        depth=depth,
                        alpha=alpha,
                    )
    save_section(
        scope,
        "rerank",
        {
            "selected_on": "validation",
            "n_val_queries": len(sc.queries),
            "params": best.model_dump(),
            "hybrid_val_ndcg@10": mean(base_res),
            "val_ndcg@10": best_score,
            "search_log": rows,
        },
    )
    return best


def tune(cfg: Config, scope: str, stack: Any) -> dict[str, Any]:
    _, sc = validation(cfg, scope, stack)
    h = tune_hybrid(cfg, scope, stack, sc)
    g = tune_graph(cfg, scope, stack, sc, h)
    r = tune_rerank(cfg, scope, stack, sc)
    return {"hybrid": h.model_dump(), "graph": g.model_dump(), "rerank": r.model_dump()}


__all__ = ["Scorer", "hybrid_start", "load_tuned_hybrid", "tune"]
