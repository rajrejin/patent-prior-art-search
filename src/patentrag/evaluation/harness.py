"""End-to-end evaluation of retrieval methods on a corpus scope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from patentrag.config import ROOT, Config
from patentrag.evaluation import runner
from patentrag.evaluation.testset import (
    CorpusIndex,
    Query,
    build_indexes,
    eligible_queries,
    load_testset,
    save_testset,
    stratified_split,
)
from patentrag.ingest.corpus import corpus_paths, load_corpus
from patentrag.manifest import base_manifest
from patentrag.models import Patent
from patentrag.retrieval.base import Retriever

RESULTS = ROOT / "results"


def testset_path(scope: str) -> Path:
    return RESULTS / f"testset_manifest_{scope}.json"


def ensure_testset(
    cfg: Config, scope: str, patents: list[Patent]
) -> tuple[list[Query], list[Query], dict[str, Any]]:
    path = testset_path(scope)
    if path.exists():
        return load_testset(path)
    elig = eligible_queries(patents, cfg.evaluation.min_relevant)
    val, test = stratified_split(elig, cfg.evaluation.n_test, cfg.evaluation.n_val, cfg.seed)
    corpus_hash = json.loads(corpus_paths(cfg, scope)[1].read_text(encoding="utf-8"))["corpus_hash"]
    meta = base_manifest(
        cfg,
        corpus_hash,
        scope=scope,
        n_eligible=len(elig),
        n_val=len(val),
        n_test=len(test),
        requested={"n_val": cfg.evaluation.n_val, "n_test": cfg.evaluation.n_test},
        min_relevant=cfg.evaluation.min_relevant,
    )
    save_testset(path, val, test, meta)
    return load_testset(path)


def subsample(queries: list[Query], seed: int = 0) -> list[Query]:
    """Seeded random subset when PATENTRAG_MAX_QUERIES is set (for slow machines); else all."""
    import os
    import random

    n = int(os.environ.get("PATENTRAG_MAX_QUERIES", "0") or 0)
    if n <= 0 or n >= len(queries):
        return queries
    pool = sorted(queries, key=lambda q: q.qid)
    return sorted(random.Random(seed).sample(pool, n), key=lambda q: q.qid)


def evaluate_methods(
    cfg: Config,
    scope: str,
    retrievers: dict[str, Retriever],
    split: str,
    out_name: str,
    hide_outgoing: dict[str, bool] | None = None,
    extra_runs: dict[str, runner.RunResult] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score retrievers (and precomputed runs such as the agent's) on one split, with
    bootstrap intervals and paired comparisons of every pair of methods."""
    patents = load_corpus(cfg, scope)
    idx: CorpusIndex = build_indexes(patents)
    val, test, meta = ensure_testset(cfg, scope, patents)
    queries = val if split == "val" else test
    ks = cfg.evaluation.ks
    runs: dict[str, runner.RunResult] = {}
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for name, r in retrievers.items():
        runs[name] = runner.run_method(
            r, queries, idx.key_of, hide_outgoing=(hide_outgoing or {}).get(name, True), label=name
        )
        scores[name] = runner.score_run(runs[name], queries, ks)
    for name, run in (extra_runs or {}).items():
        runs[name] = run
        scores[name] = runner.score_run(run, queries, ks)
    n_res = cfg.evaluation.bootstrap_resamples
    result: dict[str, Any] = {
        "scope": scope,
        "split": split,
        "n_queries": len(queries),
        "testset": meta,
        "summary": {n: runner.summarise(s, n_res, cfg.seed) for n, s in scores.items()},
        "latency_ms": {n: runner.latency_summary(r.latency_ms) for n, r in runs.items()},
        "breakdown": {n: runner.breakdown(s, queries) for n, s in scores.items()},
        "per_query": scores,
        **(extra or {}),
    }
    names = list(runs)
    metrics = ["ndcg@10", "mrr", *[f"r@{k}" for k in ks], *[f"p@{k}" for k in ks]]
    result["paired"] = {
        f"{a}-{b}": runner.compare(scores[a], scores[b], metrics, n_res, cfg.seed)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{out_name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def summary_table(result: dict[str, Any], ks: list[int]) -> str:
    cols = ["ndcg@10", "mrr", *[f"r@{k}" for k in ks], *[f"p@{k}" for k in ks]]
    lines = ["| method | " + " | ".join(cols) + " | p50 ms |", "|" + "---|" * (len(cols) + 2)]
    for name, s in result["summary"].items():
        cells = [f"{s[c]['mean']:.3f} [{s[c]['low']:.3f}, {s[c]['high']:.3f}]" for c in cols]
        lines.append(
            f"| {name} | " + " | ".join(cells) + f" | {result['latency_ms'][name]['p50']:.1f} |"
        )
    return "\n".join(lines)
