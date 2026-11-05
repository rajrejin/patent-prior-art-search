"""Run retrievers over queries, score them, and summarise with confidence intervals."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from patentrag.evaluation.metrics import bootstrap_ci, paired_bootstrap, query_metrics
from patentrag.evaluation.testset import Query
from patentrag.retrieval.base import Retriever

K_MAX = 50
WARMUP = 3


@dataclass
class RunResult:
    method: str
    ranked: dict[str, list[str]] = field(default_factory=dict)  # qid -> doc_keys
    latency_ms: list[float] = field(default_factory=list)
    provenance: dict[str, dict[str, list[str]]] = field(default_factory=dict)


def collapse_to_documents(pub_ids: list[str], key_of: dict[str, str]) -> list[str]:
    """Map pub_ids to document keys, keeping the first occurrence of each."""
    seen: set[str] = set()
    out: list[str] = []
    for pid in pub_ids:
        k = key_of[pid]
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def run_method(
    retriever: Retriever,
    queries: list[Query],
    key_of: dict[str, str],
    *,
    hide_outgoing: bool = True,
    k: int = K_MAX,
    label: str | None = None,
) -> RunResult:
    res = RunResult(label or retriever.name)
    for i, q in enumerate(queries):
        excl = q.exclusion(hide_outgoing=hide_outgoing)
        t0 = time.perf_counter()
        hits = retriever.search(q.text, k * 2, excl)
        dt = (time.perf_counter() - t0) * 1000
        if i >= WARMUP:
            res.latency_ms.append(dt)
        res.ranked[q.qid] = collapse_to_documents([h.pub_id for h in hits], key_of)[:k]
        res.provenance[q.qid] = {h.pub_id: h.provenance for h in hits[:k]}
    return res


def score_run(run: RunResult, queries: list[Query], ks: list[int]) -> dict[str, dict[str, float]]:
    return {
        q.qid: query_metrics(run.ranked.get(q.qid, []), set(q.relevant_keys), ks) for q in queries
    }


def summarise(
    per_query: dict[str, dict[str, float]], resamples: int = 1000, seed: int = 0
) -> dict[str, dict[str, float]]:
    if not per_query:
        return {}
    names = sorted(next(iter(per_query.values())))
    out: dict[str, dict[str, float]] = {}
    for m in names:
        mean, lo, hi = bootstrap_ci([v[m] for v in per_query.values()], resamples, seed)
        out[m] = {"mean": mean, "low": lo, "high": hi}
    return out


def compare(
    a: dict[str, dict[str, float]],
    b: dict[str, dict[str, float]],
    metrics: list[str],
    resamples: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    qids = sorted(set(a) & set(b))
    return {
        m: paired_bootstrap([a[q][m] for q in qids], [b[q][m] for q in qids], resamples, seed)
        for m in metrics
    }


def latency_summary(ms: list[float]) -> dict[str, float]:
    if not ms:
        return {"p50": 0.0, "p95": 0.0, "n": 0}
    return {
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "n": len(ms),
    }


def breakdown(
    per_query: dict[str, dict[str, float]],
    queries: list[Query],
    metric: str = "ndcg@10",
) -> dict[str, Any]:
    """Mean metric by number of relevant documents and by publication year."""
    by_rel: dict[str, list[float]] = {}
    by_year: dict[str, list[float]] = {}
    for q in queries:
        v = per_query[q.qid][metric]
        n = len(q.relevant_keys)
        bucket = "3" if n == 3 else "4-5" if n <= 5 else "6-10" if n <= 10 else "11+"
        by_rel.setdefault(bucket, []).append(v)
        by_year.setdefault(str(q.year), []).append(v)
    f = lambda d: {  # noqa: E731
        k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(d.items())
    }
    return {"by_num_relevant": f(by_rel), "by_year": f(by_year)}
