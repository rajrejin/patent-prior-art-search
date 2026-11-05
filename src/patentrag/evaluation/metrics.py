"""Ranking metrics (binary relevance), bootstrap intervals and paired bootstrap."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def query_metrics(ranked: Sequence[str], relevant: set[str], ks: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {
        "mrr": reciprocal_rank(ranked, relevant),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
    }
    for k in ks:
        out[f"p@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"r@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"hit@{k}"] = hit_rate_at_k(ranked, relevant, k)
    return out


def bootstrap_ci(
    values: Sequence[float], resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(mean, low, high) percentile bootstrap over queries."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(resamples, len(v)))
    means = v[idx].mean(axis=1)
    return (
        float(v.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], resamples: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Mean per-query difference a - b with a 95% interval and a two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return {"diff": 0.0, "low": 0.0, "high": 0.0, "p": 1.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    # p-value: share of resampled means on the other side of zero, doubled, floored at 1/resamples.
    tail = min((means <= 0).mean(), (means >= 0).mean())
    p = min(1.0, max(2 * tail, 1.0 / resamples))
    return {
        "diff": float(d.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "p": float(p),
        "n": int(len(d)),
    }
