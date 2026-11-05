"""Cross-encoder re-ranking of a base retriever's top results.

A cross-encoder reads the query and one document together and scores their relevance, which is
more accurate than comparing two separately computed embeddings but too slow for the whole
corpus. So the base retriever (hybrid search) supplies the top `depth` documents and only those
are re-scored. The final score blends the min-max normalised cross-encoder score with the
normalised base score: `alpha * ce + (1 - alpha) * base` (alpha 1 = cross-encoder only).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from patentrag.retrieval.base import NO_EXCLUSION, Exclusion, Hit, Retriever

Scorer = Callable[[str, Sequence[str]], list[float]]


def minmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]


def blend(hits: list[Hit], ce: Sequence[float], depth: int, alpha: float) -> list[Hit]:
    """Re-order the first `depth` hits by the blended score; later hits keep their order."""
    top, rest = hits[:depth], hits[depth:]
    fused = [
        alpha * c + (1 - alpha) * b
        for c, b in zip(minmax(list(ce)), minmax([h.score for h in top]), strict=True)
    ]
    order = sorted(range(len(top)), key=lambda i: (-fused[i], top[i].pub_id))
    out = []
    for i in order:
        h = top[i]
        extra = {**h.extra, "rerank": round(fused[i], 4)}
        out.append(Hit(h.pub_id, fused[i], [*h.provenance, "rerank"], extra))
    return out + rest


def cross_encoder_scorer(model_name: str, max_length: int = 512) -> Scorer:
    """Load a sentence-transformers cross-encoder (GPU when available)."""
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        model_name,
        max_length=max_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    def score(query: str, docs: Sequence[str]) -> list[float]:
        if not docs:
            return []
        out = model.predict([(query, d) for d in docs], batch_size=16, show_progress_bar=False)
        return [float(x) for x in out]

    return score


class RerankRetriever:
    """Base retriever plus cross-encoder re-ranking of its top `depth` results."""

    name = "rerank"

    def __init__(
        self,
        base: Retriever,
        texts: Mapping[str, str],
        scorer: Scorer,
        depth: int = 50,
        alpha: float = 1.0,
    ):
        self.base = base
        self.texts = texts
        self.scorer = scorer
        self.depth = depth
        self.alpha = alpha

    def search(self, query: str, k: int, excl: Exclusion = NO_EXCLUSION) -> list[Hit]:
        hits = self.base.search(query, max(k, self.depth), excl)
        top = hits[: self.depth]
        ce = self.scorer(query, [self.texts.get(h.pub_id, "") for h in top])
        return blend(hits, ce, self.depth, self.alpha)[:k]
