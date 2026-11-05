"""Vector RAG retriever: chunk-level cosine search aggregated to patents."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from patentrag.index.embed import Embedder
from patentrag.index.vector_store import VectorStore
from patentrag.retrieval.base import NO_EXCLUSION, Exclusion, Hit


def aggregate_chunks(
    chunk_hits: list[dict[str, Any]], mode: str = "max", sections: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Patent score from chunk similarities.

    max: best chunk similarity. top3sum: sum of the three best chunk similarities.
    Returns pub_id -> {"score", "best_chunk", "best_section"}.
    """
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for h in chunk_hits:
        if sections is None or h["section"] in sections:
            per[h["pub_id"]].append(h)
    out: dict[str, dict[str, Any]] = {}
    for pid, hs in per.items():
        hs.sort(key=lambda h: -h["sim"])
        score = hs[0]["sim"] if mode == "max" else sum(h["sim"] for h in hs[:3])
        out[pid] = {
            "score": float(score),
            "best_chunk": hs[0]["text"],
            "best_section": hs[0]["section"],
        }
    return out


class VectorRetriever:
    name = "vector"

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        aggregation: str = "max",
        sections: set[str] | None = None,
        chunk_pool: int = 400,
    ):
        self.embedder = embedder
        self.store = store
        self.aggregation = aggregation
        self.sections = sections
        self.chunk_pool = chunk_pool

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed_queries([query])[0]

    def search_vector(self, qvec: np.ndarray, k: int, excl: Exclusion = NO_EXCLUSION) -> list[Hit]:
        pool = max(self.chunk_pool, 10 * (k + len(excl.exclude)))
        agg = aggregate_chunks(self.store.query(qvec, pool), self.aggregation, self.sections)
        ranked = sorted(
            ((p, v) for p, v in agg.items() if p not in excl.exclude),
            key=lambda t: (-t[1]["score"], t[0]),
        )
        return [
            Hit(
                p,
                v["score"],
                ["vector"],
                {"best_chunk": v["best_chunk"], "section": v["best_section"]},
            )
            for p, v in ranked[:k]
        ]

    def search(self, query: str, k: int, excl: Exclusion = NO_EXCLUSION) -> list[Hit]:
        return self.search_vector(self.embed_query(query), k, excl)
