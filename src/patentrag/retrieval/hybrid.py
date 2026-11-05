"""Hybrid retrieval: BM25 and exact vector similarity fused by a weighted sum.

Both scores are min-max normalised with constants taken from the *pool* (the union of the
vector top-`pool_n` and the BM25 top-`pool_n`), so documents outside the pool keep comparable,
possibly negative, values. GraphRAG reuses `LexicalVectorScores` with the same normalisation,
which makes GraphRAG with all graph weights at zero rank exactly like this retriever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from patentrag.index.bm25 import Bm25Index
from patentrag.index.vector_store import ChunkMatrix
from patentrag.retrieval.base import NO_EXCLUSION, Exclusion, Hit
from patentrag.retrieval.vector import VectorRetriever


def _top_ids(
    ids: list[str], scores: np.ndarray, n: int, excl: Exclusion, positive: bool = False
) -> list[str]:
    if n <= 0:
        return []
    order = np.lexsort((np.array(ids, dtype=object), -scores))
    out: list[str] = []
    for i in order:
        s = scores[i]
        if not np.isfinite(s) or (positive and s <= 0):
            continue
        if ids[i] in excl.exclude:
            continue
        out.append(ids[i])
        if len(out) >= n:
            break
    return out


@dataclass
class LexicalVectorScores:
    """Raw vector and BM25 scores for every patent of one query, plus pool normalisation."""

    sim: dict[str, float]
    bm25: dict[str, float]
    pool: list[str]
    sim_lo: float = 0.0
    sim_span: float = 0.0
    bm_lo: float = 0.0
    bm_span: float = 0.0

    @classmethod
    def compute(
        cls,
        matrix: ChunkMatrix,
        best: np.ndarray,
        bm25: Bm25Index | None,
        query: str,
        excl: Exclusion,
        vector_pool: int,
        bm25_pool: int,
    ) -> LexicalVectorScores:
        sim = {p: float(best[i]) for i, p in enumerate(matrix.pub_ids) if np.isfinite(best[i])}
        bm: dict[str, float] = {}
        pool = _top_ids(matrix.pub_ids, best, vector_pool, excl)
        if bm25 is not None:
            raw = bm25.scores(query)
            bm = {p: float(s) for p, s in zip(bm25.ids, raw, strict=True)}
            seen = set(pool)
            pool += [p for p in _top_ids(bm25.ids, raw, bm25_pool, excl, True) if p not in seen]
        out = cls(sim, bm, pool)
        out.normalise_on(pool)
        return out

    def normalise_on(self, ids: list[str]) -> None:
        if not ids:
            return
        s = [self.sim.get(p, 0.0) for p in ids]
        b = [self.bm25.get(p, 0.0) for p in ids]
        self.sim_lo, self.sim_span = min(s), max(s) - min(s)
        self.bm_lo, self.bm_span = min(b), max(b) - min(b)

    def sim_n(self, pid: str) -> float:
        if self.sim_span <= 0:
            return 0.0
        return (self.sim.get(pid, self.sim_lo) - self.sim_lo) / self.sim_span

    def bm25_n(self, pid: str) -> float:
        if self.bm_span <= 0:
            return 0.0
        return (self.bm25.get(pid, self.bm_lo) - self.bm_lo) / self.bm_span

    def fused(self, pid: str, w_sim: float, w_bm25: float) -> float:
        return w_sim * self.sim_n(pid) + w_bm25 * self.bm25_n(pid)


class HybridRetriever:
    """Weighted sum of normalised exact vector similarity and BM25 score (no graph)."""

    name = "hybrid"

    def __init__(
        self,
        vector: VectorRetriever,
        matrix: ChunkMatrix,
        bm25: Bm25Index,
        w_sim: float = 1.0,
        w_bm25: float = 1.0,
        pool_n: int = 100,
    ):
        self.vector = vector
        self.matrix = matrix
        self.bm25 = bm25
        self.w_sim = w_sim
        self.w_bm25 = w_bm25
        self.pool_n = pool_n

    def search(self, query: str, k: int, excl: Exclusion = NO_EXCLUSION) -> list[Hit]:
        best, arg = self.matrix.patent_sims(self.vector.embed_query(query), self.vector.sections)
        sc = LexicalVectorScores.compute(
            self.matrix, best, self.bm25, query, excl, self.pool_n, self.pool_n
        )
        ranked = sorted(
            ((p, sc.fused(p, self.w_sim, self.w_bm25)) for p in sc.pool),
            key=lambda t: (-t[1], t[0]),
        )
        hits = []
        for p, s in ranked[:k]:
            i = self.matrix.pub_index.get(p)
            chunk = self.matrix.texts[int(arg[i])] if i is not None and arg[i] >= 0 else ""
            feat = {"sim": sc.sim_n(p), "bm25": sc.bm25_n(p)}
            hits.append(Hit(p, s, ["hybrid"], {"features": feat, "best_chunk": chunk}))
        return hits
