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
    """The `n` best-scoring ids (ties broken by id), skipping excluded and non-finite entries.

    `positive=True` also skips scores of zero or less, which for BM25 means "no shared word".
    """
    if n <= 0:
        return []
    order = np.lexsort((np.array(ids, dtype=object), -scores))
    top: list[str] = []
    for i in order:
        score = scores[i]
        if not np.isfinite(score) or (positive and score <= 0):
            continue
        if ids[i] in excl.exclude:
            continue
        top.append(ids[i])
        if len(top) >= n:
            break
    return top


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
        """`best[i]` is the highest chunk similarity of patent `matrix.pub_ids[i]` to the query."""
        sim = {
            pub_id: float(best[i])
            for i, pub_id in enumerate(matrix.pub_ids)
            if np.isfinite(best[i])
        }
        bm25_scores: dict[str, float] = {}
        # Candidate pool: the best patents by vector similarity, then the best by BM25.
        pool = _top_ids(matrix.pub_ids, best, vector_pool, excl)
        if bm25 is not None:
            raw = bm25.scores(query)
            bm25_scores = {pub_id: float(s) for pub_id, s in zip(bm25.ids, raw, strict=True)}
            already_in_pool = set(pool)
            bm25_top = _top_ids(bm25.ids, raw, bm25_pool, excl, positive=True)
            pool += [pub_id for pub_id in bm25_top if pub_id not in already_in_pool]
        scores = cls(sim, bm25_scores, pool)
        scores.normalise_on(pool)
        return scores

    def normalise_on(self, ids: list[str]) -> None:
        """Store the min and range of both score types over `ids` for min-max scaling.

        Vector similarities and BM25 scores live on different scales, so each is rescaled
        to 0..1 within the pool before they are added up.
        """
        if not ids:
            return
        sims = [self.sim.get(pub_id, 0.0) for pub_id in ids]
        bm25s = [self.bm25.get(pub_id, 0.0) for pub_id in ids]
        self.sim_lo, self.sim_span = min(sims), max(sims) - min(sims)
        self.bm_lo, self.bm_span = min(bm25s), max(bm25s) - min(bm25s)

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
        # For every patent: similarity of its best-matching chunk, and which chunk that was.
        query_vector = self.vector.embed_query(query)
        best_sim, best_chunk = self.matrix.patent_sims(query_vector, self.vector.sections)
        scores = LexicalVectorScores.compute(
            self.matrix, best_sim, self.bm25, query, excl, self.pool_n, self.pool_n
        )
        ranked = sorted(
            ((pub_id, scores.fused(pub_id, self.w_sim, self.w_bm25)) for pub_id in scores.pool),
            key=lambda item: (-item[1], item[0]),
        )
        hits = []
        for pub_id, score in ranked[:k]:
            row = self.matrix.pub_index.get(pub_id)
            has_chunk = row is not None and best_chunk[row] >= 0
            chunk_text = self.matrix.texts[int(best_chunk[row])] if has_chunk else ""
            features = {"sim": scores.sim_n(pub_id), "bm25": scores.bm25_n(pub_id)}
            hits.append(
                Hit(pub_id, score, ["hybrid"], {"features": features, "best_chunk": chunk_text})
            )
        return hits
