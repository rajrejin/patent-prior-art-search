"""Hybrid score fusion on hand-written inputs with hand-computed answers."""

from __future__ import annotations

import numpy as np
import pytest

from patentrag.retrieval.base import NO_EXCLUSION, Exclusion
from patentrag.retrieval.hybrid import HybridRetriever, LexicalVectorScores


class FakeMatrix:
    pub_ids = ["a", "b", "c"]
    pub_index = {"a": 0, "b": 1, "c": 2}
    texts = ["chunk a", "chunk b", "chunk c"]

    def __init__(self, best: list[float]):
        self.best = np.array(best, dtype=np.float32)

    def patent_sims(self, qvec, sections=None):
        return self.best, np.array([0, 1, 2])


class FakeBm25:
    ids = ["a", "b", "c"]

    def __init__(self, scores: list[float]):
        self._s = np.array(scores, dtype=np.float64)

    def scores(self, query: str) -> np.ndarray:
        return self._s


class FakeVector:
    sections = None

    def embed_query(self, query: str) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


def test_pool_is_vector_top_then_new_bm25_top():
    m = FakeMatrix([0.9, 0.5, 0.7])
    sc = LexicalVectorScores.compute(m, m.best, FakeBm25([0.0, 3.0, 1.0]), "q", NO_EXCLUSION, 2, 1)
    assert sc.pool == ["a", "c", "b"]
    # normalised on the pool: sim range 0.5..0.9, BM25 range 0..3
    assert sc.sim_n("c") == pytest.approx(0.5)
    assert sc.bm25_n("c") == pytest.approx(1 / 3)
    assert sc.fused("a", 1.0, 1.0) == pytest.approx(1.0)
    assert sc.fused("b", 1.0, 2.0) == pytest.approx(2.0)


def test_pool_respects_exclusion_and_skips_zero_bm25():
    m = FakeMatrix([0.9, 0.5, 0.7])
    excl = Exclusion(exclude=frozenset({"b"}))
    sc = LexicalVectorScores.compute(m, m.best, FakeBm25([0.0, 3.0, 1.0]), "q", excl, 2, 2)
    assert sc.pool == ["a", "c"]
    sc0 = LexicalVectorScores.compute(m, m.best, FakeBm25([0.0, 0.0, 0.0]), "q", NO_EXCLUSION, 1, 3)
    assert sc0.pool == ["a"]


def test_hybrid_weight_moves_ranking_towards_bm25():
    m = FakeMatrix([0.9, 0.5, 0.7])
    bm = FakeBm25([0.0, 3.0, 1.0])
    vec_heavy = HybridRetriever(FakeVector(), m, bm, w_sim=1.0, w_bm25=0.25, pool_n=3)  # type: ignore[arg-type]
    bm_heavy = HybridRetriever(FakeVector(), m, bm, w_sim=1.0, w_bm25=4.0, pool_n=3)  # type: ignore[arg-type]
    # a: 1 + 0, b: 0 + w, c: 0.5 + w/3
    assert [h.pub_id for h in vec_heavy.search("q", 3)] == ["a", "c", "b"]
    assert [h.pub_id for h in bm_heavy.search("q", 3)] == ["b", "c", "a"]
    hit = bm_heavy.search("q", 1)[0]
    assert hit.extra["best_chunk"] == "chunk b"
    assert hit.provenance == ["hybrid"]
