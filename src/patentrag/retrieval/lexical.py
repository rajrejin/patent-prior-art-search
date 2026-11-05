"""BM25 retriever over title + abstract + claims."""

from __future__ import annotations

from patentrag.index.bm25 import Bm25Index
from patentrag.retrieval.base import NO_EXCLUSION, Exclusion, Hit


class Bm25Retriever:
    name = "bm25"

    def __init__(self, index: Bm25Index):
        self.index = index

    def search(self, query: str, k: int, excl: Exclusion = NO_EXCLUSION) -> list[Hit]:
        scores = self.index.scores(query)
        ranked = sorted(zip(self.index.ids, scores, strict=True), key=lambda t: (-t[1], t[0]))
        hits = [Hit(p, float(s), ["bm25"]) for p, s in ranked if p not in excl.exclude and s > 0]
        return hits[:k]
