"""BM25 lexical baseline (rank_bm25), persisted with pickle."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
        "said",
        "wherein",
        "comprising",
        "comprises",
        "according",
        "claim",
        "claims",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class Bm25Index:
    def __init__(self, ids: list[str], docs: list[str]):
        self.ids = ids
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        top = top[np.lexsort((np.array(self.ids, dtype=object)[top], -s[top]))]
        return [(self.ids[i], float(s[i])) for i in top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> Bm25Index:
        with path.open("rb") as fh:
            obj = pickle.load(fh)  # noqa: S301 (our own file)
        assert isinstance(obj, Bm25Index)
        return obj
