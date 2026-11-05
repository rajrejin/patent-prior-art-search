"""Retriever interface shared by BM25, vector RAG and GraphRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Exclusion:
    """Query-time leakage controls (Section 7.1), applied without deleting any data.

    exclude: pub_ids that may not appear in results or be traversed (query patent + family).
    hide_outgoing_from: pub_ids whose outgoing CITES edges are invisible for this query.
    final_only: leakage demonstration switch, see the field comment.
    """

    exclude: frozenset[str] = field(default_factory=frozenset)
    hide_outgoing_from: frozenset[str] = field(default_factory=frozenset)
    # True only for the labelled leakage demonstration: excluded ids may still act as seeds and
    # be traversed through, and are removed from the final ranking only.
    final_only: bool = False

    def params(self) -> dict[str, list[str]]:
        return {"exclude": sorted(self.exclude), "hide_src": sorted(self.hide_outgoing_from)}


NO_EXCLUSION = Exclusion()


@dataclass
class Hit:
    pub_id: str
    score: float
    provenance: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int, excl: Exclusion = NO_EXCLUSION) -> list[Hit]: ...


def top_k(scores: dict[str, float], k: int, excl: Exclusion) -> list[tuple[str, float]]:
    """Deterministic ranking: score descending, pub_id ascending on ties."""
    items = [(p, s) for p, s in scores.items() if p not in excl.exclude]
    items.sort(key=lambda t: (-t[1], t[0]))
    return items[:k]
