"""Graph-aware re-ranking: a documented weighted sum of similarity and graph features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from patentrag.graph.schema import cpc_ancestor


@dataclass(frozen=True)
class Weights:
    sim: float = 1.0
    cpc: float = 0.2
    cite: float = 0.2
    family: float = 0.0
    applicant: float = 0.1


def cpc_overlap(candidate_cpcs: Sequence[str], votes: Mapping[str, float], level: str) -> float:
    """Share of the query's CPC vote mass covered by the candidate's ancestors at `level`."""
    total = sum(votes.values())
    if total <= 0:
        return 0.0
    mine = {cpc_ancestor(s, level) for s in candidate_cpcs}
    return sum(w for node, w in votes.items() if node in mine) / total


def citation_proximity(hop: int | None) -> float:
    """1 for a direct citation neighbour of a seed, 1/2 at two hops, and so on; 0 if unlinked."""
    return 0.0 if not hop else 1.0 / hop


def final_score(
    sim: float,
    cpc: float,
    cite: float,
    family: bool,
    applicant: bool,
    w: Weights,
) -> float:
    """Weighted sum: sim, CPC overlap, citation proximity, family flag, applicant flag."""
    return (
        w.sim * sim
        + w.cpc * cpc
        + w.cite * cite
        + w.family * float(family)
        + w.applicant * float(applicant)
    )
