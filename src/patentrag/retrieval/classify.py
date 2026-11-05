"""Query classification: vote over the CPC symbols of the most similar patents."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from patentrag.graph.schema import cpc_ancestor


def vote_cpc(
    seeds: Sequence[tuple[str, float]],
    patent_cpcs: Mapping[str, Sequence[str]],
    level: str,
    top_m: int,
) -> list[tuple[str, float]]:
    """Rank CPC nodes at `level` by similarity-weighted votes.

    `seeds` are (pub_id, similarity). Each patent votes once per distinct ancestor at `level`,
    with weight equal to its similarity. Ties break on the node id for determinism.
    """
    votes: dict[str, float] = defaultdict(float)
    for pid, sim in seeds:
        ancestors = {cpc_ancestor(sym, level) for sym in patent_cpcs.get(pid, ())}
        for node in ancestors:
            if node:
                votes[node] += sim
    ranked = sorted(votes.items(), key=lambda t: (-t[1], t[0]))
    return ranked[:top_m]
