"""Read queries used by GraphRAG. All of them take a query-time exclusion context.

Leakage controls are applied here, not by deleting data:
  * `exclude`: pub_ids that may never appear in a result or be traversed through (the query
    patent itself and its family members);
  * `hide_outgoing_from`: pub_ids whose outgoing CITES edges are invisible for this query.
"""

from __future__ import annotations

from typing import Any

from neo4j import Driver

from patentrag.retrieval.base import NO_EXCLUSION, Exclusion

# One hop in either direction over CITES. An edge is dropped when its source is in hide_src.
CITATION_HOP = """
MATCH (a:Patent)-[r:CITES]-(b:Patent)
WHERE a.pub_id IN $frontier
  AND NOT b.pub_id IN $exclude
  AND NOT startNode(r).pub_id IN $hide_src
RETURN DISTINCT a.pub_id AS src, b.pub_id AS dst,
       CASE WHEN startNode(r) = a THEN 'cites' ELSE 'cited_by' END AS direction
"""

CPC_SUBTREE_PATENTS = """
MATCH (c:CPC) WHERE c.id IN $cpc_ids
OPTIONAL MATCH (c)<-[:CHILD_OF*0..%d]-(d:CPC)
WITH DISTINCT c, d
MATCH (p:Patent)-[:CLASSIFIED_AS]->(d)
WHERE NOT p.pub_id IN $exclude
RETURN p.pub_id AS pub_id, c.id AS via, d.id AS cpc
"""

FAMILY_MEMBERS = """
MATCH (a:Patent)-[:IN_FAMILY]->(f:Family)<-[:IN_FAMILY]-(b:Patent)
WHERE a.pub_id IN $seeds AND a <> b AND NOT b.pub_id IN $exclude
RETURN DISTINCT a.pub_id AS src, b.pub_id AS dst
"""

PATENT_CPCS = """
MATCH (p:Patent)-[:CLASSIFIED_AS]->(c:CPC)
WHERE p.pub_id IN $ids
RETURN p.pub_id AS pub_id, collect(c.id) AS cpcs
"""

PATENT_APPLICANTS = """
MATCH (p:Patent)-[:FILED_BY]->(a:Applicant)
WHERE p.pub_id IN $ids
RETURN p.pub_id AS pub_id, collect(a.name) AS applicants
"""


class GraphQueries:
    def __init__(self, driver: Driver):
        self.driver = driver

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(default_access_mode="READ") as s:
            return [dict(r) for r in s.run(query, **params)]

    def citation_hop(
        self, frontier: list[str], excl: Exclusion = NO_EXCLUSION
    ) -> list[dict[str, Any]]:
        return self._run(CITATION_HOP, frontier=list(frontier), **excl.params())

    def citation_neighbours(
        self, seeds: list[str], hops: int, excl: Exclusion = NO_EXCLUSION
    ) -> dict[str, dict[str, Any]]:
        """pub_id -> {hop, via (the seed-side neighbour), direction} for nodes within `hops`."""
        found: dict[str, dict[str, Any]] = {}
        frontier = list(seeds)
        seen = set(seeds)
        for hop in range(1, hops + 1):
            nxt: list[str] = []
            for row in self.citation_hop(frontier, excl):
                dst = row["dst"]
                if dst in seen:
                    continue
                seen.add(dst)
                found[dst] = {"hop": hop, "via": row["src"], "direction": row["direction"]}
                nxt.append(dst)
            frontier = nxt
            if not frontier:
                break
        return found

    def cpc_subtree_patents(
        self, cpc_ids: list[str], depth: int, excl: Exclusion = NO_EXCLUSION
    ) -> list[dict[str, Any]]:
        """Patents classified at or below each CPC node (depth levels down)."""
        q = CPC_SUBTREE_PATENTS % max(depth, 0)
        return self._run(q, cpc_ids=list(cpc_ids), exclude=sorted(excl.exclude))

    def family_members(
        self, seeds: list[str], excl: Exclusion = NO_EXCLUSION
    ) -> list[dict[str, Any]]:
        return self._run(FAMILY_MEMBERS, seeds=list(seeds), exclude=sorted(excl.exclude))

    def patent_cpcs(self, ids: list[str]) -> dict[str, list[str]]:
        return {r["pub_id"]: r["cpcs"] for r in self._run(PATENT_CPCS, ids=list(ids))}

    def patent_applicants(self, ids: list[str]) -> dict[str, list[str]]:
        return {r["pub_id"]: r["applicants"] for r in self._run(PATENT_APPLICANTS, ids=list(ids))}
