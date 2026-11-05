"""Graph model: CPC hierarchy helpers, expected counts and Neo4j schema statements.

Nodes: Patent, CPC(level), Applicant, Family.
Edges: CITES, CLASSIFIED_AS, CHILD_OF (CPC), FILED_BY, IN_FAMILY.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from patentrag.models import Patent

LEVELS = ["section", "class", "subclass", "group", "subgroup"]
_SYMBOL = re.compile(r"^([A-HY])(\d{2})([A-Z])(\d+)/(\d+)$")

CONSTRAINTS = [
    "CREATE CONSTRAINT patent_id IF NOT EXISTS FOR (p:Patent) REQUIRE p.pub_id IS UNIQUE",
    "CREATE CONSTRAINT cpc_id IF NOT EXISTS FOR (c:CPC) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT applicant_name IF NOT EXISTS FOR (a:Applicant) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT family_id IF NOT EXISTS FOR (f:Family) REQUIRE f.family_id IS UNIQUE",
]
INDEXES = [
    "CREATE INDEX patent_doc_key IF NOT EXISTS FOR (p:Patent) ON (p.doc_key)",
    "CREATE INDEX cpc_level IF NOT EXISTS FOR (c:CPC) ON (c.level)",
]


def cpc_chain(symbol: str) -> list[tuple[str, str]]:
    """(node id, level) from section down to the full symbol, or [] for malformed symbols.

    "H01M10/0525" -> H (section), H01 (class), H01M (subclass), H01M10 (group), H01M10/0525.
    The five-level tree is an approximation: the true CPC subgroup nesting is by indentation,
    which OPS does not return.
    """
    m = _SYMBOL.match(symbol)
    if not m:
        return []
    sec, cls, sub, grp, sg = m.groups()
    return [
        (sec, "section"),
        (f"{sec}{cls}", "class"),
        (f"{sec}{cls}{sub}", "subclass"),
        (f"{sec}{cls}{sub}{grp}", "group"),
        (f"{sec}{cls}{sub}{grp}/{sg}", "subgroup"),
    ]


def cpc_ancestor(symbol: str, level: str) -> str | None:
    """The node id at `level` for a full symbol (None if malformed)."""
    for node_id, lvl in cpc_chain(symbol):
        if lvl == level:
            return node_id
    return None


def normalise_applicant(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().upper()


@dataclass
class GraphPlan:
    """Everything the loader writes, derived purely from the corpus."""

    patents: list[dict[str, object]] = field(default_factory=list)
    cpc_nodes: dict[str, str] = field(default_factory=dict)  # id -> level
    child_of: set[tuple[str, str]] = field(default_factory=set)
    classified: set[tuple[str, str]] = field(default_factory=set)
    applicants: set[str] = field(default_factory=set)
    filed_by: set[tuple[str, str]] = field(default_factory=set)
    families: set[str] = field(default_factory=set)
    in_family: set[tuple[str, str]] = field(default_factory=set)
    cites: set[tuple[str, str]] = field(default_factory=set)
    dropped_citations: int = 0

    def counts(self) -> dict[str, int]:
        return {
            "Patent": len(self.patents),
            "CPC": len(self.cpc_nodes),
            "Applicant": len(self.applicants),
            "Family": len(self.families),
            "CITES": len(self.cites),
            "CLASSIFIED_AS": len(self.classified),
            "CHILD_OF": len(self.child_of),
            "FILED_BY": len(self.filed_by),
            "IN_FAMILY": len(self.in_family),
        }


def plan_graph(patents: list[Patent]) -> GraphPlan:
    plan = GraphPlan()
    by_key: dict[str, list[str]] = defaultdict(list)
    for p in patents:
        by_key[p.pub_id.doc_key].append(str(p.pub_id))
    for p in patents:
        pid = str(p.pub_id)
        plan.patents.append(
            {
                "pub_id": pid,
                "doc_key": p.pub_id.doc_key,
                "title": p.title,
                "pub_date": p.pub_date.isoformat() if p.pub_date else "",
                "year": p.pub_date.year if p.pub_date else 0,
                "family_id": p.family_id or "",
            }
        )
        for sym in p.cpc:
            chain = cpc_chain(sym)
            if not chain:
                continue
            for node_id, level in chain:
                plan.cpc_nodes[node_id] = level
            for (child, _), (parent, _) in zip(chain[1:], chain[:-1], strict=True):
                plan.child_of.add((child, parent))
            plan.classified.add((pid, chain[-1][0]))
        for a in p.applicants:
            name = normalise_applicant(a)
            if name:
                plan.applicants.add(name)
                plan.filed_by.add((pid, name))
        if p.family_id:
            plan.families.add(p.family_id)
            plan.in_family.add((pid, p.family_id))
        for c in p.cites:
            targets = by_key.get(c.doc_key)
            if not targets or c.doc_key == p.pub_id.doc_key:
                plan.dropped_citations += 1
                continue
            for t in targets:
                plan.cites.add((pid, t))
    return plan
