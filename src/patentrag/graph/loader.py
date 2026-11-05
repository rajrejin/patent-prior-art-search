"""Batched UNWIND/MERGE loading of the corpus into Neo4j. Idempotent."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from neo4j import Driver, GraphDatabase

from patentrag.config import Secrets, load_secrets
from patentrag.graph.schema import CONSTRAINTS, INDEXES, GraphPlan, plan_graph
from patentrag.models import Patent

log = logging.getLogger(__name__)
BATCH = 2000


def connect(secrets: Secrets | None = None) -> Driver:
    s = secrets or load_secrets()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


def _chunks(rows: list[Any], n: int = BATCH) -> Iterator[list[Any]]:
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


def _run_batches(driver: Driver, query: str, rows: Iterable[Any]) -> None:
    rows = list(rows)
    with driver.session() as session:
        for chunk in _chunks(rows):

            def work(tx: Any, c: list[Any] = chunk) -> None:
                tx.run(query, rows=c).consume()

            session.execute_write(work)


def create_schema(driver: Driver) -> None:
    with driver.session() as session:
        for stmt in [*CONSTRAINTS, *INDEXES]:
            session.run(stmt).consume()


def load_plan(driver: Driver, plan: GraphPlan) -> dict[str, int]:
    create_schema(driver)
    _run_batches(
        driver,
        "UNWIND $rows AS r MERGE (p:Patent {pub_id: r.pub_id}) "
        "SET p.doc_key=r.doc_key, p.title=r.title, p.pub_date=r.pub_date, "
        "p.year=r.year, p.family_id=r.family_id",
        plan.patents,
    )
    _run_batches(
        driver,
        "UNWIND $rows AS r MERGE (c:CPC {id: r.id}) SET c.level=r.level",
        [{"id": k, "level": v} for k, v in plan.cpc_nodes.items()],
    )
    _run_batches(
        driver,
        "UNWIND $rows AS r MATCH (a:CPC {id: r.child}), (b:CPC {id: r.parent}) "
        "MERGE (a)-[:CHILD_OF]->(b)",
        [{"child": a, "parent": b} for a, b in sorted(plan.child_of)],
    )
    _run_batches(
        driver,
        "UNWIND $rows AS r MATCH (p:Patent {pub_id: r.p}), (c:CPC {id: r.c}) "
        "MERGE (p)-[:CLASSIFIED_AS]->(c)",
        [{"p": a, "c": b} for a, b in sorted(plan.classified)],
    )
    _run_batches(
        driver,
        "UNWIND $rows AS r MERGE (:Applicant {name: r.name})",
        [{"name": n} for n in sorted(plan.applicants)],
    )
    _run_batches(
        driver,
        "UNWIND $rows AS r MATCH (p:Patent {pub_id: r.p}), (a:Applicant {name: r.a}) "
        "MERGE (p)-[:FILED_BY]->(a)",
        [{"p": a, "a": b} for a, b in sorted(plan.filed_by)],
    )
    _run_batches(
        driver,
        "UNWIND $rows AS r MERGE (:Family {family_id: r.f})",
        [{"f": f} for f in sorted(plan.families)],
    )
    _run_batches(
        driver,
        "UNWIND $rows AS r MATCH (p:Patent {pub_id: r.p}), (f:Family {family_id: r.f}) "
        "MERGE (p)-[:IN_FAMILY]->(f)",
        [{"p": a, "f": b} for a, b in sorted(plan.in_family)],
    )
    _run_batches(
        driver,
        "UNWIND $rows AS r MATCH (a:Patent {pub_id: r.a}), (b:Patent {pub_id: r.b}) "
        "MERGE (a)-[:CITES]->(b)",
        [{"a": a, "b": b} for a, b in sorted(plan.cites)],
    )
    return db_counts(driver)


def _count(session: Any, query: str) -> int:
    rec = session.run(query).single()
    return int(rec["c"]) if rec else 0


def db_counts(driver: Driver) -> dict[str, int]:
    out: dict[str, int] = {}
    with driver.session() as session:
        for label in ("Patent", "CPC", "Applicant", "Family"):
            out[label] = _count(session, f"MATCH (n:{label}) RETURN count(n) AS c")
        for rel in ("CITES", "CLASSIFIED_AS", "CHILD_OF", "FILED_BY", "IN_FAMILY"):
            out[rel] = _count(session, f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
    return out


def load_corpus_into_graph(driver: Driver, patents: list[Patent]) -> dict[str, Any]:
    plan = plan_graph(patents)
    log.info("loading %s", plan.counts())
    actual = load_plan(driver, plan)
    expected = plan.counts()
    return {
        "expected": expected,
        "actual": actual,
        "match": expected == actual,
        "dropped_citations": plan.dropped_citations,
    }
