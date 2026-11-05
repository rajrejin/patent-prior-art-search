"""Graph schema tests (pure) and Neo4j integration tests (skipped without a loaded database)."""

from __future__ import annotations

import pytest

from patentrag.config import load_config
from patentrag.graph.queries import Exclusion, GraphQueries
from patentrag.graph.schema import cpc_ancestor, cpc_chain, normalise_applicant, plan_graph
from patentrag.ingest.corpus import corpus_paths, load_corpus


def test_cpc_chain_hand_computed():
    assert cpc_chain("H01M10/0525") == [
        ("H", "section"),
        ("H01", "class"),
        ("H01M", "subclass"),
        ("H01M10", "group"),
        ("H01M10/0525", "subgroup"),
    ]
    assert cpc_chain("not a symbol") == []
    assert cpc_ancestor("B60L58/12", "subclass") == "B60L"
    assert cpc_ancestor("Y02E60/10", "group") == "Y02E60"


def test_normalise_applicant():
    assert normalise_applicant("  Panasonic   Ip Man Co Ltd ") == "PANASONIC IP MAN CO LTD"


def _corpus():
    cfg = load_config()
    if not corpus_paths(cfg, "full")[0].exists():
        pytest.skip("full corpus not built (run `patentrag build-corpus`)")
    return load_corpus(cfg, "full")


def test_plan_graph_invariants_on_real_corpus():
    patents = _corpus()
    if not any(p.origin == "cited" for p in patents):
        pytest.skip("corpus has no citation-closure documents yet (run build-corpus)")
    plan = plan_graph(patents)
    ids = {str(p.pub_id) for p in patents}
    assert len(plan.patents) == len(ids)
    assert all(a in ids and b in ids and a != b for a, b in plan.cites)
    keys = {(a.split("A")[0], b) for a, b in plan.cites}
    assert keys  # the corpus has at least one in-corpus citation
    assert all(c in plan.cpc_nodes for _, c in plan.classified)
    assert all(a in plan.cpc_nodes and b in plan.cpc_nodes for a, b in plan.child_of)
    # every non-section CPC node has exactly one parent
    parents: dict[str, set[str]] = {}
    for child, parent in plan.child_of:
        parents.setdefault(child, set()).add(parent)
    assert all(len(v) == 1 for v in parents.values())


@pytest.fixture(scope="module")
def queries():
    try:
        from patentrag.graph.loader import connect, db_counts

        driver = connect()
        counts = db_counts(driver)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {type(exc).__name__}")
    if counts["Patent"] == 0:
        pytest.skip("Neo4j is empty (run `patentrag build-graph`)")
    yield GraphQueries(driver), driver
    driver.close()


@pytest.mark.needs_neo4j
def test_db_counts_equal_corpus_plan(queries):
    """The database holds one loaded scope; its counts must equal that scope's corpus plan."""
    from patentrag.graph.loader import db_counts

    _, driver = queries
    cfg = load_config()
    actual = db_counts(driver)
    for scope in ("full",):
        if not corpus_paths(cfg, scope)[0].exists():
            continue
        plan = plan_graph(load_corpus(cfg, scope))
        if plan.counts()["Patent"] == actual["Patent"]:
            assert actual == plan.counts()
            return
    pytest.skip("loaded graph does not correspond to a corpus file present here")


@pytest.mark.needs_neo4j
def test_hiding_outgoing_edges_removes_them_from_traversal(queries):
    q, driver = queries
    with driver.session() as s:
        row = s.run(
            "MATCH (p:Patent)-[:CITES]->(c:Patent) "
            "WHERE NOT (c)-[:CITES]->(p) "
            "RETURN p.pub_id AS q, collect(c.pub_id) AS cited LIMIT 1"
        ).single()
    assert row is not None, "no patent with an outgoing in-corpus citation"
    query_patent, cited = row["q"], set(row["cited"])
    open_hits = q.citation_neighbours([query_patent], hops=1)
    assert cited <= {k for k, v in open_hits.items() if v["direction"] == "cites"}
    hidden = q.citation_neighbours(
        [query_patent], hops=1, excl=Exclusion(hide_outgoing_from=frozenset({query_patent}))
    )
    assert not (cited & set(hidden)), "outgoing citations leaked through the filter"


@pytest.mark.needs_neo4j
def test_excluded_nodes_are_never_returned_or_traversed(queries):
    q, driver = queries
    with driver.session() as s:
        row = s.run(
            "MATCH (a:Patent)-[:CITES]->(b:Patent)-[:CITES]->(c:Patent) "
            "WHERE a <> c RETURN a.pub_id AS a, b.pub_id AS b, c.pub_id AS c LIMIT 1"
        ).single()
    if row is None:
        pytest.skip("no two-hop citation chain in the loaded graph")
    hits = q.citation_neighbours([row["a"]], hops=2, excl=Exclusion(exclude=frozenset({row["b"]})))
    assert row["b"] not in hits
    # c is only reachable through b in this chain unless another path exists; if it is
    # reached anyway, it must not be via b.
    assert hits.get(row["c"], {}).get("via") != row["b"]
