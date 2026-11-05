"""System tests on the real full corpus, indexes and Neo4j graph.

Each test skips with a reason when its prerequisites (corpus file, built indexes, a loaded Neo4j)
are missing, as in CI. Only the language model is replaced, by a deterministic stand-in that
answers in the required JSON format; all patent data is real.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from patentrag.config import load_config
from patentrag.index.build import index_dirs
from patentrag.ingest.corpus import corpus_paths
from patentrag.llm.client import LLMResult
from patentrag.retrieval.base import Exclusion

SCOPE = "full"


def _have_data() -> None:
    cfg = load_config()
    chroma, bm25, _ = index_dirs(cfg, SCOPE)
    if not corpus_paths(cfg, SCOPE)[0].exists() or not chroma.exists() or not bm25.exists():
        pytest.skip("full corpus or indexes not built here")


def _graph_loaded() -> None:
    try:
        from patentrag.graph.loader import connect, db_counts

        d = connect()
        n = db_counts(d)["Patent"]
        d.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {type(exc).__name__}")
    from patentrag.ingest.corpus import load_corpus

    if n != len(load_corpus(load_config(), SCOPE)):
        pytest.skip("Neo4j does not hold the full corpus")


class RuleLLM:
    """Deterministic stand-in for the local LLM: valid JSON for whichever schema is requested.

    Planner: always picks the first offered tool (with the first feature or candidate id), so
    the run also exercises the read and expansion caps and the step limit. Features are the
    idea's own sentences; judge calls answer "yes". Chart evidence is ranked by the real
    embedding model, not by the LLM.
    """

    backend = "rule"
    model = "rule"

    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, prompt, *, system=None, json_schema=None, max_tokens=700) -> LLMResult:
        from patentrag.faithfulness.claims import split_sentences

        self.calls += 1
        props = (json_schema or {}).get("properties", {})
        if "verdict" in props:
            return LLMResult('{"verdict": "yes"}', 0.0)
        if "tool" in props:
            fids = [f for f in props["feature_id"]["enum"] if f]
            first = lambda f: next((x for x in props[f]["enum"] if x), "")  # noqa: E731
            act = {
                "thought": "next step",
                "tool": props["tool"]["enum"][0],
                "feature_id": fids[0] if fids else "",
                "read_id": first("read_id"),
                "expand_id": first("expand_id"),
                "chart_ids": props["chart_ids"]["items"].get("enum", [])[:1],
            }
            return LLMResult(json.dumps(act), 0.0)
        if "features" in props:
            idea = prompt.split("INVENTION IDEA:\n", 1)[1].split("\n\nTASK", 1)[0]
            feats = (split_sentences(idea) + ["The idea is a technical invention."])[:3]
            return LLMResult(json.dumps({"features": feats}), 0.0)
        raise AssertionError(f"unexpected LLM call with schema {sorted(props)}")


class SubstringNli:
    """Stand-in NLI: entailment 1.0 when the claim text occurs in the premise, else 0.0."""

    def entailment(self, premise: str, claim: str) -> float:
        return 1.0 if claim.strip().rstrip(".") in premise else 0.0


class AlwaysNli:
    def entailment(self, premise: str, claim: str) -> float:
        return 1.0


def length_scorer(query: str, docs: Any) -> list[float]:
    """Stand-in cross-encoder: longer text scores higher (tests the plumbing, not quality)."""
    return [float(len(d)) for d in docs]


@pytest.fixture(scope="module")
def stack():
    _have_data()
    from patentrag.stack import Stack

    return Stack(load_config(), SCOPE)


@pytest.fixture(scope="module")
def queries():
    _have_data()
    from patentrag.evaluation.harness import ensure_testset
    from patentrag.ingest.corpus import load_corpus

    cfg = load_config()
    val, test, _ = ensure_testset(cfg, SCOPE, load_corpus(cfg, SCOPE))
    return val[:3]


@pytest.fixture
def isolated_results(tmp_path, monkeypatch):
    """Point every results writer at a temporary directory."""
    from patentrag.evaluation import ablations, agent_eval, harness, params, robustness

    for mod in (params, ablations, agent_eval, harness, robustness):
        monkeypatch.setattr(mod, "RESULTS", tmp_path)
    return tmp_path


def test_hybrid_and_bm25_respect_exclusion(stack, queries):
    q = queries[0]
    for r in (stack.hybrid(), stack.bm25()):
        hits = r.search(q.text, 20, q.exclusion())
        assert len(hits) == 20 and not ({h.pub_id for h in hits} & set(q.exclude))


def test_hybrid_with_zero_bm25_weight_is_exact_vector_order(stack, queries):
    import numpy as np

    h = stack.hybrid(load_config().hybrid.model_copy(update={"w_bm25": 0.0}))
    for q in queries:
        excl = q.exclusion()
        best, _ = stack.matrix.patent_sims(stack.vector.embed_query(q.text))
        order = [stack.matrix.pub_ids[i] for i in np.argsort(-best, kind="stable")]
        expected = [p for p in order if p not in excl.exclude][:10]
        assert [x.pub_id for x in h.search(q.text, 10, excl)] == expected


def test_rerank_reorders_only_the_top_depth(stack, queries):
    from patentrag.retrieval.reranker import RerankRetriever

    base = stack.hybrid()
    q = queries[0]
    before = base.search(q.text, 30, q.exclusion())
    rr = RerankRetriever(base, stack.texts, length_scorer, depth=10, alpha=1.0)
    after = rr.search(q.text, 30, q.exclusion())
    assert {h.pub_id for h in after[:10]} == {h.pub_id for h in before[:10]}
    assert [h.pub_id for h in after[10:]] == [h.pub_id for h in before[10:]]
    lens = [len(stack.texts[h.pub_id]) for h in after[:10]]
    assert lens == sorted(lens, reverse=True)


@pytest.mark.needs_neo4j
def test_graphrag_with_graph_off_ranks_like_hybrid(stack, queries):
    """The tuning starts GraphRAG from the hybrid: graph switched off, identical ranking."""
    _graph_loaded()
    from patentrag.evaluation.tuning import hybrid_start

    h = load_config().hybrid.model_copy(update={"w_bm25": 0.5})
    g = stack.graphrag(hybrid_start(load_config().graph, h))
    hyb = stack.hybrid(h)
    for q in queries:
        excl = q.exclusion()
        got = [x.pub_id for x in g.search(q.text, 10, excl)]
        assert got == [x.pub_id for x in hyb.search(q.text, 10, excl)]


@pytest.mark.needs_neo4j
def test_graphrag_provenance_and_exclusion(stack, queries):
    _graph_loaded()
    params = load_config().graph.model_copy(update={"use_cite": True, "use_family": True})
    g = stack.graphrag(params)
    q = queries[0]
    res = g.search_detailed(q.text, 20, q.exclusion())
    ids = {h.pub_id for h in res["hits"]}
    assert not (ids & set(q.exclude))
    assert res["n_candidates"] >= params.pool_n
    assert all(h.provenance for h in res["hits"])


@pytest.mark.needs_neo4j
def test_agent_uses_tools_and_returns_a_verified_chart(stack, queries):
    _graph_loaded()
    from patentrag.agent.loop import PriorArtAgent

    q = queries[0]
    agent = PriorArtAgent(
        load_config(), stack, RuleLLM(), nli_fn=AlwaysNli().entailment, nli_threshold=0.5
    )
    run = agent.run(q.text, excl=q.exclusion())
    tools = [s.tool for s in run.steps]
    assert tools[0] == "search_idea" and tools[-1] == "finish" and "claim_chart" in tools
    assert run.searched_features() == {"F1", "F2", "F3"}
    assert tools.count("read_patent") == 3 and tools.count("graph_neighbours") == 2  # caps
    assert run.steps[-2].source == "guardrail"  # the step limit forced the chart
    ranked = [p for p, _ in agent.ranking(run)]
    assert ranked and not (set(ranked) & set(q.exclude))
    rep = run.report.report if run.report else None
    assert rep is not None and rep.items and rep.items[0].chart[0].supported
    retrieval = agent.run(q.text, excl=q.exclusion(), retrieval_only=True)
    assert "claim_chart" not in [s.tool for s in retrieval.steps] and retrieval.report is None


@pytest.mark.needs_neo4j
def test_service_search_agent_report_cache_and_degraded_mode(tmp_path, monkeypatch, queries):
    _graph_loaded()
    from patentrag.service import PatentService

    cfg = load_config()
    cfg.sqlite_path = str(tmp_path / "memory.db")
    svc = PatentService(cfg, scope=SCOPE)
    svc._llm = RuleLLM()
    svc._nli = AlwaysNli()  # type: ignore[assignment]
    text = queries[0].text
    for method in ("auto", "bm25", "vector", "hybrid", "graphrag"):
        res = svc.search(text, k=5, method=method)
        assert len(res.hits) == 5 and not res.degraded
    assert svc.readiness()["neo4j"] is True

    first = svc.report(text)
    assert first.items and first.items[0].chart and first.grounded and not first.cached
    assert first.agent_steps[0].tool == "search_idea" and first.ranking
    assert svc.trace(first.trace_id)["steps"][-1]["tool"] == "summary"
    second = svc.report(text)
    assert second.cached
    # a cache hit returns exactly the checked response, not a reconstruction
    assert second.model_dump(exclude={"cached", "took_ms"}) == first.model_dump(
        exclude={"cached", "took_ms"}
    )

    class Down:
        def search_detailed(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("graph down")

    svc._retrievers["graphrag"] = Down()
    res = svc.search(text, k=5, method="graphrag")
    assert res.degraded and res.method == "hybrid" and len(res.hits) == 5
    svc.memory().close()


def test_evaluate_methods_with_a_precomputed_run(stack, isolated_results):
    from patentrag.evaluation import harness, runner

    cfg = load_config()
    extra = runner.RunResult("fixed")
    val, _, _ = harness.ensure_testset(cfg, SCOPE, list(stack.patents.values()))
    extra.ranked = {q.qid: q.relevant_keys[:1] for q in val}
    out = harness.evaluate_methods(
        cfg, SCOPE, {"bm25": stack.bm25()}, "val", "pytest_eval", extra_runs={"fixed": extra}
    )
    assert set(out["summary"]) == {"bm25", "fixed"} and "bm25-fixed" in out["paired"]
    assert out["summary"]["fixed"]["mrr"]["mean"] == 1.0
    assert (isolated_results / "pytest_eval.json").exists()
    assert harness.summary_table(out, cfg.evaluation.ks).count("\n") == 3


@pytest.mark.needs_neo4j
def test_tune_ablate_robustness_and_agent_on_a_small_subset(stack, isolated_results, monkeypatch):
    _graph_loaded()
    from patentrag.agent.loop import PriorArtAgent
    from patentrag.evaluation import ablations, agent_eval, params, robustness, tuning

    monkeypatch.setenv("PATENTRAG_MAX_QUERIES", "3")
    monkeypatch.setattr(tuning, "HYBRID_GRID", [0.5])
    monkeypatch.setattr(
        tuning, "GRAPH_GRID", [("w_cite", [{"use_cite": True, "w_cite": 0.1}], None)]
    )
    monkeypatch.setattr(
        ablations, "GRAPH_SIGNALS", {"family": {"use_family": True, "w_family": 0.1}}
    )
    cfg = load_config()
    cfg.rerank.candidates = []  # no model download in tests
    tuned = tuning.tune(cfg, SCOPE, stack)
    assert tuned["graph"]["seed_source"] == "hybrid"
    saved = params.load_all(SCOPE)
    assert saved["hybrid"]["n_val_queries"] == 3 and not saved["rerank"]["params"]["enabled"]
    abl = ablations.run_ablations(cfg, SCOPE, stack, "test")
    assert abl["n_queries"] == 3 and "graph_signals" in abl["ablations"]
    rets = {"bm25": stack.bm25(), "hybrid": stack.hybrid()}
    rob = robustness.run_robustness(cfg, SCOPE, rets)
    assert "hybrid-bm25" in rob["corpus_size"]["1.0"]["paired"]

    agent = PriorArtAgent(cfg, stack, RuleLLM(), nli_fn=AlwaysNli().entailment, nli_threshold=0.5)
    res = agent_eval.tune_agent(cfg, SCOPE, agent)
    assert [r["feature_weight"] for r in res["log"]] == agent_eval.FEATURE_WEIGHTS
    assert params.load_all(SCOPE)["agent"]["behaviour_val"]["n_runs"] == 3


@pytest.mark.needs_neo4j
def test_faithfulness_runner_summary_controls_and_audit(stack, isolated_results, monkeypatch):
    _graph_loaded()
    from patentrag.agent.loop import PriorArtAgent
    from patentrag.evaluation import agent_eval, faith_eval

    monkeypatch.setattr(faith_eval, "threshold_path", lambda: isolated_results / "th.json")
    monkeypatch.setenv("PATENTRAG_AGENT_WORKERS", "2")  # concurrent runs, as on the GPU VM
    cfg = load_config()
    llm, nli = RuleLLM(), SubstringNli()
    agent = PriorArtAgent(cfg, stack, llm, nli_fn=nli.entailment, nli_threshold=0.5)
    qs = agent_eval.report_queries(cfg, SCOPE, 0, 3)["test"]
    path = agent_eval.runs_path("report", "test", SCOPE)
    recs = agent_eval.run_queries(agent, qs, path, retrieval_only=False)
    again = agent_eval.run_queries(agent, qs, path, retrieval_only=False)
    assert [r["qid"] for r in again] == [r["qid"] for r in recs]  # resumed from checkpoint
    summary = faith_eval.summarise(recs, 0.5)
    assert summary["first_choice"]["evidence_from_cited_patent"] == 1.0
    assert 0.0 <= summary["shipped"]["pair_fill_rate"] <= 1.0
    assert faith_eval.negative_control(llm, nli, recs, 0.5)["n_claims"] > 0
    cal = faith_eval.calibrate_threshold(nli, recs)
    assert 0.0 < cal["threshold"] < 1.0 and (isolated_results / "th.json").exists()
    audit = faith_eval.audit(llm, recs)
    pairs = sum(len(i["chart"]) for r in recs for i in (r["draft"] or {}).get("items", []))
    assert audit["n_rows_kept"] + audit["n_rows_removed"] == pairs > 0
    assert agent_eval.behaviour(recs)["n_runs"] == 3


def test_cli_commands_without_network(tmp_path):
    from typer.testing import CliRunner

    from patentrag.cli import app

    r = CliRunner()
    assert r.invoke(app, ["config"]).exit_code == 0
    res = r.invoke(app, ["check-credentials", "--no-live"])
    assert res.exit_code in (0, 1) and ("PASS" in res.output or "FAIL" in res.output)
    from patentrag.config import load_secrets

    secret = load_secrets().ops_secret
    assert not secret or secret not in res.output  # never printed


def test_exclusion_type_is_shared():
    ex = Exclusion(exclude=frozenset({"A"}), hide_outgoing_from=frozenset({"A"}))
    assert ex.params() == {"exclude": ["A"], "hide_src": ["A"]}
