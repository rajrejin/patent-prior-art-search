"""Agent control flow and fusion arithmetic with in-memory fakes (no data, no model server).

The fake retriever returns fixed hand-written rankings; the scripted LLM returns pre-set JSON
actions, so each test fixes exactly which decisions the planner makes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from patentrag.agent.loop import PriorArtAgent, Search, action_schema, fuse
from patentrag.config import load_config
from patentrag.llm.client import LLMResult
from patentrag.retrieval.base import NO_EXCLUSION, Exclusion, Hit
from patentrag.retrieval.reranker import blend, minmax

TEXTS = {
    "EP1A1": "Battery pack. A battery pack has a cooling plate under the cells.",
    "EP2A1": "Electrode. An electrode has a carbon coating layer.",
    "EP3A1": "Separator. A separator has ceramic particles.",
    "EP4A1": "Excluded patent. It must never be returned.",
}
IDEA = "A battery pack has a cooling plate. The cells sit on the plate."
FEATURES = ["A battery pack has a cooling plate.", "The cells sit on the plate."]


@dataclass
class FakePatent:
    title: str


class FakeStack:
    def __init__(self) -> None:
        self.texts = dict(TEXTS)
        self.patents = {p: FakePatent(t.split(".")[0]) for p, t in TEXTS.items()}

    @property
    def graph_queries(self):
        raise RuntimeError("no graph in unit tests")


class FakeRetriever:
    """Idea search ranks EP1, EP2, EP3; any other text ranks EP3, EP2."""

    name = "fake"

    def search(self, query: str, k: int, excl: Exclusion = NO_EXCLUSION) -> list[Hit]:
        ids = ["EP1A1", "EP2A1", "EP3A1", "EP4A1"] if query == IDEA else ["EP3A1", "EP2A1"]
        return [Hit(p, 1.0 / (i + 1)) for i, p in enumerate(ids) if p not in excl.exclude][:k]


class ScriptedLLM:
    """Planner actions come from `actions` in order; other calls get fixed valid answers."""

    backend = model = "scripted"

    def __init__(self, actions: list[dict]):
        self.actions = list(actions)
        self.planner_calls = 0

    def available(self) -> bool:
        return True

    def complete(self, prompt, *, system=None, json_schema=None, max_tokens=700) -> LLMResult:
        props = (json_schema or {}).get("properties", {})
        if "tool" in props:
            self.planner_calls += 1
            act = self.actions.pop(0) if self.actions else {"tool": "finish"}
            empty = {"feature_id": "", "read_id": "", "expand_id": "", "chart_ids": []}
            full = {"thought": "t", **empty, **act}
            return LLMResult(json.dumps(full), 0.0)
        if "features" in props:
            return LLMResult(json.dumps({"features": FEATURES}), 0.0)
        if "verdict" in props:
            return LLMResult('{"verdict": "yes"}', 0.0)
        raise AssertionError(f"unexpected LLM call with schema {sorted(props)}")


def make_agent(actions: list[dict], **params) -> tuple[PriorArtAgent, ScriptedLLM]:
    cfg = load_config()
    llm = ScriptedLLM(actions)
    p = cfg.agent.model_copy(update=params)
    agent = PriorArtAgent(
        cfg,
        FakeStack(),
        llm,
        retriever=FakeRetriever(),
        params=p,
        nli_fn=lambda prem, claim: 1.0,
        nli_threshold=0.5,
    )
    return agent, llm


def test_fuse_weighted_reciprocal_rank_hand_computed():
    idea = Search("idea", "q", [Hit("A", 1), Hit("B", 1)])
    feat = Search("F1", "f", [Hit("B", 1), Hit("C", 1)])
    # k=1: A = 1/2; B = 1/3 + 0.5 * 1/2 = 0.5833; C = 0.5 * 1/3 = 0.1667
    ranked = fuse([idea, feat], 0.5, 1)
    assert [p for p, _ in ranked] == ["B", "A", "C"]
    assert ranked[0][1] == pytest.approx(1 / 3 + 0.25)
    assert [p for p, _ in fuse([idea, feat], 0.0, 1)] == ["A", "B"]  # weight 0 = idea only


def test_action_schema_offers_only_allowed_tools_and_valid_ids_per_tool():
    s = action_schema(["search_idea", "finish"], ["F1"], ["EP2A1"], ["EP1A1"], ["EP1A1", "EP2A1"])
    p = s["properties"]
    assert p["tool"]["enum"] == ["search_idea", "finish"]
    assert p["feature_id"]["enum"] == ["", "F1"]
    assert p["read_id"]["enum"] == ["", "EP2A1"] and p["expand_id"]["enum"] == ["", "EP1A1"]
    assert p["chart_ids"]["items"]["enum"] == ["EP1A1", "EP2A1"] and p["chart_ids"]["maxItems"] == 3
    empty = action_schema(["search_idea"], [], [], [], [])["properties"]
    assert empty["chart_ids"]["maxItems"] == 0


def test_planner_decisions_are_followed_and_recorded():
    agent, llm = make_agent(
        [
            {"tool": "search_idea"},
            {"tool": "extract_features"},
            {"tool": "search_feature", "feature_id": "F2"},
            {"tool": "read_patent", "read_id": "EP3A1"},
            {"tool": "claim_chart", "chart_ids": ["EP1A1", "EP2A1"]},
            {"tool": "finish"},
        ]
    )
    run = agent.run(IDEA, excl=Exclusion(exclude=frozenset({"EP4A1"})))
    assert [s.tool for s in run.steps] == [
        "search_idea",
        "extract_features",
        "search_feature",
        "read_patent",
        "claim_chart",
        "finish",
    ]
    assert all(s.source == "planner" for s in run.steps)
    assert run.searched_features() == {"F2"} and run.read == ["EP3A1"]
    assert "EP4A1" not in [p for p, _ in agent.ranking(run)]
    rep = run.report.report if run.report else None
    assert rep is not None and rep.items[0].pub_id == "EP1A1"
    # no embedding model in the fake stack: evidence is ranked by shared words
    assert rep.items[0].chart[0].evidence == "A battery pack has a cooling plate under the cells."
    stats = run.stats()
    assert stats["planner_actions_used"] == 6 and stats["replaced_actions"] == 0


def test_invalid_actions_are_replaced_by_the_fallback_policy():
    agent, _ = make_agent(
        [
            {"tool": "search_idea"},
            {"tool": "search_idea"},  # not offered again -> fallback: extract_features
            {"tool": "read_patent", "read_id": ""},  # no id -> fallback: search_feature F1
            {"tool": "finish"},  # before the chart -> guardrail runs the chart first
            {"tool": "finish"},
        ],
        max_feature_searches=1,
    )
    run = agent.run(IDEA)
    assert [(s.tool, s.source) for s in run.steps] == [
        ("search_idea", "planner"),
        ("extract_features", "fallback"),
        ("search_feature", "fallback"),
        ("claim_chart", "guardrail"),
        ("finish", "planner"),
    ]
    assert run.steps[1].error.startswith("tool search_idea not available")
    assert run.report is not None and run.report.report is not None


def test_step_limit_forces_the_chart_and_finish():
    reads = [{"tool": "read_patent", "read_id": p} for p in ("EP1A1", "EP2A1", "EP3A1")]
    agent, _ = make_agent([{"tool": "search_idea"}, *reads], max_steps=3)
    run = agent.run(IDEA)
    assert [s.tool for s in run.steps][-2:] == ["claim_chart", "finish"]
    assert run.steps[-1].source == "guardrail" and len(run.steps) == 5


def test_retrieval_mode_has_no_claim_chart():
    agent, _ = make_agent(
        [{"tool": "search_idea"}, {"tool": "claim_chart", "chart_ids": ["EP1A1"]}]
    )
    run = agent.run(IDEA, retrieval_only=True)
    assert "claim_chart" not in [s.tool for s in run.steps] and run.report is None
    assert run.steps[1].source == "fallback"  # claim_chart is not offered in retrieval mode


def test_graph_tool_degrades_when_the_graph_is_down():
    agent, _ = make_agent(
        [{"tool": "search_idea"}, {"tool": "graph_neighbours", "expand_id": "EP1A1"}]
    )
    run = agent.run(IDEA, retrieval_only=True)
    assert run.steps[1].tool == "graph_neighbours"
    assert run.steps[1].observation.startswith("graph unavailable")


def test_blend_reorders_top_depth_and_keeps_the_tail():
    hits = [Hit("A", 3.0), Hit("B", 2.0), Hit("C", 1.0), Hit("D", 0.5)]
    out = blend(hits, [0.0, 1.0, 0.5], depth=3, alpha=1.0)
    assert [h.pub_id for h in out] == ["B", "C", "A", "D"]
    # alpha 0.5: A = 0.5*0 + 0.5*1 = 0.5, B = 0.5*1 + 0.5*0.5 = 0.75, C = 0.25 + 0 = 0.25
    half = blend(hits, [0.0, 1.0, 0.5], depth=3, alpha=0.5)
    assert [h.pub_id for h in half][:3] == ["B", "A", "C"]
    assert half[0].score == pytest.approx(0.75) and "rerank" in half[0].provenance
    assert minmax([2.0, 2.0]) == [0.0, 0.0]


def test_concurrent_runs_record_every_query_once(tmp_path, monkeypatch):
    from patentrag.evaluation import agent_eval
    from patentrag.evaluation.testset import Query

    monkeypatch.setenv("PATENTRAG_AGENT_WORKERS", "3")
    agent, _ = make_agent([])  # no scripted actions: the planner always answers finish
    qs = [Query(f"Q{i}", f"EP9{i}A1", IDEA, ["EP1"], [], 2020, "H01M") for i in range(7)]
    path = tmp_path / "runs.jsonl"
    recs = agent_eval.run_queries(agent, qs, path, retrieval_only=True)
    assert [r["qid"] for r in recs] == [q.qid for q in qs]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 7
    assert agent_eval.behaviour(recs)["concurrent_runs"] == 3
    again = agent_eval.run_queries(agent, qs, path, retrieval_only=True)  # resumes, runs nothing
    assert len(again) == 7 and len(path.read_text(encoding="utf-8").splitlines()) == 7
