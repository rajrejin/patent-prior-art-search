"""The prior-art agent: an LLM chooses tools step by step, observes the results, and stops with a
verified claim chart.

Loop (at most `max_steps`):
  1. The planner LLM sees the idea, the features found so far, the candidate patents and the
     history of observations, and answers with one JSON action: a tool name plus arguments.
     The JSON schema only offers tools that are currently allowed and ids that exist, so the
     action is always well formed.
  2. The action is checked (for example: a feature that was already searched, a patent that was
     already read). An invalid action is replaced by the next step of a fixed fallback policy,
     and the replacement is recorded.
  3. The tool runs and its observation is appended to the history.
Guardrails: the run always contains a search of the whole idea and, in report mode, a claim
chart; finishing without one runs it first (recorded as an intervention). Every step is traced.

Tools: search_idea, extract_features, search_feature, read_patent, graph_neighbours,
claim_chart, finish. The final ranking fuses all searches by weighted reciprocal rank
(the idea search has weight 1, each feature search `feature_weight`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from patentrag.agent.memory import Memory, query_hash
from patentrag.agent.report import (
    RankFn,
    ReportRun,
    attempt_summary,
    embedding_ranker,
    extract_features,
    generate_report,
    rank_by_overlap,
)
from patentrag.config import AgentCfg, Config
from patentrag.faithfulness.judge import llm_judge
from patentrag.faithfulness.threshold import load_threshold
from patentrag.llm.client import LLMClient, generate_structured
from patentrag.retrieval.base import NO_EXCLUSION, Exclusion, Hit, Retriever

TOOLS: dict[str, str] = {
    "search_idea": "Search the patent corpus with the whole invention idea.",
    "extract_features": "Split the idea into 3 to 5 technical features.",
    "search_feature": "Search the corpus for one feature. Set feature_id.",
    "read_patent": "Read title, abstract and first claim of one unread candidate. Set read_id.",
    "graph_neighbours": "Show patents linked to one candidate in the knowledge graph (citations, "
    "family, CPC class). Set expand_id.",
    "claim_chart": "Map the features to evidence sentences in the best 1 to 3 candidates and "
    "verify every row. Set chart_ids.",
    "finish": "Stop. The verified claim chart is the answer.",
}
PLANNER_SYSTEM = (
    "You are a patent prior-art search agent. Find the existing patents closest to an "
    "invention idea, then build a verified claim chart. Choose ONE tool per turn. Search the "
    "whole idea first, extract the features, search the most distinctive features, inspect "
    "promising candidates if useful, then build the claim chart with the best candidates and "
    "finish."
)


class Action(BaseModel):
    thought: str
    tool: str
    feature_id: str = ""
    read_id: str = ""
    expand_id: str = ""
    chart_ids: list[str] = []


def action_schema(
    tools: list[str],
    feature_ids: list[str],
    read_ids: list[str],
    expand_ids: list[str],
    chart_ids: list[str],
) -> dict[str, Any]:
    """One id field per tool, each an enum of the ids that are valid for that tool right now
    (unread candidates, candidates not yet expanded, all candidates), so a repeated read or
    expansion cannot even be expressed."""
    chart: dict[str, Any] = (
        {"type": "string", "enum": chart_ids} if chart_ids else {"type": "string"}
    )
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "maxLength": 200},
            "tool": {"type": "string", "enum": tools},
            "feature_id": {"type": "string", "enum": ["", *feature_ids]},
            "read_id": {"type": "string", "enum": ["", *read_ids]},
            "expand_id": {"type": "string", "enum": ["", *expand_ids]},
            "chart_ids": {"type": "array", "items": chart, "maxItems": 3 if chart_ids else 0},
        },
        "required": ["thought", "tool", "feature_id", "read_id", "expand_id", "chart_ids"],
    }


@dataclass
class Search:
    kind: str  # "idea" or a feature id
    text: str
    hits: list[Hit]


@dataclass
class Step:
    n: int
    tool: str
    args: dict[str, Any]
    thought: str = ""
    source: str = "planner"  # planner | fallback | guardrail
    error: str = ""  # why the planner's action was replaced
    observation: str = ""
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "tool": self.tool,
            "args": self.args,
            "thought": self.thought,
            "source": self.source,
            "error": self.error,
            "observation": self.observation,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class AgentRun:
    query: str
    features: list[str] = field(default_factory=list)
    searches: list[Search] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    neighbours: dict[str, list[str]] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    report: ReportRun | None = None
    planner_calls: int = 0
    planner_errors: int = 0
    trace_id: str = ""
    seconds: float = 0.0

    def searched_features(self) -> set[str]:
        return {s.kind for s in self.searches if s.kind != "idea"}

    def has(self, tool: str) -> bool:
        return any(s.tool == tool for s in self.steps)

    def stats(self) -> dict[str, Any]:
        tools: dict[str, int] = {}
        for s in self.steps:
            tools[s.tool] = tools.get(s.tool, 0) + 1
        return {
            "steps": len(self.steps),
            "planner_calls": self.planner_calls,
            "planner_actions_used": sum(s.source == "planner" for s in self.steps),
            "replaced_actions": sum(s.source == "fallback" for s in self.steps),
            "guardrail_steps": sum(s.source == "guardrail" for s in self.steps),
            "planner_errors": self.planner_errors,
            "tools": tools,
            "feature_searches": len(self.searched_features()),
            "seconds": round(self.seconds, 1),
        }


def fuse(searches: list[Search], feature_weight: float, rrf_k: int) -> list[tuple[str, float]]:
    """Weighted reciprocal-rank fusion; the idea search has weight 1. Ties break on pub_id."""
    score: dict[str, float] = {}
    for s in searches:
        w = 1.0 if s.kind == "idea" else feature_weight
        if w <= 0:
            continue
        for rank, h in enumerate(s.hits, 1):
            score[h.pub_id] = score.get(h.pub_id, 0.0) + w / (rrf_k + rank)
    return sorted(score.items(), key=lambda t: (-t[1], t[0]))


class PriorArtAgent:
    def __init__(
        self,
        cfg: Config,
        stack: Any,
        llm: LLMClient | None,
        *,
        retriever: Retriever | None = None,
        params: AgentCfg | None = None,
        memory: Memory | None = None,
        nli_fn: Callable[[str, str], float] | None = None,
        nli_threshold: float | None = None,
    ):
        self.cfg = cfg
        self.stack = stack
        self.llm = llm
        self.retriever = retriever or stack.best()
        self.p = params or cfg.agent
        self.memory = memory
        self.nli_fn = nli_fn
        self._threshold = nli_threshold

    @property
    def nli_threshold(self) -> float:
        return load_threshold() if self._threshold is None else self._threshold

    # -- state views -----------------------------------------------------------------------------
    def ranking(self, run: AgentRun) -> list[tuple[str, float]]:
        return fuse(run.searches, self.p.feature_weight, self.p.rrf_k)

    def candidates(self, run: AgentRun, n: int = 10) -> list[str]:
        ids = [p for p, _ in self.ranking(run)[:n]]
        for nbrs in run.neighbours.values():
            ids += [p for p in nbrs if p not in ids]
        return ids[: n + 5]

    def _title(self, pid: str) -> str:
        p = self.stack.patents.get(pid)
        return (p.title if p else "")[:80]

    # -- tools -----------------------------------------------------------------------------------
    def _search(self, run: AgentRun, kind: str, text: str, excl: Exclusion) -> str:
        hits = self.retriever.search(text, self.p.search_k, excl)
        run.searches.append(Search(kind, text, hits))
        top = ", ".join(f"{h.pub_id} ({self._title(h.pub_id)[:50]})" for h in hits[:5])
        return f"{len(hits)} results; top: {top}"

    def _features(self, run: AgentRun) -> str:
        assert self.llm is not None
        run.features, _ = extract_features(self.llm, run.query, 1)
        if not run.features:
            return "could not extract features"
        return " | ".join(f"F{i}: {f}" for i, f in enumerate(run.features, 1))

    def _read(self, run: AgentRun, pid: str) -> str:
        run.read.append(pid)
        return " ".join(self.stack.texts.get(pid, "").split()[:120])

    def _graph(self, run: AgentRun, pid: str, excl: Exclusion) -> str:
        try:
            q = self.stack.graph_queries
            nb = q.citation_neighbours([pid], 1, excl)
            fam = [r["dst"] for r in q.family_members([pid], excl)]
            cpcs = q.patent_cpcs([pid]).get(pid, [])
        except Exception as exc:  # noqa: BLE001
            run.neighbours[pid] = []
            return f"graph unavailable ({type(exc).__name__})"
        ids = [x for x in nb if x not in excl.exclude]
        run.neighbours[pid] = [x for x in ids + fam if x in self.stack.patents][:5]
        cites = [f"{x} ({nb[x]['direction']})" for x in ids[:6]]
        links, family = ", ".join(cites) or "none", ", ".join(fam[:3]) or "none"
        return f"citation links: {links}; family: {family}; CPC: {', '.join(cpcs[:4])}"

    def _ranker(self) -> RankFn:
        """Evidence sentences are ranked with the retrieval embedding model when there is one."""
        vector = getattr(self.stack, "vector", None)
        return embedding_ranker(vector.embedder) if vector is not None else rank_by_overlap

    def _chart(self, run: AgentRun, ids: list[str], max_retries: int) -> str:
        assert self.llm is not None
        llm = self.llm
        ids = [p for p in ids if p in self.stack.texts][: self.p.max_chart_items]
        sources = {p: self.stack.texts[p] for p in ids}
        run.report = generate_report(
            llm,
            run.query,
            sources,
            features=run.features or None,
            max_items=self.p.max_chart_items,
            max_retries=max_retries,
            llm_fn=lambda src, claim: llm_judge(llm, src, claim),
            nli_fn=self.nli_fn,
            nli_threshold=self.nli_threshold,
            flag_if_either=self.cfg.faithfulness.flag_if_either,
            rank_fn=self._ranker(),
        )
        if not run.features and run.report.report:
            run.features = run.report.report.features
        rep = run.report.report
        if rep is None:
            return run.report.warning or "no chart"
        rows = sum(len(i.chart) for i in rep.items)
        return (
            f"{len(rep.items)} patents with {rows} verified rows; "
            f"{rep.removed_rows} feature and patent pairs without verified evidence"
        )

    # -- planning ----------------------------------------------------------------------------------
    def allowed(self, run: AgentRun, retrieval_only: bool) -> list[str]:
        out = []
        if not run.has("search_idea"):
            out.append("search_idea")
        if not run.has("extract_features"):
            out.append("extract_features")
        if run.features and len(run.searched_features()) < min(
            self.p.max_feature_searches, len(run.features)
        ):
            out.append("search_feature")
        if run.searches:
            cands = self.candidates(run)
            if len(run.read) < self.p.max_reads and any(p not in run.read for p in cands):
                out.append("read_patent")
            if len(run.neighbours) < self.p.max_expansions and any(
                p not in run.neighbours for p in cands
            ):
                out.append("graph_neighbours")
            if not retrieval_only and run.report is None:
                out.append("claim_chart")
            out.append("finish")
        return out

    def fallback(self, run: AgentRun, retrieval_only: bool) -> tuple[str, dict[str, Any]]:
        """The fixed policy used when the planner's action is invalid."""
        if not run.has("search_idea"):
            return "search_idea", {}
        if not run.has("extract_features"):
            return "extract_features", {}
        todo = [f"F{i}" for i in range(1, len(run.features) + 1)]
        todo = [f for f in todo if f not in run.searched_features()]
        if todo and len(run.searched_features()) < self.p.max_feature_searches:
            return "search_feature", {"feature_id": todo[0]}
        if not retrieval_only and run.report is None:
            top = [p for p, _ in self.ranking(run)[: self.p.max_chart_items]]
            return "claim_chart", {"pub_ids": top}
        return "finish", {}

    def prompt(self, run: AgentRun, retrieval_only: bool, cands: list[str]) -> str:
        idea = " ".join(run.query.split()[:150])
        feats = "\n".join(f"F{i}: {f}" for i, f in enumerate(run.features, 1)) or "(none yet)"
        hist = "\n".join(
            f"{s.n}. {s.tool} {s.args or ''} -> {s.observation[:220]}" for s in run.steps[-8:]
        )
        found_by: dict[str, list[str]] = {}
        for s in run.searches:
            for h in s.hits[:10]:
                found_by.setdefault(h.pub_id, []).append(s.kind)
        cand_lines = "\n".join(
            f"{p}: {self._title(p)} [found by: {', '.join(found_by.get(p, ['graph']))}]"
            + (" [read]" if p in run.read else "")
            for p in cands
        )
        tools = "\n".join(f"- {t}: {TOOLS[t]}" for t in self.allowed(run, retrieval_only))
        searched = ", ".join(sorted(run.searched_features())) or "none"
        return (
            f"INVENTION IDEA:\n{idea}\n\nFEATURES:\n{feats}\nFeatures already searched: "
            f"{searched}\n\nCANDIDATES (best first):\n{cand_lines or '(none yet)'}\n\n"
            f"HISTORY:\n{hist or '(start)'}\n\nAVAILABLE TOOLS:\n{tools}\n\n"
            "Reply with JSON: thought (one short sentence), tool, and only the field that "
            "tool needs: feature_id for search_feature, read_id for read_patent, expand_id for "
            "graph_neighbours, chart_ids for claim_chart. Leave the other fields empty."
        )

    def plan(self, run: AgentRun, retrieval_only: bool) -> tuple[str, dict[str, Any], str, str]:
        """(tool, args, thought, error). An empty error means the planner's action is used."""
        allowed = self.allowed(run, retrieval_only)
        cands = self.candidates(run)
        fids = [f"F{i}" for i in range(1, len(run.features) + 1)]
        assert self.llm is not None
        run.planner_calls += 1
        try:
            act, _ = generate_structured(
                self.llm,
                Action,
                self.prompt(run, retrieval_only, cands),
                system=PLANNER_SYSTEM,
                json_schema=action_schema(
                    allowed,
                    fids,
                    [p for p in cands if p not in run.read],
                    [p for p in cands if p not in run.neighbours],
                    cands,
                ),
                max_tokens=160,
            )
        except ValueError as exc:
            run.planner_errors += 1
            return "", {}, "", f"invalid output: {str(exc)[:80]}"
        err = self.check(run, act, allowed, cands)
        args: dict[str, Any] = {}
        if act.tool == "search_feature":
            args = {"feature_id": act.feature_id}
        elif act.tool == "read_patent":
            args = {"pub_ids": [act.read_id]}
        elif act.tool == "graph_neighbours":
            args = {"pub_ids": [act.expand_id]}
        elif act.tool == "claim_chart":
            args = {"pub_ids": [p for p in act.chart_ids if p in cands]}
        return act.tool, args, act.thought[:200], err

    def check(self, run: AgentRun, act: Action, allowed: list[str], cands: list[str]) -> str:
        if act.tool not in allowed:
            return f"tool {act.tool} not available now"
        if act.tool == "search_feature":
            n = len(run.features)
            ok_ids = {f"F{i}" for i in range(1, n + 1)}
            if act.feature_id not in ok_ids or act.feature_id in run.searched_features():
                return f"feature {act.feature_id or '(none)'} missing or already searched"
        if act.tool == "read_patent" and (act.read_id not in cands or act.read_id in run.read):
            return f"read_id {act.read_id or '(none)'} is not an unread candidate"
        if act.tool == "graph_neighbours" and (
            act.expand_id not in cands or act.expand_id in run.neighbours
        ):
            return f"expand_id {act.expand_id or '(none)'} is not an unexpanded candidate"
        if act.tool == "claim_chart" and not [p for p in act.chart_ids if p in cands]:
            return "no candidate id given for the claim chart"
        return ""

    # -- run ---------------------------------------------------------------------------------------
    def execute(
        self, run: AgentRun, tool: str, args: dict[str, Any], excl: Exclusion, retries: int
    ) -> str:
        if tool == "search_idea":
            return self._search(run, "idea", run.query, excl)
        if tool == "extract_features":
            return self._features(run)
        if tool == "search_feature":
            fid = args["feature_id"]
            return self._search(run, fid, run.features[int(fid[1:]) - 1], excl)
        if tool == "read_patent":
            return self._read(run, args["pub_ids"][0])
        if tool == "graph_neighbours":
            return self._graph(run, args["pub_ids"][0], excl)
        if tool == "claim_chart":
            return self._chart(run, args["pub_ids"], retries)
        return "finished"

    def run(
        self,
        query: str,
        *,
        excl: Exclusion = NO_EXCLUSION,
        retrieval_only: bool = False,
        max_retries: int | None = None,
    ) -> AgentRun:
        """Run the agent. `retrieval_only` removes the claim-chart tool (retrieval evaluation)."""
        if self.llm is None:
            raise RuntimeError("the agent needs an LLM client")
        t_start = time.perf_counter()
        retries = self.cfg.faithfulness.max_retries if max_retries is None else max_retries
        run = AgentRun(query)
        while True:
            n = len(run.steps) + 1
            if n > self.p.max_steps:
                tool, args = self.fallback(run, retrieval_only)
                tool = tool if tool == "claim_chart" else "finish"
                thought, source, err = "", "guardrail", "step limit reached"
            else:
                tool, args, thought, err = self.plan(run, retrieval_only)
                source = "planner"
                if err:
                    tool, args = self.fallback(run, retrieval_only)
                    source = "fallback"
            if tool == "finish" and not run.has("search_idea"):
                tool, args, source, err = "search_idea", {}, "guardrail", "finish before search"
            elif tool == "finish" and not retrieval_only and run.report is None:
                top = [p for p, _ in self.ranking(run)[: self.p.max_chart_items]]
                tool, args, source = "claim_chart", {"pub_ids": top}, "guardrail"
                err = "finish before claim chart"
            t0 = time.perf_counter()
            obs = self.execute(run, tool, args, excl, retries)
            run.steps.append(
                Step(n, tool, args, thought, source, err, obs, time.perf_counter() - t0)
            )
            if tool == "finish":
                break
        run.seconds = time.perf_counter() - t_start
        if self.memory is not None and not retrieval_only:
            rep = run.report.report if run.report else None
            ok = bool(rep and rep.items)
            run.trace_id = self.memory.save_trace(
                query_hash(query, model=self.cfg.llm.model),
                "ok" if ok and not (run.report and run.report.warning) else "warning",
                [s.as_dict() for s in run.steps]
                + [{"tool": "summary", "output": self.summary(run)}],
                rep.model_dump() if rep else None,
            )
        return run

    def summary(self, run: AgentRun) -> dict[str, Any]:
        rep = run.report
        return {
            "stats": run.stats(),
            "features": run.features,
            "ranking": [p for p, _ in self.ranking(run)[:10]],
            "report_attempts": [attempt_summary(a) for a in rep.attempts] if rep else [],
            "warning": rep.warning if rep else "",
        }
