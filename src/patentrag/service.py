"""Service layer: one object that owns the retrieval stack, the agent, the LLM and memory.

Used by the MCP server, the API and the CLI. Components load lazily, so a missing Neo4j or
Ollama degrades the service instead of crashing it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from patentrag.agent.loop import AgentRun, PriorArtAgent
from patentrag.agent.memory import Memory, query_hash
from patentrag.config import Config, load_config
from patentrag.evaluation.params import load_tuned_agent, load_tuned_rerank
from patentrag.faithfulness.nli import NliJudge
from patentrag.index.build import index_dirs
from patentrag.ingest.corpus import corpus_paths
from patentrag.llm.client import LLMClient
from patentrag.llm.ollama_backend import OllamaClient
from patentrag.retrieval.base import NO_EXCLUSION, Exclusion, Hit
from patentrag.schemas import (
    AgentStepOut,
    ChartRowOut,
    ClaimSupport,
    Method,
    ReportItem,
    ReportResponse,
    SearchHit,
    SearchResponse,
)
from patentrag.stack import Stack

log = logging.getLogger(__name__)


def default_scope(cfg: Config) -> str:
    for scope in ("full", "slice"):
        if corpus_paths(cfg, scope)[0].exists():
            return scope
    return "full"


class PatentService:
    def __init__(self, cfg: Config | None = None, scope: str | None = None):
        self.cfg = cfg or load_config()
        self.scope = scope or default_scope(self.cfg)
        self.stack = Stack(self.cfg, self.scope)
        self._lock = threading.Lock()
        self._llm: LLMClient | None = None
        self._nli: NliJudge | None = None
        self._memory: Memory | None = None
        self._agent: PriorArtAgent | None = None
        self._retrievers: dict[str, Any] = {}

    @property
    def corpus_hash(self) -> str:
        man = corpus_paths(self.cfg, self.scope)[1]
        return (
            str(json.loads(man.read_text(encoding="utf-8"))["corpus_hash"]) if man.exists() else ""
        )

    # -- lazy components -------------------------------------------------------------------------
    def retriever(self, method: str) -> Any:
        with self._lock:
            if method not in self._retrievers:
                s = self.stack
                build = {
                    "bm25": s.bm25,
                    "vector": lambda: s.vector,
                    "hybrid": s.hybrid,
                    "rerank": lambda: s.rerank(load_tuned_rerank(self.cfg, self.scope)),
                    "graphrag": s.graphrag,
                    "auto": s.best,
                }[method]
                self._retrievers[method] = build()
            return self._retrievers[method]

    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = OllamaClient(self.cfg)
        return self._llm

    def memory(self) -> Memory:
        if self._memory is None:
            self._memory = Memory(self.cfg.path(self.cfg.sqlite_path))
        return self._memory

    def nli(self) -> NliJudge:
        if self._nli is None:
            self._nli = NliJudge(self.cfg.faithfulness.nli_model)
        return self._nli

    def agent(self) -> PriorArtAgent:
        if self._agent is None:
            self._agent = PriorArtAgent(
                self.cfg,
                self.stack,
                self.llm(),
                retriever=self.retriever("auto"),
                params=load_tuned_agent(self.cfg, self.scope),
                memory=self.memory(),
                nli_fn=self.nli().entailment,
            )
        return self._agent

    # -- health ----------------------------------------------------------------------------------
    def readiness(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        chroma, bm25, _ = index_dirs(self.cfg, self.scope)
        out["indexes"] = chroma.exists() and bm25.exists()
        try:
            from patentrag.graph.loader import connect, db_counts

            d = connect()
            out["neo4j"] = db_counts(d)["Patent"] > 0
            d.close()
        except Exception as exc:  # noqa: BLE001
            out["neo4j"] = False
            out["neo4j_error"] = type(exc).__name__
        out["llm"] = self.llm().available()
        out["scope"] = self.scope
        return out

    # -- search ----------------------------------------------------------------------------------
    def _to_hits(self, hits: list[Hit], k: int) -> list[SearchHit]:
        out = []
        patents = self.stack.patents
        for i, h in enumerate(hits[:k], 1):
            p = patents[h.pub_id]
            snippet = str(h.extra.get("best_chunk", "")) or p.abstract
            out.append(
                SearchHit(
                    rank=i,
                    pub_id=h.pub_id,
                    title=p.title,
                    score=round(float(h.score), 5),
                    pub_date=p.pub_date.isoformat() if p.pub_date else "",
                    applicants=p.applicants[:3],
                    cpc=p.cpc[:5],
                    snippet=" ".join(snippet.split())[:400],
                    provenance=h.provenance[:6],
                )
            )
        return out

    def search(
        self, query: str, k: int = 10, method: Method = "auto", excl: Exclusion = NO_EXCLUSION
    ) -> SearchResponse:
        t0 = time.perf_counter()
        used: str = method
        if method == "auto":
            used = "rerank" if load_tuned_rerank(self.cfg, self.scope).enabled else "hybrid"
        degraded, reason = False, ""
        inferred: list[str] = []
        n_cand: int | None = None
        if method == "graphrag":
            try:
                res = self.retriever("graphrag").search_detailed(query, k, excl)
                hits = res["hits"]
                inferred = [c for c, _ in res["cpc"]]
                n_cand = res["n_candidates"]
            except Exception as exc:  # noqa: BLE001
                log.warning("graph unavailable, degrading to hybrid: %s", type(exc).__name__)
                self._retrievers.pop("graphrag", None)
                used, degraded, reason = "hybrid", True, f"graph unavailable ({type(exc).__name__})"
                hits = self.retriever("hybrid").search(query, k, excl)
        else:
            hits = self.retriever(method).search(query, k, excl)
        return SearchResponse(
            method=used,  # type: ignore[arg-type]
            requested_method=method,
            degraded=degraded,
            degraded_reason=reason,
            inferred_cpc=inferred,
            n_candidates=n_cand,
            took_ms=round((time.perf_counter() - t0) * 1000, 1),
            hits=self._to_hits(hits, k),
        )

    # -- agent report ----------------------------------------------------------------------------
    def report(self, query: str, excl: Exclusion = NO_EXCLUSION) -> ReportResponse:
        """Run the agent. The whole checked response is cached, so a cache hit returns exactly
        what was checked. Evaluation calls (with exclusions) are never cached."""
        t0 = time.perf_counter()
        key = query_hash(query, model=self.cfg.llm.model, kind="agent")
        cacheable = not excl.exclude
        if cacheable:
            hit = self.memory().get(key, self.corpus_hash, "report_response")
            if hit is not None:
                cached = ReportResponse.model_validate(hit)
                ms = round((time.perf_counter() - t0) * 1000, 1)
                return cached.model_copy(update={"cached": True, "took_ms": ms})
        run = self.agent().run(query, excl=excl)
        out = self.to_response(run, round((time.perf_counter() - t0) * 1000, 1))
        if cacheable:
            self.memory().put(key, self.corpus_hash, "report_response", out.model_dump())
        return out

    def to_response(self, run: AgentRun, took_ms: float) -> ReportResponse:
        rr = run.report
        report = rr.report if rr else None
        chart_attempts = [a for a in rr.attempts if a.stage == "chart"] if rr else []
        last = chart_attempts[-1] if chart_attempts else None
        sup = last.support if last else None
        cit = last.citation if last else None
        # The returned report holds only rows that passed every check (agent/report.py).
        items = [
            ReportItem(
                pub_id=i.pub_id,
                title=i.title,
                relevance=i.relevance,
                evidence=i.evidence,
                evidence_verbatim=True,
                claims_supported=True,
                chart=[ChartRowOut(feature=r.feature, evidence=r.evidence) for r in i.chart],
            )
            for i in (report.items if report else [])
        ]
        return ReportResponse(
            trace_id=run.trace_id,
            summary=report.summary if report else "",
            features=report.features if report else run.features,
            items=items,
            limitations=report.limitations if report else "",
            claim_support=[
                ClaimSupport(
                    claim=v.claim.text,
                    source=v.claim.source,
                    supported=v.supported,
                    llm_supported=v.llm_supported,
                    nli_probability=None if v.nli_prob is None else round(v.nli_prob, 3),
                )
                for v in (sup.verdicts if sup else [])
            ],
            grounded=bool(items),
            draft_passed=bool(rr and rr.ok),
            removed_rows=report.removed_rows if report else 0,
            attempts=len(chart_attempts),
            warning=rr.warning if rr else "The agent did not produce a claim chart.",
            valid_citation_fraction=cit.valid_citation_fraction if cit else None,
            evidence_valid_fraction=cit.evidence_valid_fraction if cit else None,
            agent_steps=[AgentStepOut(**s.as_dict()) for s in run.steps],
            ranking=[p for p, _ in self.agent().ranking(run)[:10]],
            took_ms=took_ms,
        )

    def trace(self, trace_id: str) -> dict[str, Any] | None:
        return self.memory().load_trace(trace_id)
