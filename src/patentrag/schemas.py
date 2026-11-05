"""Public response models shared by the MCP server, the API and the UI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Method = Literal["auto", "bm25", "vector", "hybrid", "rerank", "graphrag"]


class SearchHit(BaseModel):
    rank: int
    pub_id: str
    title: str
    score: float
    pub_date: str = ""
    applicants: list[str] = Field(default_factory=list)
    cpc: list[str] = Field(default_factory=list)
    snippet: str = ""
    provenance: list[str] = Field(default_factory=list, description="Why this result was found")


class SearchResponse(BaseModel):
    method: Method
    requested_method: Method
    degraded: bool = False
    degraded_reason: str = ""
    inferred_cpc: list[str] = Field(default_factory=list)
    n_candidates: int | None = None
    took_ms: float
    hits: list[SearchHit]


class ChartRowOut(BaseModel):
    feature: str = Field(description="A technical feature of the invention idea")
    evidence: str = Field(description="Sentence of this patent's retrieved text that states it")


class ReportItem(BaseModel):
    pub_id: str
    title: str
    relevance: str
    evidence: str
    evidence_verbatim: bool
    claims_supported: bool | None = None
    chart: list[ChartRowOut] = Field(
        default_factory=list, description="Claim chart: verified feature-to-evidence rows"
    )


class ClaimSupport(BaseModel):
    claim: str
    source: str
    supported: bool
    llm_supported: bool | None = None
    nli_probability: float | None = None


class AgentStepOut(BaseModel):
    n: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    thought: str = ""
    source: str = Field(
        description="planner (the LLM chose it), fallback (replaced an invalid action) or "
        "guardrail (forced by a rule)"
    )
    error: str = ""
    observation: str = ""
    seconds: float = 0.0


class ReportResponse(BaseModel):
    trace_id: str
    summary: str
    features: list[str] = Field(default_factory=list)
    items: list[ReportItem]
    limitations: str
    claim_support: list[ClaimSupport]
    grounded: bool = Field(
        description="The returned report has at least one chart row and every returned row "
        "passed all grounding checks"
    )
    draft_passed: bool = Field(
        default=False, description="Every feature and patent pair got a verified row"
    )
    removed_rows: int = Field(
        default=0, description="Feature and patent pairs left out: no sentence passed the checks"
    )
    attempts: int
    warning: str = ""
    valid_citation_fraction: float | None = None
    evidence_valid_fraction: float | None = None
    agent_steps: list[AgentStepOut] = Field(default_factory=list)
    ranking: list[str] = Field(
        default_factory=list, description="The agent's fused ranking of all its searches"
    )
    took_ms: float
    cached: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    request_id: str = ""
