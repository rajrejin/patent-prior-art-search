"""Pydantic schemas for LLM outputs and the prior-art report."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChartRow(BaseModel):
    """One claim-chart row: a feature of the idea and the source sentence that discloses it."""

    feature: str
    evidence: str
    supported: bool | None = None


class PriorArtItem(BaseModel):
    pub_id: str = Field(description="Publication id exactly as given in the sources")
    title: str
    relevance: str = Field(description="One or two sentences on why this document is relevant")
    evidence: str = Field(description="Exact sentence copied from the source text")
    chart: list[ChartRow] = Field(default_factory=list)


class PriorArtReport(BaseModel):
    summary: str = Field(description="One or two sentences restating the invention idea")
    items: list[PriorArtItem]
    limitations: str = ""
    features: list[str] = Field(default_factory=list)
    removed_rows: int = 0


class FeatureDraft(BaseModel):
    """Step 1 of the claim chart: the idea's technical features, restated from the idea."""

    features: list[str]


class JudgeVerdict(BaseModel):
    verdict: Literal["yes", "no", "unsupported"]


def features_json_schema(min_features: int, max_features: int) -> dict[str, Any]:
    schema = FeatureDraft.model_json_schema()
    schema["properties"]["features"].update(
        {"minItems": min_features, "maxItems": max_features, "items": {"type": "string"}}
    )
    return schema
