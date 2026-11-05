"""API request models. Response models live in patentrag.schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from patentrag.schemas import Method


class SearchRequest(BaseModel):
    query: str = Field(min_length=10, max_length=4000, description="Invention description")
    k: int = Field(default=10, ge=1, le=50)
    method: Method = "auto"


class ReportRequest(BaseModel):
    query: str = Field(min_length=20, max_length=4000, description="Invention description")
