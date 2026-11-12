"""Normalised patent schema."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class PubId(BaseModel):
    """Publication identifier: country, number, kind (for example EP, 3456789, A1)."""

    model_config = {"frozen": True}

    country: str
    number: str
    kind: str = ""

    @property
    def doc_key(self) -> str:
        """Kind-independent key, e.g. EP3456789 (used to join citations)."""
        return f"{self.country}{self.number}"

    def __str__(self) -> str:
        return f"{self.country}{self.number}{self.kind}"


class CitationInfo(BaseModel):
    """One cited document with the metadata OPS exposes (category X/Y/A, phase, who cited)."""

    pub_id: PubId
    category: str = ""
    phase: str = ""
    cited_by: str = ""
    pub_date: date | None = None


class Patent(BaseModel):
    pub_id: PubId
    title: str = ""
    titles: dict[str, str] = Field(default_factory=dict)
    abstract: str = ""
    claims: str | None = None
    description: str | None = None
    language: str = ""
    applicants: list[str] = Field(default_factory=list)
    inventors: list[str] = Field(default_factory=list)
    filing_date: date | None = None
    pub_date: date | None = None
    cpc: list[str] = Field(default_factory=list)
    ipc: list[str] = Field(default_factory=list)
    cites: list[PubId] = Field(default_factory=list)
    citation_details: list[CitationInfo] = Field(default_factory=list)
    family_id: str | None = None
    family_members: list[PubId] = Field(default_factory=list)
    fields_present: list[str] = Field(default_factory=list)
    origin: str = "seed"  # seed: sampled from the technology area; cited: added by citation closure

    def text_for_query(self) -> str:
        return f"{self.title}. {self.abstract}".strip()
