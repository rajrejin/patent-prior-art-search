"""Split a report into atomic claims (rule-based, sentence level)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from patentrag.llm.schemas import PriorArtReport

_ABBREV = ("e.g.", "i.e.", "fig.", "figs.", "no.", "approx.", "vs.", "et al.", "cf.", "etc.")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


@dataclass(frozen=True)
class AtomicClaim:
    text: str
    source: str  # "query" for the summary, otherwise the cited pub_id
    premise: str = ""  # text to judge against; empty means the whole source (or the query)


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    pieces = _SPLIT.split(text)
    out: list[str] = []
    for piece in pieces:
        if out and out[-1].lower().endswith(_ABBREV):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return [s.strip() for s in out if len(s.strip()) > 3]


def atomic_claims(report: PriorArtReport) -> list[AtomicClaim]:
    """The statements a report makes, each with the text it must be supported by.

    Claim-chart reports: every feature is checked against the query, and every chart row
    ("this document discloses feature F") against the evidence sentence chosen for it.
    Free-text reports: summary sentences against the query, relevance sentences against the
    source. Evidence text itself is verified verbatim by the deterministic check.
    """
    if report.features:
        chart = [AtomicClaim(f, "query") for f in report.features]
        for it in report.items:
            chart += [AtomicClaim(r.feature, it.pub_id, r.evidence) for r in it.chart]
        return chart
    claims = [AtomicClaim(s, "query") for s in split_sentences(report.summary)]
    for it in report.items:
        claims += [AtomicClaim(s, it.pub_id) for s in split_sentences(it.relevance)]
    return claims
