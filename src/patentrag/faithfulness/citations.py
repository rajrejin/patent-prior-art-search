"""Deterministic citation checks: named patents must be retrieved; evidence must be verbatim."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patentrag.llm.schemas import PriorArtReport

_WS = re.compile(r"\s+")
_NUMBER = re.compile(
    r"\b(EP|WO|US|JP|CN|KR|DE|FR|GB|CA|AU|TW|RU|CH|AT|ES|IT|NL|SE)\s*[-:]?\s*"
    r"(\d[\d\s,./]{3,}\d)(?:\s*([A-Z]\d?))?",
)
_PUB = re.compile(r"^([A-Z]{2})(\d+)")


def normalise_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def doc_key(pub_id: str) -> str:
    """EP3667784A1 -> EP3667784 (country plus digits; the kind code is dropped)."""
    m = _PUB.match(pub_id.replace(" ", ""))
    return f"{m.group(1)}{m.group(2)}" if m else pub_id


def extract_doc_keys(text: str) -> set[str]:
    """Every patent number mentioned in free text, as country plus digits."""
    out = set()
    for m in _NUMBER.finditer(text):
        digits = re.sub(r"\D", "", m.group(2))
        if len(digits) >= 5:
            out.add(f"{m.group(1)}{digits}")
    return out


@dataclass
class CitationCheck:
    named: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    valid_citation_fraction: float = 1.0
    evidence_total: int = 0
    evidence_valid: int = 0
    evidence_valid_fraction: float = 1.0
    bad_evidence: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.invalid and not self.bad_evidence


def report_text(report: PriorArtReport) -> str:
    parts = [report.summary, report.limitations]
    for it in report.items:
        parts += [it.pub_id, it.title, it.relevance, it.evidence]
        parts += [r.evidence for r in it.chart]
    return "\n".join(parts)


def check_citations(report: PriorArtReport, retrieved_text: dict[str, str]) -> CitationCheck:
    """`retrieved_text` maps pub_id to the exact text that was shown to the model."""
    allowed = {doc_key(p) for p in retrieved_text}
    named = extract_doc_keys(report_text(report))
    named |= {doc_key(it.pub_id) for it in report.items}
    invalid = sorted(k for k in named if k not in allowed)
    res = CitationCheck(named=sorted(named), invalid=invalid)
    res.valid_citation_fraction = 1.0 if not named else (len(named) - len(invalid)) / len(named)
    by_key = {doc_key(p): normalise_ws(t) for p, t in retrieved_text.items()}
    for it in report.items:
        src = by_key.get(doc_key(it.pub_id), "")
        quotes = [r.evidence for r in it.chart] if it.chart else [it.evidence]
        for quote in quotes:
            res.evidence_total += 1
            ev = normalise_ws(quote).strip("\"'`“”")
            if ev and ev in src:
                res.evidence_valid += 1
            elif it.pub_id not in res.bad_evidence:
                res.bad_evidence.append(it.pub_id)
    if res.evidence_total:
        res.evidence_valid_fraction = res.evidence_valid / res.evidence_total
    return res
