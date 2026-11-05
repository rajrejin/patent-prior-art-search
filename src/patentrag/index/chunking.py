"""Chunking: title+abstract, claims (independent first), optional description windows."""

from __future__ import annotations

import re
from dataclasses import dataclass

from patentrag.models import Patent

_DEPENDENT = re.compile(
    r"\b(according to|as claimed in|as defined in|as recited in|as set forth in|of|in|by|with"
    r"|as in|as described in|as per)\s+(any\s+(one\s+)?of\s+)?(the\s+)?(preceding\s+)?"
    r"claims?\s*\d*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    pub_id: str
    section: str  # abstract | claims | description
    text: str


def split_claims(claims: str) -> list[tuple[int, str]]:
    """Split a claims block into (number, text).

    Claims are found as the increasing sequence 1, 2, 3, ... so that numbers inside a claim
    ("0.5 mm", "claim 1.") are not mistaken for claim starts. Text before claim 1 (such as a
    "CLAIMS" heading) is dropped. If no sequence is found the whole block is one claim.
    """
    text = " ".join(claims.split())
    starts: list[tuple[int, int, int]] = []  # (number, start of match, start of claim text)
    pos = 0
    n = 1
    while True:
        m = re.compile(rf"(?<![\w.,/-]){n}\s*[.)]\s+(?=\S)").search(text, pos)
        if not m:
            break
        starts.append((n, m.start(), m.end()))
        pos = m.end()
        n += 1
    if not starts:
        return [(1, text)] if text else []
    out = []
    for i, (num, _, body_start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        body = text[body_start:end].strip()
        if body:
            out.append((num, body))
    return out


def is_dependent(text: str) -> bool:
    return bool(_DEPENDENT.search(text[:300]))


def independent_claims(claims: str) -> list[tuple[int, str]]:
    return [(n, t) for n, t in split_claims(claims) if not is_dependent(t)]


def window_chunks(text: str, size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size_words:
        return [" ".join(words)]
    step = max(1, size_words - overlap_words)
    out = []
    for i in range(0, len(words), step):
        out.append(" ".join(words[i : i + size_words]))
        if i + size_words >= len(words):
            break
    return out


def chunk_patent(
    p: Patent,
    *,
    use_claims: bool = True,
    max_claim_chunks: int = 4,
    claim_words: int = 170,
    desc_words: int = 170,
    desc_overlap: int = 30,
    max_desc_chunks: int = 20,
) -> list[Chunk]:
    pid = str(p.pub_id)
    chunks: list[Chunk] = []
    head = p.text_for_query()
    if head.strip(". "):
        chunks.append(Chunk(f"{pid}::abstract::0", pid, "abstract", head))
    if use_claims and p.claims:
        indep = independent_claims(p.claims) or split_claims(p.claims)[:1]
        pieces: list[str] = []
        for _, t in indep:
            pieces.extend(window_chunks(t, claim_words, 0))
        for i, t in enumerate(pieces[:max_claim_chunks]):
            chunks.append(Chunk(f"{pid}::claims::{i}", pid, "claims", t))
    if p.description:
        for i, t in enumerate(window_chunks(p.description, desc_words, desc_overlap)):
            if i >= max_desc_chunks:
                break
            chunks.append(Chunk(f"{pid}::description::{i}", pid, "description", t))
    return chunks


def patent_text(p: Patent, use_claims: bool = True) -> str:
    """Whole-document text for the lexical baseline: title + abstract (+ claims)."""
    parts = [p.title, p.abstract]
    if use_claims and p.claims:
        parts.append(p.claims)
    return " ".join(x for x in parts if x)


def source_text(p: Patent, max_words: int = 190) -> str:
    """The retrieved text for one patent: title, abstract and the first independent claim."""
    parts = [p.title.strip().rstrip(".") + "."]
    if p.abstract:
        parts.append(p.abstract)
    if p.claims:
        indep = independent_claims(p.claims)
        if indep:
            parts.append("Claim " + str(indep[0][0]) + ": " + indep[0][1])
    words = " ".join(parts).split()
    return " ".join(words[:max_words])
