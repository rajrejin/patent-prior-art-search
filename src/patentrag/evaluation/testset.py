"""Test-set construction with leakage controls (Section 7.1).

Query = title + abstract of a corpus patent. Relevant = its cited documents that are inside the
corpus, with the query's own document key and its family removed. Relevance is defined at
document-key level (country + number): A1 and B1 of the same EP number are one document.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patentrag.models import Patent
from patentrag.retrieval.base import Exclusion


@dataclass
class Query:
    qid: str
    pub_id: str
    text: str
    relevant_keys: list[str]
    exclude: list[str]  # pub_ids: the query patent, same-document kinds, family members
    year: int
    subclass: str
    split: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def exclusion(self, hide_outgoing: bool = True) -> Exclusion:
        ex = frozenset(self.exclude)
        if hide_outgoing:
            return Exclusion(exclude=ex, hide_outgoing_from=ex)
        return Exclusion(exclude=ex, final_only=True)


@dataclass
class CorpusIndex:
    by_key: dict[str, list[str]]
    by_family: dict[str, list[str]]
    key_of: dict[str, str]


def build_indexes(patents: list[Patent]) -> CorpusIndex:
    by_key: dict[str, list[str]] = defaultdict(list)
    by_family: dict[str, list[str]] = defaultdict(list)
    key_of: dict[str, str] = {}
    for p in patents:
        pid = str(p.pub_id)
        key_of[pid] = p.pub_id.doc_key
        by_key[p.pub_id.doc_key].append(pid)
        if p.family_id:
            by_family[p.family_id].append(pid)
    return CorpusIndex(dict(by_key), dict(by_family), key_of)


def make_query(p: Patent, idx: CorpusIndex, query_text: str | None = None) -> Query:
    """Build one query; applies leakage controls (1), (3) and (4) here.

    (1) the query patent (all its kinds) is excluded; (3) family members are excluded from the
    results and from the relevant set; (4) the text is title + abstract only, no numbers.
    Control (2), hiding the query patent's outgoing CITES edges, is applied at query time through
    `Query.exclusion()`.
    """
    own = set(idx.by_key[p.pub_id.doc_key]) | set(idx.by_family.get(p.family_id or "", []))
    own_keys = {idx.key_of[x] for x in own}
    relevant = sorted(
        {c.doc_key for c in p.cites if c.doc_key in idx.by_key and c.doc_key not in own_keys}
    )
    text = query_text if query_text is not None else p.text_for_query()
    return Query(
        qid=str(p.pub_id),
        pub_id=str(p.pub_id),
        text=text,
        relevant_keys=relevant,
        exclude=sorted(own),
        year=p.pub_date.year if p.pub_date else 0,
        subclass=next((c[:4] for c in p.cpc), ""),
    )


def eligible_queries(patents: list[Patent], min_relevant: int = 3) -> list[Query]:
    idx = build_indexes(patents)
    out = []
    for p in patents:
        if p.origin != "seed" or not p.abstract:
            continue  # queries come from the sampled seed patents only
        q = make_query(p, idx)
        if len(q.relevant_keys) >= min_relevant:
            out.append(q)
    return sorted(out, key=lambda q: q.qid)


def stratified_split(
    queries: list[Query], n_test: int, n_val: int, seed: int = 0
) -> tuple[list[Query], list[Query]]:
    """Seeded split stratified by (year, subclass). Returns (validation, test)."""
    rng = random.Random(seed)  # noqa: S311
    strata: dict[tuple[int, str], list[Query]] = defaultdict(list)
    for q in queries:
        strata[(q.year, q.subclass)].append(q)
    for key in sorted(strata):  # sorted so the shuffle sequence does not depend on input order
        strata[key].sort(key=lambda q: q.qid)
        rng.shuffle(strata[key])
    # round-robin across strata so every stratum contributes to both splits
    order: list[Query] = []
    keys = sorted(strata)
    while any(strata[k] for k in keys):
        for k in keys:
            if strata[k]:
                order.append(strata[k].pop())
    total = len(order)
    n_test = min(n_test, total * 3 // 4)
    n_val = min(n_val, total - n_test)
    # interleave: alternate so both splits see all strata, then cut to sizes
    test = [q for i, q in enumerate(order) if i % 4 != 0][:n_test]
    chosen = {q.qid for q in test}
    val = [q for q in order if q.qid not in chosen][:n_val]
    for q in test:
        q.split = "test"
    for q in val:
        q.split = "val"
    return sorted(val, key=lambda q: q.qid), sorted(test, key=lambda q: q.qid)


def save_testset(path: Path, val: list[Query], test: list[Query], meta: dict[str, Any]) -> None:
    body: dict[str, Any] = {
        "meta": meta,
        "val": [q.__dict__ for q in val],
        "test": [q.__dict__ for q in test],
    }
    blob = json.dumps(body, indent=1, sort_keys=True)
    body["meta"]["sha256"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")


def load_testset(path: Path) -> tuple[list[Query], list[Query], dict[str, Any]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    return (
        [Query(**q) for q in body["val"]],
        [Query(**q) for q in body["test"]],
        body["meta"],
    )
