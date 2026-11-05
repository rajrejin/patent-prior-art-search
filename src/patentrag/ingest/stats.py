"""Corpus validation report (results/corpus_stats*.json)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from patentrag.models import Patent


def corpus_stats(patents: list[Patent], min_relevant: int = 3) -> dict[str, Any]:
    keys = {p.pub_id.doc_key for p in patents}
    fam_keys: dict[str, set[str]] = {}
    for p in patents:
        if p.family_id:
            fam_keys.setdefault(p.family_id, set()).add(p.pub_id.doc_key)

    n = len(patents)
    missing = {
        f: sum(1 for p in patents if f not in p.fields_present)
        for f in ("title", "abstract", "applicants", "cpc", "cites", "claims", "family")
    }
    total_cites = in_corpus = 0
    eligible = 0
    for p in patents:
        if p.origin != "seed":
            continue
        own_family = fam_keys.get(p.family_id or "", set())
        inside = {c.doc_key for c in p.cites if c.doc_key in keys and c.doc_key != p.pub_id.doc_key}
        inside -= own_family
        total_cites += len({c.doc_key for c in p.cites})
        in_corpus += len({c.doc_key for c in p.cites} & keys)
        if len(inside) >= min_relevant:
            eligible += 1
    subclass = Counter(c[:4] for p in patents for c in {x[:4] for x in p.cpc})
    applicants = Counter(a for p in patents for a in set(p.applicants))
    years = Counter(str(p.pub_date.year) for p in patents if p.pub_date)
    fam_sizes = Counter(len(p.family_members) + 1 for p in patents if "family" in p.fields_present)
    return {
        "n_patents": n,
        "n_distinct_documents": len(keys),
        "by_origin": dict(Counter(p.origin for p in patents)),
        "missing_field_counts": missing,
        "languages": dict(Counter(p.language or "unknown" for p in patents)),
        "per_year": dict(sorted(years.items())),
        "top20_applicants": applicants.most_common(20),
        "per_cpc_subclass_top30": subclass.most_common(30),
        "citations": {
            "total_cited_documents": total_cites,
            "in_corpus": in_corpus,
            "coverage": (in_corpus / total_cites) if total_cites else 0.0,
            "dropped_outside_corpus": total_cites - in_corpus,
        },
        "family_size_distribution": dict(sorted(fam_sizes.items())),
        "patents_with_claims": n - missing["claims"],
        "eligible_query_patents": {"min_relevant": min_relevant, "count": eligible},
    }
