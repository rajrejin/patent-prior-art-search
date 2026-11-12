"""Robustness study: query form, language, and corpus size.

Corpus-size subsampling is done at query time: documents outside the sampled fraction are added
to the exclusion set, so they vanish from results and from graph traversal (no data is deleted),
and the relevant set is intersected with the kept documents.
"""

from __future__ import annotations

import json
import random
from typing import Any

from patentrag.config import ROOT, Config
from patentrag.evaluation import runner
from patentrag.evaluation.harness import ensure_testset, subsample
from patentrag.evaluation.testset import Query, build_indexes, make_query
from patentrag.index.chunking import independent_claims
from patentrag.ingest.corpus import load_corpus
from patentrag.retrieval.base import Exclusion, Retriever

RESULTS = ROOT / "results"
METRICS = ["ndcg@10", "r@10", "mrr"]


def _score(
    cfg: Config, r: Retriever, queries: list[Query], key_of: dict[str, str], label: str
) -> dict[str, Any]:
    run = runner.run_method(r, queries, key_of, label=label)
    per_q = runner.score_run(run, queries, cfg.evaluation.ks)
    summ = runner.summarise(per_q, cfg.evaluation.bootstrap_resamples, cfg.seed)
    return {"summary": {m: summ[m] for m in METRICS}, "per_query": per_q}


def _paired(cfg: Config, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return runner.compare(
        a["per_query"],
        b["per_query"],
        ["ndcg@10", "r@10"],
        cfg.evaluation.bootstrap_resamples,
        cfg.seed,
    )


def _slim(res: dict[str, Any]) -> dict[str, Any]:
    return {"summary": res["summary"]}


PAIRS = [("hybrid", "bm25"), ("hybrid", "vector"), ("rerank", "hybrid")]


def _pairs(cfg: Config, scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Paired differences for the method pairs that are present."""
    return {
        f"{a}-{b}": _paired(cfg, scores[a], scores[b])
        for a, b in PAIRS
        if a in scores and b in scores
    }


def run_robustness(cfg: Config, scope: str, retrievers: dict[str, Retriever]) -> dict[str, Any]:
    patents = load_corpus(cfg, scope)
    by_id = {str(p.pub_id): p for p in patents}
    idx = build_indexes(patents)
    _, test, _ = ensure_testset(cfg, scope, patents)
    test = subsample(test, cfg.seed)
    out: dict[str, Any] = {"n_test_queries": len(test)}

    # 1. query form -----------------------------------------------------------------------------
    with_claims = [q for q in test if by_id[q.pub_id].claims and by_id[q.pub_id].abstract]
    forms: dict[str, list[Query]] = {
        "title_only": [],
        "abstract_only": [],
        "abstract_plus_claim": [],
    }
    for q in with_claims:
        p = by_id[q.pub_id]
        indep = independent_claims(p.claims or "")
        if not indep:
            continue
        texts = {
            "title_only": p.title,
            "abstract_only": p.abstract,
            "abstract_plus_claim": f"{p.abstract} {indep[0][1]}",
        }
        for name, text in texts.items():
            forms[name].append(make_query(p, idx, query_text=text))
    out["query_form"] = {"n_queries": len(forms["title_only"]), "results": {}}
    form_scores: dict[str, dict[str, Any]] = {}
    for name, qs in forms.items():
        if not qs:
            continue
        for mname, r in retrievers.items():
            res = _score(cfg, r, qs, idx.key_of, f"{name}/{mname}")
            form_scores[f"{name}/{mname}"] = res
            out["query_form"]["results"][f"{name}/{mname}"] = _slim(res)
    out["query_form"]["paired"] = {
        name: _pairs(
            cfg,
            {m: form_scores[f"{name}/{m}"] for m in retrievers if f"{name}/{m}" in form_scores},
        )
        for name in forms
    }

    # 2. language: German title against English title ---------------------------------------------
    lang_qs: dict[str, list[Query]] = {"title_en": [], "title_de": []}
    for q in test:
        p = by_id[q.pub_id]
        if p.titles.get("de") and p.titles.get("en"):
            lang_qs["title_en"].append(make_query(p, idx, query_text=p.titles["en"]))
            lang_qs["title_de"].append(make_query(p, idx, query_text=p.titles["de"]))
    out["language"] = {"n_queries": len(lang_qs["title_en"]), "results": {}}
    for name, qs in lang_qs.items():
        if not qs:
            continue
        for mname, r in retrievers.items():
            out["language"]["results"][f"{name}/{mname}"] = _slim(
                _score(cfg, r, qs, idx.key_of, f"{name}/{mname}")
            )

    # 3. corpus size -------------------------------------------------------------------------------
    all_ids = sorted(by_id)
    rng = random.Random(cfg.seed)  # noqa: S311
    order = all_ids[:]
    rng.shuffle(order)
    out["corpus_size"] = {}
    for frac in (0.25, 0.5, 1.0):
        keep = set(order[: int(len(order) * frac)])
        keep_keys = {idx.key_of[i] for i in keep}
        qs = []
        for q in test:
            rel = [k for k in q.relevant_keys if k in keep_keys]
            if q.pub_id in keep and rel:
                qq = Query(**{**q.__dict__, "relevant_keys": rel})
                qq.exclude = sorted(set(q.exclude) | (set(all_ids) - keep))
                qs.append(qq)
        res_f = {}
        scores: dict[str, dict[str, Any]] = {}
        for mname, r in retrievers.items():
            s = _score(cfg, r, qs, idx.key_of, f"size{frac}/{mname}")
            scores[mname] = s
            res_f[mname] = _slim(s)
        out["corpus_size"][str(frac)] = {
            "n_documents": len(keep),
            "n_queries": len(qs),
            "results": res_f,
            "paired": _pairs(cfg, scores),
        }
    (RESULTS / f"robustness_{scope}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


__all__ = ["run_robustness", "Exclusion"]
