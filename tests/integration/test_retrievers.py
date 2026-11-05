"""Retriever sanity checks on the built full-corpus indexes (skipped when they are absent).

A patent's own title and abstract, used as the query, must retrieve that patent first.
"""

from __future__ import annotations

import pytest

from patentrag.config import load_config
from patentrag.index.build import index_dirs
from patentrag.ingest.corpus import corpus_paths
from patentrag.retrieval.base import Exclusion

SCOPE = "full"


@pytest.fixture(scope="module")
def stack():
    cfg = load_config()
    chroma, bm25, _ = index_dirs(cfg, SCOPE)
    if not corpus_paths(cfg, SCOPE)[0].exists() or not chroma.exists() or not bm25.exists():
        pytest.skip("full corpus or indexes not built (run `patentrag build-index`)")
    from patentrag.stack import Stack

    return Stack(cfg, SCOPE)


@pytest.fixture(scope="module")
def seeds(stack):
    ps = [p for p in stack.patents.values() if p.abstract and p.origin == "seed"]
    return ps[:: max(1, len(ps) // 25)][:25]


@pytest.mark.parametrize("method", ["bm25", "vector", "hybrid"])
def test_own_abstract_retrieves_itself_first(stack, seeds, method):
    r = {"bm25": stack.bm25, "vector": lambda: stack.vector, "hybrid": stack.hybrid}[method]()
    hits = [r.search(p.text_for_query(), 1)[0].pub_id == str(p.pub_id) for p in seeds]
    assert sum(hits) / len(hits) >= 0.95


def test_exclusion_removes_self_from_hybrid_results(stack, seeds):
    p = seeds[0]
    pid = str(p.pub_id)
    res = stack.hybrid().search(p.text_for_query(), 10, Exclusion(exclude=frozenset({pid})))
    assert pid not in {h.pub_id for h in res} and len(res) == 10
