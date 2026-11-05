"""Chunking, aggregation, voting and re-rank arithmetic, plus leakage controls on real data."""

from __future__ import annotations

import pytest

from patentrag.config import load_config
from patentrag.evaluation.runner import collapse_to_documents
from patentrag.evaluation.testset import build_indexes, eligible_queries, stratified_split
from patentrag.index.bm25 import Bm25Index, tokenize
from patentrag.index.chunking import (
    chunk_patent,
    is_dependent,
    split_claims,
    window_chunks,
)
from patentrag.ingest.corpus import corpus_paths, load_corpus
from patentrag.retrieval.classify import vote_cpc
from patentrag.retrieval.rerank import Weights, citation_proximity, cpc_overlap, final_score
from patentrag.retrieval.vector import aggregate_chunks


def _corpus():
    cfg = load_config()
    if not corpus_paths(cfg, "full")[0].exists():
        pytest.skip("full corpus not built (run `patentrag build-corpus`)")
    return load_corpus(cfg, "full")


# --- pure logic --------------------------------------------------------------------------


def test_window_chunks_overlap():
    words = " ".join(str(i) for i in range(10))
    assert window_chunks(words, 4, 1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert window_chunks("a b", 4, 1) == ["a b"]
    assert window_chunks("", 4, 1) == []


def test_split_claims_and_dependence():
    block = (
        "CLAIMS 1. A cell comprising X. 2. The cell according to claim 1, wherein Y is 0.5 mm.\n"
        "and Z.\n3. A pack."
    )
    claims = split_claims(block)
    assert [n for n, _ in claims] == [1, 2, 3]
    assert "and Z." in claims[1][1]  # unnumbered text extends the previous claim
    assert "0.5 mm" in claims[1][1]  # a number inside a claim is not a claim start
    assert not is_dependent(claims[0][1])
    assert is_dependent(claims[1][1])
    assert not is_dependent(claims[2][1])


def test_aggregate_max_vs_top3_sum():
    hits = [
        {"pub_id": "A", "sim": 0.9, "text": "a1", "section": "abstract"},
        {"pub_id": "A", "sim": 0.5, "text": "a2", "section": "claims"},
        {"pub_id": "B", "sim": 0.8, "text": "b1", "section": "abstract"},
        {"pub_id": "B", "sim": 0.7, "text": "b2", "section": "claims"},
        {"pub_id": "B", "sim": 0.6, "text": "b3", "section": "claims"},
        {"pub_id": "B", "sim": 0.1, "text": "b4", "section": "claims"},
    ]
    mx = aggregate_chunks(hits, "max")
    assert mx["A"]["score"] == 0.9 and mx["B"]["score"] == 0.8
    s3 = aggregate_chunks(hits, "top3sum")
    assert s3["A"]["score"] == pytest.approx(1.4)
    assert s3["B"]["score"] == pytest.approx(2.1)  # 0.8 + 0.7 + 0.6, the 0.1 chunk is ignored
    assert s3["B"]["best_chunk"] == "b1"
    only_claims = aggregate_chunks(hits, "max", sections={"claims"})
    assert only_claims["A"]["score"] == 0.5


def test_vote_cpc_weighted_hand_computed():
    cpcs = {
        "P1": ["H01M10/0525", "H01M4/134"],
        "P2": ["H01M10/42"],
        "P3": ["B60L58/12"],
    }
    seeds = [("P1", 0.9), ("P2", 0.6), ("P3", 0.3)]
    # level=group: P1 votes H01M10 and H01M4 (0.9 each); P2 votes H01M10 (0.6); P3 votes B60L58.
    ranked = vote_cpc(seeds, cpcs, "group", 3)
    assert ranked[0] == ("H01M10", pytest.approx(1.5))
    assert ranked[1] == ("H01M4", pytest.approx(0.9))
    assert ranked[2] == ("B60L58", pytest.approx(0.3))
    sub = vote_cpc(seeds, cpcs, "subclass", 1)
    assert sub == [("H01M", pytest.approx(1.5))]  # P1 counts once for H01M, not twice


def test_cpc_overlap_and_final_score_arithmetic():
    votes = {"H01M10": 3.0, "B60L58": 1.0}
    assert cpc_overlap(["H01M10/0525"], votes, "group") == pytest.approx(0.75)
    assert cpc_overlap(["G01R31/36"], votes, "group") == 0.0
    assert cpc_overlap(["H01M10/42", "B60L58/12"], votes, "group") == pytest.approx(1.0)
    assert citation_proximity(None) == 0.0 and citation_proximity(2) == 0.5
    w = Weights(sim=1.0, cpc=0.2, cite=0.2, family=0.1, applicant=0.05)
    assert final_score(0.8, 0.5, 1.0, True, False, w) == pytest.approx(0.8 + 0.1 + 0.2 + 0.1)


def test_bm25_ranks_term_overlap_first():
    idx = Bm25Index(["d1", "d2", "d3"], ["lithium anode coating", "vehicle charger plug", "anode"])
    top = idx.search("anode coating", 3)
    assert top[0][0] == "d1"
    assert tokenize("The Anode, of claim 1") == ["anode"]


def test_collapse_to_documents_keeps_first_kind():
    key_of = {"EP1A1": "EP1", "EP1B1": "EP1", "EP2A1": "EP2"}
    assert collapse_to_documents(["EP1B1", "EP2A1", "EP1A1"], key_of) == ["EP1", "EP2"]


# --- real-data tests -----------------------------------------------------------------------


def test_chunking_real_claims():
    patents = [p for p in _corpus() if p.claims]
    if not patents:
        pytest.skip("no patents with claims in the corpus")
    for p in patents[:50]:
        chunks = chunk_patent(p)
        secs = [c.section for c in chunks]
        assert secs.count("abstract") <= 1
        assert 1 <= secs.count("claims") <= 4 or not p.claims
        assert all(c.pub_id == str(p.pub_id) and c.text for c in chunks)
        claim_chunks = [c for c in chunks if c.section == "claims"]
        assert all(len(c.text.split()) <= 170 for c in claim_chunks)
        assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_leakage_controls_on_real_queries():
    patents = _corpus()
    queries = eligible_queries(patents, 1)
    if not queries:
        pytest.skip("no query patent has an in-corpus citation in this corpus")
    idx = build_indexes(patents)
    by_id = {str(p.pub_id): p for p in patents}
    for q in queries:
        p = by_id[q.pub_id]
        # (1) the query patent itself is excluded
        assert q.pub_id in q.exclude
        # (3) family members are excluded and are never relevant
        fam = set(idx.by_family.get(p.family_id or "", []))
        assert fam <= set(q.exclude)
        fam_keys = {idx.key_of[x] for x in fam}
        assert not (fam_keys & set(q.relevant_keys))
        assert p.pub_id.doc_key not in q.relevant_keys
        # relevance only covers documents inside the corpus
        assert all(k in idx.by_key for k in q.relevant_keys)
        # (4) the query text carries no citation list or publication numbers
        for c in p.cites:
            assert c.number not in q.text
        assert str(p.pub_id) not in q.text and p.pub_id.number not in q.text
        # (2) outgoing edges of every excluded document are hidden at query time
        ex = q.exclusion()
        assert ex.hide_outgoing_from == ex.exclude
        assert q.exclusion(hide_outgoing=False).hide_outgoing_from == frozenset()


def test_split_is_disjoint_deterministic_and_stratified():
    patents = _corpus()
    queries = eligible_queries(patents, 1)
    if len(queries) < 8:
        pytest.skip("too few eligible queries in the corpus to test the split")
    a = stratified_split(queries, 10, 5, seed=0)
    b = stratified_split(list(reversed(queries)), 10, 5, seed=0)
    assert [q.qid for q in a[0]] == [q.qid for q in b[0]]
    assert [q.qid for q in a[1]] == [q.qid for q in b[1]]
    assert not ({q.qid for q in a[0]} & {q.qid for q in a[1]})
