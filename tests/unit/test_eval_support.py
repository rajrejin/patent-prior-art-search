"""Small tests for evaluation helpers (hand-written inputs, hand-computed answers)."""

from __future__ import annotations

from patentrag.evaluation.faith_eval import summarise
from patentrag.evaluation.harness import subsample
from patentrag.evaluation.testset import Query


def _q(i: int) -> Query:
    return Query(f"Q{i}", f"Q{i}", "t", ["k"], [], 2020, "H01M")


def test_subsample_is_seeded_and_unchanged_when_unset(monkeypatch):
    qs = [_q(i) for i in range(20)]
    monkeypatch.delenv("PATENTRAG_MAX_QUERIES", raising=False)
    assert subsample(qs) == qs
    monkeypatch.setenv("PATENTRAG_MAX_QUERIES", "5")
    a, b = subsample(qs, 0), subsample(list(reversed(qs)), 0)
    assert len(a) == 5 and [q.qid for q in a] == [q.qid for q in b]
    assert [q.qid for q in a] == sorted(q.qid for q in a)


def _att(n, claims, evidence=1.0):
    return {
        "n": n,
        "seconds": 1.0,
        "ok": False,
        "error": "",
        "valid_citation_fraction": 1.0,
        "evidence_valid_fraction": evidence,
        "invalid_numbers": [],
        "claims": claims,
    }


def _c(source, text, llm, nli, premise="s"):
    return {
        "source": source,
        "text": text,
        "premise": premise,
        "llm": llm,
        "nli": nli,
        "supported": llm and nli >= 0.5,
    }


def test_summarise_hand_computed():
    first = [
        _c("query", "f1", True, 0.9, ""),
        _c("A", "f1", True, 0.9),
        _c("A", "f2", False, 0.9),
        _c("B", "f1", True, 0.1),
        _c("B", "f2", False, 0.1),
    ]
    retry = [_c("A", "f2", True, 0.8, "s2"), _c("B", "f1", False, 0.1, "s2")]
    report = {"items": [{"chart": [{}, {}]}], "removed_rows": 2, "features": ["f1"]}
    rec = {"qid": "x", "attempts": [_att(0, first, 0.5), _att(1, retry)], "report": report}
    out = summarise([rec, {"qid": "y", "attempts": [], "report": None}], threshold=0.5)
    b = out["first_choice"]
    assert b["n_claims"] == 4  # rows only; the feature is reported separately
    assert b["support_rate_llm"] == 0.5 and b["support_rate_nli"] == 0.5
    assert b["support_rate_combined"] == 0.25
    # llm vs nli agree on A/f1 and B/f2 only: agreement 0.5; kappa = 0 (po 0.5, pe 0.5)
    assert b["agreement_llm_vs_nli"] == 0.5 and b["kappa_llm_vs_nli"] == 0.0
    assert b["evidence_from_cited_patent"] == 0.5
    a = out["after_retries"]  # each pair's last judgement: A/f2 now supported
    assert a["n_claims"] == 4 and a["support_rate_combined"] == 0.5
    assert out["features"]["n_claims"] == 1 and out["features"]["support_rate_combined"] == 1.0
    sh = out["shipped"]
    assert sh["reports_with_verified_rows"] == 0.5 and sh["mean_verified_rows"] == 1.0
    assert sh["pair_fill_rate"] == 0.5 and sh["feature_extraction_failed"] == 0.5
    assert out["mean_attempts"] == 1.0
