"""Metric arithmetic against hand-computed answers (tiny inputs, not datasets)."""

from __future__ import annotations

import math

import pytest

from patentrag.evaluation.metrics import (
    bootstrap_ci,
    hit_rate_at_k,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    query_metrics,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]
REL = {"b", "e", "z"}  # z is never retrieved


def test_precision_recall_hit():
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, REL, 2) == pytest.approx(1 / 2)
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, REL, 1) == 0.0
    assert hit_rate_at_k(RANKED, REL, 1) == 0.0
    assert hit_rate_at_k(RANKED, REL, 2) == 1.0


def test_mrr():
    assert reciprocal_rank(RANKED, REL) == pytest.approx(1 / 2)
    assert reciprocal_rank(["x", "y"], REL) == 0.0


def test_ndcg_hand_computed():
    # relevant at ranks 2 and 5. DCG = 1/log2(3) + 1/log2(6).
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    # ideal with 3 relevant docs in the top 5: ranks 1, 2, 3.
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["b"], {"b"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {"b"}, 10) == 0.0


def test_empty_relevant_set_is_zero_not_error():
    assert recall_at_k(RANKED, set(), 5) == 0.0
    assert ndcg_at_k(RANKED, set(), 5) == 0.0


def test_query_metrics_keys():
    m = query_metrics(RANKED, REL, [5, 10])
    assert {"mrr", "ndcg@10", "p@5", "r@10", "hit@5"} <= set(m)
    assert m["p@10"] == pytest.approx(2 / 10)  # precision divides by k even with fewer results


def test_bootstrap_ci_contains_mean_and_is_seeded():
    vals = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    mean, lo, hi = bootstrap_ci(vals, 1000, seed=0)
    assert mean == pytest.approx(5 / 8)
    assert lo <= mean <= hi
    assert bootstrap_ci(vals, 1000, seed=0) == (mean, lo, hi)


def test_bootstrap_constant_values_have_zero_width():
    mean, lo, hi = bootstrap_ci([0.5] * 10, 200, seed=0)
    assert mean == lo == hi == 0.5


def test_paired_bootstrap_detects_consistent_gain():
    a = [0.6, 0.7, 0.8, 0.9, 0.7, 0.8]
    b = [0.5, 0.6, 0.7, 0.8, 0.6, 0.7]
    r = paired_bootstrap(a, b, 1000, seed=0)
    assert r["diff"] == pytest.approx(0.1)
    assert r["low"] == pytest.approx(0.1) and r["high"] == pytest.approx(0.1)
    assert r["p"] <= 0.002


def test_paired_bootstrap_symmetric_noise_not_significant():
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    r = paired_bootstrap(a, b, 1000, seed=0)
    assert r["diff"] == pytest.approx(0.0)
    assert r["low"] < 0 < r["high"]
