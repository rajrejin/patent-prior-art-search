"""Faithfulness logic: string checks with hand-written inputs, judges with injected functions."""

from __future__ import annotations

import numpy as np
import pytest

from patentrag.agent.memory import Memory, query_hash
from patentrag.agent.report import (
    build_chart,
    candidate_sentences,
    clean_features,
    embedding_ranker,
    generate_report,
    rank_by_overlap,
)
from patentrag.faithfulness.citations import check_citations, doc_key, extract_doc_keys
from patentrag.faithfulness.claims import atomic_claims, split_sentences
from patentrag.faithfulness.judge import check_support, cohens_kappa
from patentrag.faithfulness.nli import passages
from patentrag.index.chunking import source_text
from patentrag.llm.client import LLMResult, generate_structured
from patentrag.llm.schemas import (
    FeatureDraft,
    JudgeVerdict,
    PriorArtItem,
    PriorArtReport,
    features_json_schema,
)
from patentrag.models import Patent, PubId

SOURCES = {
    "EP3667784A1": "An anode comprising a substrate and a carbon coating layer between the "
    "substrate and an active material layer. The OI value is greater than 15.",
    "WO2020129478A1": "A battery state estimation device that estimates the state of charge.",
}


def make_report(**over) -> PriorArtReport:
    base = {
        "summary": "An anode with a carbon coating layer.",
        "items": [
            PriorArtItem(
                pub_id="EP3667784A1",
                title="Anode",
                relevance="It describes an anode with a carbon coating layer.",
                evidence="The OI value is greater than 15.",
            )
        ],
        "limitations": "Text similarity only.",
    }
    base.update(over)
    return PriorArtReport(**base)


def test_doc_key_and_number_extraction():
    assert doc_key("EP3667784A1") == "EP3667784"
    assert doc_key("WO 2020129478 A1") == "WO2020129478"
    found = extract_doc_keys("See EP 3667784, WO2020/129478 A1 and US 9,812,739 B2.")
    assert found == {"EP3667784", "WO2020129478", "US9812739"}
    assert extract_doc_keys("Claim 15 and 20 mm values 2020") == set()


def test_citation_check_passes_for_grounded_report():
    res = check_citations(make_report(), SOURCES)
    assert res.passed and res.valid_citation_fraction == 1.0 and res.evidence_valid_fraction == 1.0


def test_citation_check_flags_unretrieved_number_anywhere():
    r = make_report(summary="Related to US 9,812,739 and the anode idea.")
    res = check_citations(r, SOURCES)
    assert res.invalid == ["US9812739"]
    assert not res.passed
    assert res.valid_citation_fraction == pytest.approx(0.5)  # named: EP3667784, US9812739


def test_citation_check_flags_non_verbatim_evidence_but_tolerates_whitespace():
    ok = make_report(
        items=[
            PriorArtItem(
                pub_id="EP3667784A1",
                title="t",
                relevance="r",
                evidence="The  OI value is\ngreater than 15.",
            )
        ]
    )
    assert check_citations(ok, SOURCES).passed
    bad = make_report(
        items=[
            PriorArtItem(
                pub_id="EP3667784A1", title="t", relevance="r", evidence="The OI value is 15."
            )
        ]
    )
    res = check_citations(bad, SOURCES)
    assert res.bad_evidence == ["EP3667784A1"] and res.evidence_valid_fraction == 0.0


def test_split_sentences_handles_abbreviations_and_decimals():
    text = "It uses graphite, e.g. natural graphite. The ratio is 0.5 mm. Another sentence here."
    assert split_sentences(text) == [
        "It uses graphite, e.g. natural graphite.",
        "The ratio is 0.5 mm.",
        "Another sentence here.",
    ]


def test_atomic_claims_sources():
    claims = atomic_claims(make_report())
    assert [(c.source) for c in claims] == ["query", "EP3667784A1"]


def test_cohens_kappa_hand_computed():
    a = [True, True, False, False]
    assert cohens_kappa(a, a) == 1.0
    assert cohens_kappa(a, [not x for x in a]) == -1.0
    # po = 3/4, pa1 = 3/4, pb1 = 1/2 -> pe = 3/8 + 1/8 = 1/2 -> kappa = (3/4 - 1/2) / (1/2) = 0.5
    assert cohens_kappa([True, True, True, False], [True, True, False, False]) == pytest.approx(0.5)
    assert cohens_kappa([], []) == 0.0


def test_support_flags_if_either_judge_says_unsupported():
    r = make_report()
    res = check_support(
        r,
        SOURCES,
        "anode idea",
        llm_fn=lambda src, claim: True,
        nli_fn=lambda src, claim: (
            0.2 if "carbon coating layer" in claim and "describes" in claim else 0.9
        ),
        nli_threshold=0.5,
        flag_if_either=True,
    )
    assert res.n == 2
    assert [v.supported for v in res.verdicts] == [True, False]
    assert res.rate("llm") == 1.0 and res.rate("nli") == 0.5
    lenient = check_support(
        r,
        SOURCES,
        "q",
        llm_fn=lambda s, c: True,
        nli_fn=lambda s, c: 0.0,
        nli_threshold=0.5,
        flag_if_either=False,
    )
    assert all(v.supported for v in lenient.verdicts)


def test_passages_windows_cover_text():
    text = " ".join(str(i) for i in range(300))
    ws = passages(text, words=110, stride=80, limit=8)
    assert ws[0].startswith("0 ") and "299" in ws[-1]


class ScriptedLLM:
    """Returns pre-set JSON strings in order; used to test the report loop's control flow."""

    backend = "scripted"
    model = "scripted"

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = 0

    def available(self) -> bool:
        return True

    def complete(self, prompt, *, system=None, json_schema=None, max_tokens=700) -> LLMResult:
        self.calls += 1
        return LLMResult(self.outputs.pop(0), 0.0)


QUERY = "An anode with a carbon coating layer and an OI value above 15."
FEATURES = ["An anode has a carbon coating layer.", "The OI value is above 15."]


def features_json(features: list[str] = FEATURES) -> str:
    return FeatureDraft(features=features).model_dump_json()


EP_SENTS = split_sentences(SOURCES["EP3667784A1"])


def first_sentence_first(features: list[str], sentences: list[str]) -> list[list[int]]:
    """Ranking stub: every feature tries the sentences in source order."""
    return [list(range(len(sentences))) for _ in features]


def nli_matching_words(premise: str, claim: str) -> float:
    """Features are supported by the idea; a row only by a sentence about the same thing."""
    if premise == QUERY:
        return 0.9
    return 0.9 if any(w in claim and w in premise for w in ("carbon", "OI")) else 0.1


def test_clean_features_dedupes_and_terminates():
    raw = ["  an anode ", "An anode", "", "x", "b c d", "e f g", "h i j", "k l m", "n o p"]
    assert clean_features(raw) == ["an anode.", "b c d.", "e f g.", "h i j.", "k l m."]


def test_rank_by_overlap_puts_the_sentence_sharing_most_words_first():
    order = rank_by_overlap(FEATURES, EP_SENTS)
    assert order == [[0, 1], [1, 0]]


class FakeEmbedder:
    """2-d vectors: 'carbon' texts point one way, everything else the other."""

    def embed(self, texts, kind):
        return np.array([[1.0, 0.0] if "carbon" in t else [0.0, 1.0] for t in texts])


def test_embedding_ranker_orders_by_cosine():
    rank = embedding_ranker(FakeEmbedder())
    assert rank(["carbon layer", "OI value"], EP_SENTS) == [[0, 1], [1, 0]]


def test_candidate_sentences_keep_the_n_best_per_pair():
    c = candidate_sentences(FEATURES, SOURCES, rank_by_overlap, 1)
    assert c[("EP3667784A1", FEATURES[1])] == ["The OI value is greater than 15."]
    assert len(c) == 4 and all(len(v) == 1 for v in c.values())


def test_build_chart_rows_follow_source_and_feature_order():
    rows = {
        ("EP3667784A1", FEATURES[1]): EP_SENTS[1],
        ("EP3667784A1", FEATURES[0]): EP_SENTS[0],
    }
    rep = build_chart(FEATURES, rows, SOURCES)
    item = rep.items[0]
    assert [r.feature for r in item.chart] == FEATURES
    assert item.relevance == "Discloses features F1, F2 of the idea."
    assert item.title.startswith("An anode comprising")
    assert rep.summary.startswith("Invention features: F1: An anode")
    assert check_citations(rep, SOURCES).passed


def test_chart_claims_are_judged_against_their_evidence_sentence():
    rep = build_chart(FEATURES, {("EP3667784A1", FEATURES[1]): EP_SENTS[1]}, SOURCES)
    claims = atomic_claims(rep)
    assert [(c.source, c.premise) for c in claims] == [
        ("query", ""),
        ("query", ""),
        ("EP3667784A1", "The OI value is greater than 15."),
    ]


def test_features_schema_bounds_the_list():
    fs = features_json_schema(2, 5)["properties"]["features"]
    assert fs["minItems"] == 2 and fs["maxItems"] == 5


def run_report(outputs: list[str], nli=lambda p, c: 0.9, retries: int = 2, rank=None):
    return generate_report(
        ScriptedLLM(outputs),
        QUERY,
        SOURCES,
        max_items=3,
        max_retries=retries,
        llm_fn=lambda s, c: True,
        nli_fn=nli,
        nli_threshold=0.5,
        rank_fn=rank or first_sentence_first,
    )


def test_rejected_row_is_retried_with_the_next_sentence():
    run = run_report([features_json()], nli_matching_words)
    # attempt 1: the first EP sentence passes for F1 only; attempt 2: F2 gets the OI sentence;
    # the WO source has a single sentence, so its two pairs have nothing left to try
    assert len(run.attempts) == 2 and not run.ok
    assert run.report is not None
    chart = run.report.items[0].chart
    assert [(r.feature, r.evidence) for r in chart] == [
        (FEATURES[0], EP_SENTS[0]),
        (FEATURES[1], EP_SENTS[1]),
    ]
    assert all(r.supported for r in chart)
    assert run.report.removed_rows == 2 and "2 of 4" in run.warning


def test_every_pair_verified_on_the_first_try_needs_no_retry():
    one = {"EP3667784A1": SOURCES["EP3667784A1"]}
    run = generate_report(
        ScriptedLLM([features_json()]),
        QUERY,
        one,
        max_items=3,
        max_retries=2,
        llm_fn=lambda s, c: True,
        nli_fn=nli_matching_words,
        nli_threshold=0.5,
        rank_fn=rank_by_overlap,
    )
    assert run.ok and len(run.attempts) == 1 and run.warning == ""
    assert run.report is not None and run.report.removed_rows == 0


def test_report_without_verified_rows_has_no_items():
    run = run_report([features_json()], lambda p, c: 0.9 if p == QUERY else 0.1)
    assert run.report is not None and run.report.items == [] and run.report.removed_rows == 4
    assert run.draft is not None and len(run.draft.items) == 2
    assert "no prior art" in run.warning


def test_rows_of_an_unsupported_feature_are_dropped():
    def nli(premise: str, claim: str) -> float:
        return 0.1 if premise == QUERY and "OI" in claim else 0.9

    run = run_report([features_json()], nli)
    assert run.report is not None and run.report.features == [FEATURES[0]]
    assert all(r.feature == FEATURES[0] for it in run.report.items for r in it.chart)


def test_feature_extraction_failure_produces_no_report():
    one = features_json(["Only one feature."])
    run = run_report([one, one], retries=1)
    assert run.report is None and [a.stage for a in run.attempts] == ["features", "features"]
    assert "could not be split" in run.warning


def test_generate_structured_recovers_from_invalid_json():
    llm = ScriptedLLM(["not json", '{"verdict": "yes"}'])
    v, _ = generate_structured(llm, JudgeVerdict, "p")
    assert v.verdict == "yes" and llm.calls == 2
    with pytest.raises(ValueError):
        generate_structured(ScriptedLLM(["x", "y"]), JudgeVerdict, "p")


def test_source_text_contains_title_and_abstract():
    p = Patent(
        pub_id=PubId(country="EP", number="1", kind="A1"),
        title="Anode thing.",
        abstract="Abstract words here.",
        claims="1. An anode comprising X. 2. The anode of claim 1.",
    )
    t = source_text(p)
    assert t.startswith("Anode thing.") and "Abstract words here." in t and "Claim 1:" in t


def test_memory_cache_and_trace_roundtrip(tmp_path):
    m = Memory(tmp_path / "m.db")
    k = query_hash("  an  anode ", k=5)
    assert k == query_hash("an anode", k=5) and k != query_hash("an anode", k=6)
    assert m.get(k, "corpusA", "report") is None
    m.put(k, "corpusA", "report", {"x": 1})
    assert m.get(k, "corpusA", "report") == {"x": 1}
    assert m.get(k, "corpusB", "report") is None  # keyed by corpus hash
    tid = m.save_trace(k, "ok", [{"tool": "classify"}], {"summary": "s"})
    assert m.load_trace(tid)["steps"] == [{"tool": "classify"}]
    assert m.load_trace("missing") is None
    m.close()
