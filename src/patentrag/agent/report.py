"""Grounded prior-art reports as a claim chart, with deterministic and claim-level checks.

1. extract_features: the LLM restates the invention idea as a few technical features.
2. retrieve evidence: for every charted patent and every feature, the patent's sentences are
   ranked by embedding similarity to the feature. Evidence is always a sentence of the patent's
   own text, so patent numbers and quotes cannot be invented.
3. verify: every feature is judged against the idea, and every row ("this sentence states this
   feature") by two judges. A rejected row is retried with the next-ranked sentence, up to
   `max_retries` times.
4. ship: a feature and patent pair whose sentences were all rejected is left out of the chart
   and counted; a report with no verified row is returned with a warning.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from patentrag.faithfulness.citations import CitationCheck, check_citations, normalise_ws
from patentrag.faithfulness.claims import split_sentences
from patentrag.faithfulness.judge import SupportResult, check_support
from patentrag.llm.client import LLMClient, generate_structured
from patentrag.llm.schemas import (
    ChartRow,
    FeatureDraft,
    PriorArtItem,
    PriorArtReport,
    features_json_schema,
)

SYSTEM = (
    "You are a careful patent search assistant. Use ONLY the text you are given. "
    "Never state facts that the text does not contain."
)
FEATURES_TASK = (
    "INVENTION IDEA:\n{query}\n\nTASK: List {lo} to {hi} key technical features of this "
    "invention idea as JSON. Each feature is one short sentence that restates part of the idea "
    "in its own words, without adding facts that the idea does not state."
)
LIMITATIONS = (
    "Text-similarity assessment of retrieved titles, abstracts and first claims; "
    "not a legal opinion on novelty or freedom to operate."
)
MIN_FEATURES, MAX_FEATURES = 2, 5

# ranks the sentences for each feature: (features, sentences) -> one index list per feature,
# best first
RankFn = Callable[[list[str], list[str]], list[list[int]]]


def rank_by_overlap(features: list[str], sentences: list[str]) -> list[list[int]]:
    """Word-overlap ranking, used when no embedding model is available (tests, fallbacks)."""

    def words(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    sw = [words(s) for s in sentences]
    out = []
    for f in features:
        fw = words(f)
        out.append(sorted(range(len(sentences)), key=lambda i: (-len(fw & sw[i]), i)))
    return out


def embedding_ranker(embedder: Any) -> RankFn:
    """Rank sentences by cosine similarity to each feature (e5: feature as query, sentence as
    passage). The judges never see these scores; they only see the chosen sentence."""

    def rank(features: list[str], sentences: list[str]) -> list[list[int]]:
        q = embedder.embed(features, "query")
        s = embedder.embed(sentences, "passage")
        sims = q @ s.T
        return [[int(i) for i in np.argsort(-row, kind="stable")] for row in sims]

    return rank


def clean_features(raw: list[str]) -> list[str]:
    """Strip, drop empty and duplicate features (case-insensitive), keep at most MAX_FEATURES."""
    out: list[str] = []
    seen: set[str] = set()
    for f in raw:
        f = " ".join(f.split()).strip(" -*")
        if len(f) < 4 or f.lower() in seen:
            continue
        seen.add(f.lower())
        out.append(f if f.endswith(".") else f + ".")
    return out[:MAX_FEATURES]


def relevance_text(rows: list[ChartRow], features: list[str]) -> str:
    nums = sorted({features.index(r.feature) + 1 for r in rows if r.feature in features})
    if not nums:
        return ""
    label = "feature" if len(nums) == 1 else "features"
    return f"Discloses {label} " + ", ".join(f"F{n}" for n in nums) + " of the idea."


def summary_text(features: list[str]) -> str:
    return "Invention features: " + " ".join(f"F{i}: {f}" for i, f in enumerate(features, 1))


def title_of(text: str) -> str:
    sents = split_sentences(text) or [text]
    return sents[0].rstrip(".")


def build_chart(
    features: list[str], rows: dict[tuple[str, str], str], sources: dict[str, str]
) -> PriorArtReport:
    """A report with one row per (patent, feature) pair, in source and feature order."""
    items: list[PriorArtItem] = []
    for pid in sources:
        chart = [ChartRow(feature=f, evidence=rows[(pid, f)]) for f in features if (pid, f) in rows]
        if chart:
            items.append(
                PriorArtItem(
                    pub_id=pid,
                    title=title_of(sources[pid]),
                    relevance=relevance_text(chart, features),
                    evidence=chart[0].evidence,
                    chart=chart,
                )
            )
    return PriorArtReport(
        summary=summary_text(features), items=items, limitations=LIMITATIONS, features=features
    )


def mark_and_prune(
    report: PriorArtReport, sup: SupportResult, sources: dict[str, str]
) -> PriorArtReport:
    """Keep only verified content: supported features, and rows that are supported, whose
    feature is kept and whose evidence sentence occurs in the cited patent's own text."""
    verdict = {(v.claim.source, v.claim.text, v.claim.premise): v.supported for v in sup.verdicts}
    features = [f for f in report.features if verdict.get(("query", f, ""), False)]
    items: list[PriorArtItem] = []
    removed = 0
    for it in report.items:
        kept = []
        own = normalise_ws(sources.get(it.pub_id, ""))
        for r in it.chart:
            ok = (
                r.feature in features
                and verdict.get((it.pub_id, r.feature, r.evidence), False)
                and bool(r.evidence)
                and normalise_ws(r.evidence) in own
            )
            if ok:
                kept.append(r.model_copy(update={"supported": True}))
            else:
                removed += 1
        if kept:
            items.append(
                it.model_copy(
                    update={
                        "chart": kept,
                        "evidence": kept[0].evidence,
                        "relevance": relevance_text(kept, report.features),
                    }
                )
            )
    return report.model_copy(
        update={
            "features": features,
            "summary": summary_text(features) if features else "",
            "items": items,
            "removed_rows": removed,
        }
    )


@dataclass
class Attempt:
    n: int
    seconds: float
    citation: CitationCheck | None
    support: SupportResult | None
    ok: bool
    error: str = ""
    stage: str = "chart"


@dataclass
class ReportRun:
    report: PriorArtReport | None
    sources: dict[str, str]
    attempts: list[Attempt] = field(default_factory=list)
    ok: bool = False  # every feature and patent pair got a verified row
    warning: str = ""
    draft: PriorArtReport | None = None  # every pair's last tried row, before pruning

    @property
    def first_attempt(self) -> Attempt | None:
        return self.attempts[0] if self.attempts else None


def _memo(fn: Callable[[str, str], Any] | None) -> Callable[[str, str], Any] | None:
    """Judge each (premise, claim) pair once per report: features repeat across attempts."""
    if fn is None:
        return None
    cache: dict[tuple[str, str], Any] = {}

    def wrapped(premise: str, claim: str) -> Any:
        key = (premise, claim)
        if key not in cache:
            cache[key] = fn(premise, claim)
        return cache[key]

    return wrapped


def extract_features(llm: LLMClient, query: str, retries: int) -> tuple[list[str], list[Attempt]]:
    attempts: list[Attempt] = []
    prompt = FEATURES_TASK.format(query=query, lo=MIN_FEATURES + 1, hi=MAX_FEATURES)
    schema = features_json_schema(MIN_FEATURES, MAX_FEATURES)
    for n in range(retries + 1):
        t0 = time.perf_counter()
        try:
            draft, _ = generate_structured(
                llm, FeatureDraft, prompt, system=SYSTEM, json_schema=schema, max_tokens=300
            )
            feats = clean_features(draft.features)
            if len(feats) < MIN_FEATURES:
                raise ValueError(f"only {len(feats)} usable features")
            return feats, attempts
        except ValueError as exc:
            attempts.append(
                Attempt(n, time.perf_counter() - t0, None, None, False, str(exc), "features")
            )
    return [], attempts


def candidate_sentences(
    features: list[str], sources: dict[str, str], rank_fn: RankFn, n: int
) -> dict[tuple[str, str], list[str]]:
    """The n best sentences of each source for each feature, best first."""
    out: dict[tuple[str, str], list[str]] = {}
    for pid, text in sources.items():
        sents = split_sentences(text) or [text]
        for f, order in zip(features, rank_fn(features, sents), strict=True):
            out[(pid, f)] = [sents[i] for i in order[:n]]
    return out


def generate_report(
    llm: LLMClient,
    query: str,
    sources: dict[str, str],
    *,
    features: list[str] | None = None,
    max_items: int,
    max_retries: int,
    llm_fn: Callable[[str, str], bool] | None,
    nli_fn: Callable[[str, str], float] | None,
    nli_threshold: float,
    flag_if_either: bool = True,
    rank_fn: RankFn = rank_by_overlap,
    on_attempt: Callable[[Attempt], None] | None = None,
) -> ReportRun:
    run = ReportRun(None, sources)
    if not features:
        features, failed = extract_features(llm, query, max_retries)
        run.attempts += failed
    if not features:
        run.warning = "The idea could not be split into features; no report was produced."
        return run
    llm_fn, nli_fn = _memo(llm_fn), _memo(nli_fn)
    sources = dict(list(sources.items())[:max_items])
    cands = candidate_sentences(features, sources, rank_fn, max_retries + 1)
    last: dict[tuple[str, str], str] = {}  # each pair's last tried sentence
    verified: set[tuple[str, str]] = set()
    verdicts = SupportResult()
    for n in range(max_retries + 1):
        todo = {k: c[n] for k, c in cands.items() if k not in verified and n < len(c)}
        if not todo:
            break
        t0 = time.perf_counter()
        draft = build_chart(features, todo, sources)
        cit = check_citations(draft, sources)
        sup = check_support(
            draft,
            sources,
            query,
            llm_fn=llm_fn,
            nli_fn=nli_fn,
            nli_threshold=nli_threshold,
            flag_if_either=flag_if_either,
        )
        ok_rows = {(v.claim.source, v.claim.text) for v in sup.verdicts if v.supported}
        for k, sentence in todo.items():
            last[k] = sentence
            if k in ok_rows:
                verified.add(k)
        known = {(v.claim.source, v.claim.text, v.claim.premise) for v in verdicts.verdicts}
        verdicts.verdicts += [
            v for v in sup.verdicts if (v.claim.source, v.claim.text, v.claim.premise) not in known
        ]
        att = Attempt(n, time.perf_counter() - t0, cit, sup, cit.passed and not sup.unsupported)
        run.attempts.append(att)
        if on_attempt:
            on_attempt(att)
    run.draft = build_chart(features, last, sources)
    run.ok = len(verified) == len(cands) > 0
    run.report = mark_and_prune(run.draft, verdicts, sources)
    if not run.report.items:
        run.warning = "No feature was found in the charted patents; no prior art is reported."
    elif run.report.removed_rows:
        run.warning = (
            f"{run.report.removed_rows} of {len(cands)} feature and patent pairs have no "
            f"verified evidence (up to {max_retries + 1} sentences were tried for each)."
        )
    return run


def attempt_summary(a: Attempt) -> dict[str, Any]:
    return {
        "attempt": a.n,
        "stage": a.stage,
        "seconds": round(a.seconds, 1),
        "ok": a.ok,
        "error": a.error,
        "valid_citation_fraction": a.citation.valid_citation_fraction if a.citation else None,
        "evidence_valid_fraction": a.citation.evidence_valid_fraction if a.citation else None,
        "n_claims": a.support.n if a.support else None,
        "unsupported": [v.claim.text[:160] for v in a.support.unsupported] if a.support else [],
    }
