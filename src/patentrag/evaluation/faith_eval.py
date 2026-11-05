"""Faithfulness evaluation: checks, retries, judge agreement, negative control, audit.

The records come from full agent runs on real queries (title plus abstract of test patents,
with leakage controls), see `evaluation/agent_eval.py`. `audit` re-judges the claim-chart rows
with a larger model that the pipeline does not use.
"""

from __future__ import annotations

import json
import random
from typing import Any

from patentrag.agent.report import Attempt
from patentrag.config import ROOT
from patentrag.faithfulness.claims import split_sentences
from patentrag.faithfulness.judge import cohens_kappa
from patentrag.faithfulness.threshold import threshold_path

RESULTS = ROOT / "results"


def attempt_record(a: Attempt) -> dict[str, Any]:
    claims = []
    if a.support:
        for v in a.support.verdicts:
            claims.append(
                {
                    "source": v.claim.source,
                    "text": v.claim.text,
                    "premise": v.claim.premise,
                    "llm": v.llm_supported,
                    "nli": v.nli_prob,
                    "supported": v.supported,
                }
            )
    return {
        "n": a.n,
        "stage": a.stage,
        "seconds": round(a.seconds, 1),
        "ok": a.ok,
        "error": a.error,
        "valid_citation_fraction": a.citation.valid_citation_fraction if a.citation else None,
        "evidence_valid_fraction": a.citation.evidence_valid_fraction if a.citation else None,
        "invalid_numbers": a.citation.invalid if a.citation else [],
        "claims": claims,
    }


def _chart_attempts(r: dict[str, Any]) -> list[dict[str, Any]]:
    return [a for a in r["attempts"] if a.get("stage", "chart") == "chart" and not a["error"]]


def _final_claims(r: dict[str, Any]) -> list[dict[str, Any]]:
    """Each claim's last judgement: features once, and for every feature and patent pair the
    row that was verified or, if none was, the last sentence tried."""
    last: dict[tuple[str, str], dict[str, Any]] = {}
    for a in _chart_attempts(r):
        for c in a["claims"]:
            last[(c["source"], c["text"])] = c
    return list(last.values())


def _rates(claims: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    llm = [bool(c["llm"]) for c in claims]
    nli = [(c["nli"] or 0.0) >= threshold for c in claims]
    return {
        "n_claims": len(claims),
        "support_rate_llm": _mean(llm),
        "support_rate_nli": _mean(nli),
        "support_rate_combined": _mean([c["supported"] for c in claims]),
        "kappa_llm_vs_nli": cohens_kappa(llm, nli) if claims else None,
        "agreement_llm_vs_nli": _mean([x == y for x, y in zip(llm, nli, strict=True)]),
    }


def summarise(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Row support for the first-ranked sentence and after retries, feature support, judge
    agreement, and what the returned reports contain."""

    def rows(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [c for c in claims if c["source"] != "query"]

    firsts = [atts[0] for atts in map(_chart_attempts, records) if atts]
    first = _rates(rows([c for a in firsts for c in a["claims"]]), threshold)
    first["evidence_from_cited_patent"] = _mean([a["evidence_valid_fraction"] for a in firsts])
    final = [c for r in records for c in _final_claims(r)]
    shipped = [r.get("report") for r in records]
    kept_rows = [
        sum(len(i.get("chart", [])) for i in (rep or {}).get("items", [])) for rep in shipped
    ]
    removed = [(rep or {}).get("removed_rows", 0) for rep in shipped]
    pairs = [k + d for k, d in zip(kept_rows, removed, strict=True)]
    return {
        "first_choice": first,
        "after_retries": _rates(rows(final), threshold),
        "features": _rates([c for c in final if c["source"] == "query"], threshold),
        "shipped": {
            "reports_with_verified_rows": _mean([k > 0 for k in kept_rows]),
            "mean_verified_items": _mean([len((rep or {}).get("items", [])) for rep in shipped]),
            "mean_verified_rows": _mean(kept_rows),
            "mean_pairs": _mean(pairs),
            "pair_fill_rate": sum(kept_rows) / sum(pairs) if sum(pairs) else 0.0,
            "mean_features": _mean([len((rep or {}).get("features", [])) for rep in shipped]),
            "feature_extraction_failed": _mean([rep is None for rep in shipped]),
        },
        "mean_attempts": _mean([len(_chart_attempts(r)) for r in records]),
        "mean_seconds_per_report": _mean(
            [sum(a["seconds"] for a in r["attempts"]) for r in records]
        ),
        "nli_threshold": threshold,
    }


def _mean(xs: list[Any]) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _mismatch_premise(c: dict[str, Any], other: dict[str, Any], rng: random.Random) -> str:
    """Text of a random source of another query; one random sentence of it when the claim was
    judged against a single evidence sentence, so both premises have similar length."""
    text = str(other["sources"][rng.choice(sorted(other["sources"]))])
    if c.get("premise"):
        sents = split_sentences(text) or [text]
        return rng.choice(sents)
    return text


def negative_control(
    llm: Any, nli: Any, records: list[dict[str, Any]], threshold: float, seed: int = 0
) -> dict[str, Any]:
    """Judge each report's claims against text from a different, random query.

    A judge that can tell support from non-support must give a far lower rate here.
    """
    from patentrag.faithfulness.judge import llm_judge

    rng = random.Random(seed)  # noqa: S311
    rows = []
    usable = [r for r in records if _final_claims(r)]
    if len(usable) < 2:
        return {"error": "need at least two reports"}
    for i, r in enumerate(usable):
        others = [o for j, o in enumerate(usable) if j != i]
        for c in _final_claims(r):
            if c["source"] == "query":
                continue
            text = _mismatch_premise(c, rng.choice(others), rng)
            rows.append(
                {
                    "matched_llm": bool(c["llm"]),
                    "matched_nli": (c["nli"] or 0.0) >= threshold,
                    "mismatch_llm": llm_judge(llm, text, c["text"]),
                    "mismatch_nli": nli.entailment(text, c["text"]) >= threshold,
                }
            )
    n = len(rows)
    out: dict[str, Any] = {"n_claims": n}
    for j in ("llm", "nli"):
        out[f"matched_support_rate_{j}"] = _mean([x[f"matched_{j}"] for x in rows])
        out[f"mismatched_support_rate_{j}"] = _mean([x[f"mismatch_{j}"] for x in rows])
    out["judges_discriminate"] = all(
        out[f"matched_support_rate_{j}"] - out[f"mismatched_support_rate_{j}"] >= 0.3
        for j in ("llm", "nli")
    )
    return out


def calibrate_threshold(nli: Any, records: list[dict[str, Any]], seed: int = 0) -> dict:
    """Pick the NLI threshold on validation reports: matched claims versus mismatched text.

    Maximises balanced accuracy of the NLI judge at telling a claim's own source from a random
    other query's source. Test queries are never used here.
    """
    rng = random.Random(seed)  # noqa: S311
    usable = [r for r in records if _final_claims(r)]
    pos: list[float] = []
    neg: list[float] = []
    for i, r in enumerate(usable):
        others = [o for j, o in enumerate(usable) if j != i]
        for c in _final_claims(r):
            if c["source"] == "query" or c["nli"] is None or not others:
                continue
            pos.append(float(c["nli"]))
            neg.append(nli.entailment(_mismatch_premise(c, rng.choice(others), rng), c["text"]))
    best_t, best_acc = 0.5, -1.0
    for t in [x / 20 for x in range(2, 19)]:
        tpr = _mean([p >= t for p in pos])
        tnr = _mean([q < t for q in neg])
        acc = (tpr + tnr) / 2
        if acc > best_acc:
            best_t, best_acc = t, acc
    out = {
        "threshold": best_t,
        "balanced_accuracy": best_acc,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "selected_on": "validation",
    }
    threshold_path().write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def audit(judge: Any, records: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Re-judge every chart row of the final draft with a model the pipeline does not use.

    Rows the pipeline kept should be supported far more often than rows it removed, and the
    audit judge itself must reject rows paired with a random sentence of another query.
    """
    from patentrag.faithfulness.judge import llm_judge

    rng = random.Random(seed)  # noqa: S311
    usable = [r for r in records if _final_claims(r)]
    kept: list[bool] = []
    removed: list[bool] = []
    features: list[bool] = []
    control: list[bool] = []
    for i, r in enumerate(usable):
        others = [o for j, o in enumerate(usable) if j != i]
        shipped = {
            (it["pub_id"], row["feature"], row["evidence"])
            for it in (r["report"] or {}).get("items", [])
            for row in it.get("chart", [])
        }
        for c in _final_claims(r):
            if c["source"] == "query":
                features.append(llm_judge(judge, c["premise"] or r.get("query", ""), c["text"]))
                continue
            premise = c["premise"] or r["sources"].get(c["source"], "")
            verdict = llm_judge(judge, premise, c["text"])
            in_report = (c["source"], c["text"], c["premise"]) in shipped
            (kept if in_report else removed).append(verdict)
            if others:
                control.append(
                    llm_judge(judge, _mismatch_premise(c, rng.choice(others), rng), c["text"])
                )
    return {
        "judge_model": getattr(judge, "model", ""),
        "n_reports": len(usable),
        "feature_support_rate": _mean(features),
        "n_rows_kept": len(kept),
        "n_rows_removed": len(removed),
        "support_rate_kept_rows": _mean(kept),
        "support_rate_removed_rows": _mean(removed),
        "support_rate_all_draft_rows": _mean(kept + removed),
        "support_rate_mismatched_control": _mean(control),
        "n_control": len(control),
    }
