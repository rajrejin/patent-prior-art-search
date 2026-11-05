"""Agent evaluation: retrieval quality of the agent's fused ranking, its tool use, and its
reports.

Each agent run is appended to a JSONL checkpoint (gitignored: it contains patent text), so an
interrupted run resumes. Retrieval runs use `retrieval_only=True` (no claim chart); report runs
are the full agent and feed the faithfulness study in `evaluation/faith_eval.py`.
"""

from __future__ import annotations

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from patentrag.agent.loop import AgentRun, PriorArtAgent, Search, fuse
from patentrag.config import Config
from patentrag.evaluation import runner
from patentrag.evaluation.faith_eval import attempt_record
from patentrag.evaluation.harness import ensure_testset, subsample
from patentrag.evaluation.params import RESULTS, save_section
from patentrag.evaluation.testset import Query, build_indexes
from patentrag.evaluation.tuning import Scorer, mean
from patentrag.retrieval.base import Hit

FEATURE_WEIGHTS = [0.0, 0.25, 0.5, 1.0]


def runs_path(kind: str, split: str, scope: str) -> Path:
    return RESULTS / f"agent_{kind}_{split}_{scope}.jsonl"


def record(q: Query, run: AgentRun, agent: PriorArtAgent) -> dict[str, Any]:
    rep = run.report
    return {
        "qid": q.qid,
        "query": q.text,
        "features": run.features,
        "searches": [
            {"kind": s.kind, "text": s.text, "hits": [[h.pub_id, h.score] for h in s.hits]}
            for s in run.searches
        ],
        "steps": [s.as_dict() for s in run.steps],
        "stats": run.stats(),
        "sources": rep.sources if rep else {},
        "ok": bool(rep and rep.ok),
        "warning": rep.warning if rep else "",
        "attempts": [attempt_record(a) for a in rep.attempts] if rep else [],
        "report": rep.report.model_dump() if rep and rep.report else None,
        "draft": rep.draft.model_dump() if rep and rep.draft else None,
    }


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["qid"]] = rec
    return out


def workers() -> int:
    """Agent runs evaluated concurrently (PATENTRAG_AGENT_WORKERS, default 1).

    Runs are independent, so threads change throughput, not results; the LLM server must accept
    that many parallel requests (Ollama: OLLAMA_NUM_PARALLEL)."""
    return max(1, int(os.environ.get("PATENTRAG_AGENT_WORKERS", "1") or 1))


def run_queries(
    agent: PriorArtAgent, queries: list[Query], path: Path, *, retrieval_only: bool
) -> list[dict[str, Any]]:
    done = load_records(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = [q for q in queries if q.qid not in done]
    lock = threading.Lock()

    def one(q: Query) -> None:
        run = agent.run(q.text, excl=q.exclusion(), retrieval_only=retrieval_only)
        rec = record(q, run, agent)
        with lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            done[q.qid] = rec

    if todo:
        one(todo[0])  # the first run loads every model once, before any thread starts
    if workers() > 1:
        with ThreadPoolExecutor(workers()) as pool:
            list(pool.map(one, todo[1:]))
    else:
        for q in todo[1:]:
            one(q)
    return [done[q.qid] for q in queries if q.qid in done]


def searches_of(rec: dict[str, Any]) -> list[Search]:
    return [
        Search(s["kind"], s["text"], [Hit(p, float(sc)) for p, sc in s["hits"]])
        for s in rec["searches"]
    ]


def ranked(
    recs: list[dict[str, Any]], key_of: dict[str, str], weight: float, rrf_k: int
) -> dict[str, list[str]]:
    out = {}
    for r in recs:
        ids = [p for p, _ in fuse(searches_of(r), weight, rrf_k)]
        out[r["qid"]] = runner.collapse_to_documents(ids, key_of)[: runner.K_MAX]
    return out


def behaviour(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """How the agent used its tools: who chose each step, which tools, how many steps."""
    stats = [r["stats"] for r in recs]
    steps = sum(s["steps"] for s in stats)
    calls = sum(s["planner_calls"] for s in stats)
    tools: dict[str, int] = {}
    for s in stats:
        for t, n in s["tools"].items():
            tools[t] = tools.get(t, 0) + n
    return {
        "n_runs": len(recs),
        "mean_steps": steps / len(recs) if recs else 0.0,
        "planner_calls": calls,
        "planner_action_accept_rate": sum(s["planner_actions_used"] for s in stats) / calls
        if calls
        else 0.0,
        "replaced_actions": sum(s["replaced_actions"] for s in stats),
        "guardrail_steps": sum(s["guardrail_steps"] for s in stats),
        "planner_output_errors": sum(s["planner_errors"] for s in stats),
        "mean_feature_searches": sum(s["feature_searches"] for s in stats) / len(recs)
        if recs
        else 0.0,
        "tool_use": dict(sorted(tools.items())),
        "mean_seconds": sum(s["seconds"] for s in stats) / len(recs) if recs else 0.0,
        "concurrent_runs": workers(),
    }


def tune_agent(cfg: Config, scope: str, agent: PriorArtAgent) -> dict[str, Any]:
    """Run the retrieval agent on validation queries and choose the feature-search weight."""
    patents = list(agent.stack.patents.values())
    val, _, _ = ensure_testset(cfg, scope, patents)
    val = subsample(val, cfg.seed)
    key_of = build_indexes(patents).key_of
    recs = run_queries(agent, val, runs_path("retrieval", "val", scope), retrieval_only=True)
    sc = Scorer(cfg, key_of, val)
    rows = []
    best_w, best = 0.0, -1.0
    for w in FEATURE_WEIGHTS:
        res = sc.score_ranked(f"agent|{w}", ranked(recs, key_of, w, agent.p.rrf_k), [])
        rows.append({"feature_weight": w, "ndcg@10": mean(res), "r@10": mean(res, "r@10")})
        if mean(res) > best + 1e-9:
            best_w, best = w, mean(res)
    params = agent.p.model_copy(update={"feature_weight": best_w})
    save_section(
        scope,
        "agent",
        {
            "selected_on": "validation",
            "n_val_queries": len(val),
            "params": params.model_dump(),
            "val_ndcg@10": best,
            "search_log": rows,
            "behaviour_val": behaviour(recs),
        },
    )
    return {"params": params.model_dump(), "log": rows}


def agent_run_result(
    cfg: Config, scope: str, agent: PriorArtAgent, queries: list[Query], split: str
) -> tuple[runner.RunResult, dict[str, Any]]:
    """The agent as one more method for `harness.evaluate_methods` (retrieval mode)."""
    key_of = build_indexes(list(agent.stack.patents.values())).key_of
    recs = run_queries(agent, queries, runs_path("retrieval", split, scope), retrieval_only=True)
    res = runner.RunResult("agent")
    res.ranked = ranked(recs, key_of, agent.p.feature_weight, agent.p.rrf_k)
    res.latency_ms = [r["stats"]["seconds"] * 1000 for r in recs]
    return res, behaviour(recs)


def report_queries(cfg: Config, scope: str, n_val: int, n_test: int) -> dict[str, list[Query]]:
    """Seeded random validation (threshold calibration) and test (evaluation) queries."""
    from patentrag.ingest.corpus import load_corpus

    val, test, _ = ensure_testset(cfg, scope, load_corpus(cfg, scope))
    out = {}
    for name, pool_src, n in (("val", val, n_val), ("test", test, n_test)):
        pool = sorted(pool_src, key=lambda q: q.qid)
        random.Random(cfg.seed).shuffle(pool)  # noqa: S311
        out[name] = pool[:n]
    return out
