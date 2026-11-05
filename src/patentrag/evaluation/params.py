"""Tuned parameters: one JSON file per scope, written only by tuning on validation queries.

results/tuned_params_<scope>.json holds a section per component ("hybrid", "graph", "rerank",
"agent"), each with the chosen "params" and its search log. Loaders fall back to the config
defaults when a section is missing, so serving works before any tuning has run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from patentrag.config import ROOT, AgentCfg, Config, GraphCfg, HybridCfg, RerankCfg

RESULTS = ROOT / "results"


def params_path(scope: str) -> Path:
    return RESULTS / f"tuned_params_{scope}.json"


def load_all(scope: str) -> dict[str, Any]:
    p = params_path(scope)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_section(scope: str, name: str, section: dict[str, Any]) -> None:
    data = load_all(scope)
    data[name] = section
    params_path(scope).parent.mkdir(parents=True, exist_ok=True)
    params_path(scope).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _params(scope: str, name: str) -> dict[str, Any] | None:
    sec = load_all(scope).get(name)
    return sec["params"] if sec else None


def load_tuned_params(cfg: Config, scope: str) -> GraphCfg:
    p = _params(scope, "graph")
    return GraphCfg.model_validate(p) if p else cfg.graph


def load_tuned_hybrid(cfg: Config, scope: str) -> HybridCfg:
    p = _params(scope, "hybrid")
    return HybridCfg.model_validate(p) if p else cfg.hybrid


def load_tuned_rerank(cfg: Config, scope: str) -> RerankCfg:
    p = _params(scope, "rerank")
    return RerankCfg.model_validate(p) if p else cfg.rerank


def load_tuned_agent(cfg: Config, scope: str) -> AgentCfg:
    p = _params(scope, "agent")
    return AgentCfg.model_validate(p) if p else cfg.agent
