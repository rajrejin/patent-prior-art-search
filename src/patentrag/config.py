"""Configuration: one YAML file plus environment overrides for paths and secrets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("PATENTRAG_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class DataCfg(BaseModel):
    topic: str
    cpc_prefixes: list[str]
    authorities: list[str]
    year_start: int
    year_end: int
    slice_year: int
    slice_month: int
    slice_size: int
    target_min: int
    target_max: int
    seeds_per_month: int = 7
    raw_dir: str
    corpus_dir: str
    fetch_state: str
    weekly_quota_bytes: int
    quota_stop_fraction: float = 0.8
    fetch_claims: bool = True
    fetch_descriptions: bool = False


class GraphCfg(BaseModel):
    seeds_n: int = 20
    pool_n: int = 100
    cpc_top: int = 3
    cpc_level: str = "group"
    cpc_depth: int = 1
    citation_hops: int = 1
    max_candidates: int = 200
    use_cpc: bool = True
    use_cite: bool = True
    use_family: bool = True
    use_rerank: bool = True
    w_sim: float = 1.0
    w_cpc: float = 0.2
    w_cite: float = 0.2
    w_family: float = 0.0
    w_applicant: float = 0.1
    # lexical signal (0 / "vector" reproduce the vector-only GraphRAG)
    w_bm25: float = 0.0
    bm25_pool: int = 0
    seed_source: str = "vector"  # "vector" or "hybrid"


class HybridCfg(BaseModel):
    w_sim: float = 1.0
    w_bm25: float = 1.0
    pool_n: int = 100


class RerankCfg(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: list[str] = Field(
        default_factory=lambda: ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]
    )
    depth: int = 50
    alpha: float = 1.0


class AgentCfg(BaseModel):
    max_steps: int = 10
    search_k: int = 20
    max_feature_searches: int = 4
    max_chart_items: int = 3
    max_reads: int = 3
    max_expansions: int = 2
    feature_weight: float = 0.5  # RRF weight of feature searches relative to the idea search
    rrf_k: int = 60
    chart_boost: bool = False  # verified claim-chart documents first in the final ranking


class IndexCfg(BaseModel):
    embedding_model: str
    chroma_dir: str
    bm25_dir: str
    batch_size: int = 32
    max_seq_length: int = 256
    aggregation: str = "max"


class EvalCfg(BaseModel):
    n_test: int = 300
    n_val: int = 100
    min_relevant: int = 3
    ks: list[int] = Field(default_factory=lambda: [5, 10, 20])
    bootstrap_resamples: int = 1000


class LLMCfg(BaseModel):
    backend: str = "ollama"
    model: str = "qwen2.5:3b-instruct"
    temperature: float = 0.0
    num_ctx: int = 4096


class FaithCfg(BaseModel):
    nli_model: str
    max_retries: int = 2
    flag_if_either: bool = True


class ApiCfg(BaseModel):
    max_query_chars: int = 4000
    max_k: int = 50
    timeout_s: int = 120
    report_timeout_s: int = 1800  # synchronous /v1/report; long runs should use report jobs
    max_jobs: int = 50
    cors: bool = False


class Config(BaseModel):
    seed: int = 0
    data: DataCfg
    graph: GraphCfg = GraphCfg()
    hybrid: HybridCfg = HybridCfg()
    rerank: RerankCfg = RerankCfg()
    agent: AgentCfg = AgentCfg()
    index: IndexCfg
    evaluation: EvalCfg = EvalCfg()
    llm: LLMCfg = LLMCfg()
    faithfulness: FaithCfg
    api: ApiCfg = ApiCfg()
    sqlite_path: str = "stores/sqlite/memory.db"

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: Path | None = None) -> Config:
    """Load YAML config. PATENTRAG_CONFIG overrides the path; .env is loaded for secrets."""
    load_dotenv(ROOT / ".env")
    cfg_path = Path(os.environ.get("PATENTRAG_CONFIG", path or DEFAULT_CONFIG))
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return Config.model_validate(raw)


class Secrets(BaseModel):
    """Secrets are read from the environment only. Never log or print these."""

    ops_key: str = ""
    ops_secret: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    ollama_host: str = "http://localhost:11434"
    api_key: str = ""

    def __repr__(self) -> str:
        return "Secrets(<redacted>)"

    __str__ = __repr__


def load_secrets() -> Secrets:
    load_dotenv(ROOT / ".env")
    e = os.environ
    return Secrets(
        ops_key=e.get("OPS_CONSUMER_KEY", ""),
        ops_secret=e.get("OPS_CONSUMER_SECRET", ""),
        neo4j_uri=e.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=e.get("NEO4J_USER", "neo4j"),
        neo4j_password=e.get("NEO4J_PASSWORD", ""),
        ollama_host=e.get("OLLAMA_HOST", "http://localhost:11434"),
        api_key=e.get("PATENTRAG_API_KEY", ""),
    )
