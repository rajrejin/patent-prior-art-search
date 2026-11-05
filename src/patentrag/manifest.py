"""Reproducibility manifests: config hash, corpus hash, library versions, models, seed, commit."""

from __future__ import annotations

import subprocess
from importlib import metadata
from typing import Any

from patentrag.config import ROOT, Config

LIBS = ["numpy", "torch", "sentence-transformers", "chromadb", "rank-bm25", "neo4j", "pydantic"]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "uncommitted"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def lib_versions() -> dict[str, str]:
    out = {}
    for name in LIBS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "absent"
    return out


def base_manifest(cfg: Config, corpus_hash: str, **extra: Any) -> dict[str, Any]:
    return {
        "config_hash": cfg.config_hash(),
        "corpus_hash": corpus_hash,
        "seed": cfg.seed,
        "git_commit": git_commit(),
        "libraries": lib_versions(),
        **extra,
    }
