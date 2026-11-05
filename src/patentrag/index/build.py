"""Build the vector (ChromaDB) and BM25 indexes for a corpus scope."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from patentrag.config import Config
from patentrag.index.bm25 import Bm25Index
from patentrag.index.chunking import Chunk, chunk_patent, patent_text
from patentrag.index.embed import Embedder, info
from patentrag.index.vector_store import VectorStore
from patentrag.ingest.corpus import corpus_paths, load_corpus
from patentrag.manifest import base_manifest

log = logging.getLogger(__name__)


def index_dirs(cfg: Config, scope: str) -> tuple[Path, Path, Path]:
    chroma = cfg.path(cfg.index.chroma_dir) / scope
    bm25 = cfg.path(cfg.index.bm25_dir) / scope / "bm25.pkl"
    manifest = cfg.path("stores") / f"index_manifest_{scope}.json"
    return chroma, bm25, manifest


def build_indexes(cfg: Config, scope: str) -> dict[str, Any]:
    patents = load_corpus(cfg, scope)
    corpus_hash = json.loads(corpus_paths(cfg, scope)[1].read_text(encoding="utf-8"))["corpus_hash"]
    chroma_dir, bm25_path, manifest_path = index_dirs(cfg, scope)

    t0 = time.time()
    ids = [str(p.pub_id) for p in patents]
    Bm25Index(ids, [patent_text(p) for p in patents]).save(bm25_path)
    t_bm25 = time.time() - t0

    chunks: list[Chunk] = [c for p in patents for c in chunk_patent(p)]
    log.info("%d patents -> %d chunks", len(patents), len(chunks))
    t1 = time.time()
    emb = Embedder(cfg.index.embedding_model, cfg.index.max_seq_length, cfg.index.batch_size)
    vecs = emb.embed_passages_checkpointed(
        [c.chunk_id for c in chunks],
        [c.text for c in chunks],
        cfg.path("stores") / "embeddings" / scope,
    )
    t_embed = time.time() - t1
    store = VectorStore(chroma_dir)
    store.reset()
    store.upsert(chunks, vecs)

    sections: dict[str, int] = {}
    for c in chunks:
        sections[c.section] = sections.get(c.section, 0) + 1
    manifest = base_manifest(
        cfg,
        corpus_hash,
        scope=scope,
        embedding=info(emb),
        n_patents=len(patents),
        n_chunks=len(chunks),
        chunks_by_section=sections,
        seconds={"bm25": round(t_bm25, 1), "embed": round(t_embed, 1)},
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
