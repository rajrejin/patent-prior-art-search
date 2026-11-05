"""Embedding with sentence-transformers (CPU), e5 prefixes, L2 normalisation, checkpointing."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, max_seq_length: int = 256, batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.manual_seed(0)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.max_seq_length = max_seq_length
        self.is_e5 = "e5" in model_name.lower()

    @property
    def revision(self) -> str:
        try:
            first: Any = self.model[0]
            return str(getattr(first.auto_model.config, "_commit_hash", None) or "unknown")
        except Exception:  # noqa: BLE001
            return "unknown"

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension() or 0)

    def _prep(self, texts: list[str], kind: str) -> list[str]:
        if not self.is_e5:
            return texts
        return [f"{kind}: {t}" for t in texts]

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        """kind is 'query' or 'passage'. Returns float32, L2-normalised."""
        vecs = self.model.encode(
            self._prep(texts, kind),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts, "query")

    def embed_passages_checkpointed(
        self, ids: list[str], texts: list[str], ckpt_dir: Path, shard: int = 512
    ) -> np.ndarray:
        """Embed passages in shards; finished shards are reused on rerun."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha256(
            (self.model_name + "|".join(ids) + "|".join(texts)).encode()
        ).hexdigest()[:12]
        parts: list[np.ndarray] = []
        for i in range(0, len(texts), shard):
            f = ckpt_dir / f"{tag}_{i // shard:05d}.npy"
            if f.exists():
                parts.append(np.load(f))
                continue
            arr = self.embed(texts[i : i + shard], "passage")
            tmp = f.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(f)
            parts.append(arr)
            log.info("embedded %d/%d", min(i + shard, len(texts)), len(texts))
        return np.vstack(parts) if parts else np.zeros((0, self.dim), dtype=np.float32)


def info(e: Embedder) -> dict[str, Any]:
    return {"model": e.model_name, "revision": e.revision, "dim": e.dim}
