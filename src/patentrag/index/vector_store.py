"""ChromaDB persistent store of chunk embeddings (cosine space)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from patentrag.index.chunking import Chunk

COLLECTION = "patent_chunks"


class VectorStore:
    def __init__(self, path: Path):
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION)
        self.col = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return int(self.col.count())

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 2000) -> None:
        for i in range(0, len(chunks), batch):
            cs = chunks[i : i + batch]
            self.col.upsert(
                ids=[c.chunk_id for c in cs],
                embeddings=vectors[i : i + batch].tolist(),
                documents=[c.text for c in cs],
                metadatas=[{"pub_id": c.pub_id, "section": c.section} for c in cs],
            )

    def query(self, vector: np.ndarray, n: int) -> list[dict[str, Any]]:
        n = max(1, min(n, self.count()))
        res: Any = self.col.query(
            query_embeddings=[vector.tolist()],
            n_results=n,
            include=["metadatas", "distances", "documents"],
        )
        out = []
        for cid, meta, dist, doc in zip(
            res["ids"][0],
            res["metadatas"][0],
            res["distances"][0],
            res["documents"][0],
            strict=True,
        ):
            out.append(
                {
                    "chunk_id": cid,
                    "pub_id": meta["pub_id"],
                    "section": meta["section"],
                    "sim": 1.0 - float(dist),
                    "text": doc,
                }
            )
        return out

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        res: Any = self.col.get(ids=ids, include=["embeddings"])
        return {
            i: np.asarray(e, dtype=np.float32)
            for i, e in zip(res["ids"], res["embeddings"], strict=True)
        }


class ChunkMatrix:
    """All chunk embeddings in memory for exact similarity to arbitrary patents.

    Built from the same ChromaDB collection; used by the GraphRAG re-ranker, which must score
    candidates that the approximate top-k query did not return.
    """

    def __init__(self, store: VectorStore):
        res: Any = store.col.get(include=["embeddings", "metadatas", "documents"])
        self.chunk_ids: list[str] = list(res["ids"])
        self.matrix = np.asarray(res["embeddings"], dtype=np.float32)
        self.texts: list[str] = list(res["documents"])
        pubs = [m["pub_id"] for m in res["metadatas"]]
        self.sections = [m["section"] for m in res["metadatas"]]
        self.pub_ids = sorted(set(pubs))
        self.pub_index = {p: i for i, p in enumerate(self.pub_ids)}
        self.chunk_pub = np.array([self.pub_index[p] for p in pubs], dtype=np.int64)

    def patent_sims(
        self, qvec: np.ndarray, sections: set[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(best similarity per patent, index of the best chunk per patent); -1 if no chunk."""
        sims = self.matrix @ qvec
        if sections is not None:
            mask = np.array([s in sections for s in self.sections])
            sims = np.where(mask, sims, -np.inf)
        best = np.full(len(self.pub_ids), -np.inf, dtype=np.float32)
        np.maximum.at(best, self.chunk_pub, sims)
        # best chunk index: first chunk attaining the max for its patent
        arg = np.full(len(self.pub_ids), -1, dtype=np.int64)
        is_best = sims >= best[self.chunk_pub]
        for ci in np.nonzero(is_best)[0][::-1]:
            arg[self.chunk_pub[ci]] = ci
        return best, arg
