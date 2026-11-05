"""The retrieval stack: indexes, models and retrievers, built once and shared.

Evaluation, the agent and the service all build their retrievers here, so the evaluated
configuration is exactly the served one. Heavy parts (graph connection, cross-encoder) load
on first use.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from patentrag.config import Config, GraphCfg, HybridCfg, RerankCfg
from patentrag.evaluation.params import (
    load_tuned_hybrid,
    load_tuned_params,
    load_tuned_rerank,
)
from patentrag.graph.schema import normalise_applicant
from patentrag.index.bm25 import Bm25Index
from patentrag.index.build import index_dirs
from patentrag.index.chunking import source_text
from patentrag.index.vector_store import ChunkMatrix
from patentrag.ingest.corpus import load_corpus
from patentrag.models import Patent
from patentrag.retrieval.base import Retriever
from patentrag.retrieval.hybrid import HybridRetriever
from patentrag.retrieval.lexical import Bm25Retriever
from patentrag.retrieval.reranker import RerankRetriever, Scorer, cross_encoder_scorer
from patentrag.retrieval.vector import VectorRetriever


class Stack:
    def __init__(self, cfg: Config, scope: str):
        self.cfg = cfg
        self.scope = scope
        self._scorers: dict[str, Scorer] = {}

    # -- data and indexes ------------------------------------------------------------------------
    @cached_property
    def patents(self) -> dict[str, Patent]:
        return {str(p.pub_id): p for p in load_corpus(self.cfg, self.scope)}

    @cached_property
    def texts(self) -> dict[str, str]:
        """The text shown to the cross-encoder and the LLM for each patent."""
        return {pid: source_text(p) for pid, p in self.patents.items()}

    @cached_property
    def vector(self) -> VectorRetriever:
        from patentrag.index.embed import Embedder
        from patentrag.index.vector_store import VectorStore

        c = self.cfg.index
        emb = Embedder(c.embedding_model, c.max_seq_length, c.batch_size)
        return VectorRetriever(emb, VectorStore(index_dirs(self.cfg, self.scope)[0]), c.aggregation)

    @cached_property
    def matrix(self) -> ChunkMatrix:
        return ChunkMatrix(self.vector.store)

    @cached_property
    def bm25_index(self) -> Bm25Index:
        return Bm25Index.load(index_dirs(self.cfg, self.scope)[1])

    @cached_property
    def graph_queries(self) -> Any:
        from patentrag.graph.loader import connect
        from patentrag.graph.queries import GraphQueries

        driver = connect()
        driver.verify_connectivity()
        return GraphQueries(driver)

    # -- retrievers ------------------------------------------------------------------------------
    def bm25(self) -> Retriever:
        return Bm25Retriever(self.bm25_index)

    def hybrid(self, params: HybridCfg | None = None) -> HybridRetriever:
        p = params or load_tuned_hybrid(self.cfg, self.scope)
        return HybridRetriever(
            self.vector, self.matrix, self.bm25_index, p.w_sim, p.w_bm25, p.pool_n
        )

    def scorer(self, model: str) -> Scorer:
        if model not in self._scorers:
            self._scorers[model] = cross_encoder_scorer(model)
        return self._scorers[model]

    def rerank(
        self, params: RerankCfg | None = None, base: Retriever | None = None
    ) -> RerankRetriever:
        p = params or load_tuned_rerank(self.cfg, self.scope)
        return RerankRetriever(
            base or self.hybrid(), self.texts, self.scorer(p.model), p.depth, p.alpha
        )

    def best(self) -> Retriever:
        """The retriever chosen on validation queries: hybrid, plus re-ranking if it helped."""
        p = load_tuned_rerank(self.cfg, self.scope)
        return self.rerank(p) if p.enabled else self.hybrid()

    def graphrag(self, params: GraphCfg | None = None) -> Any:
        from patentrag.retrieval.graphrag import GraphRagRetriever

        apps = {normalise_applicant(a) for p in self.patents.values() for a in p.applicants}
        return GraphRagRetriever(
            self.vector,
            self.matrix,
            self.graph_queries,
            params or load_tuned_params(self.cfg, self.scope),
            apps,
            self.bm25_index,
        )
