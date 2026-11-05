"""GraphRAG: classify -> traverse -> re-rank, with provenance for every result.

Input is the query text only. Steps:
  a. seeds: top-N patents by vector search, or by the hybrid (vector plus BM25) score when
     `seed_source="hybrid"`; vote over their CPC symbols -> ranked CPC nodes.
  b. candidates: the vector top-`pool_n` and BM25 top-`bm25_pool` patents (so the graph can
     only add candidates, never remove the text retrievers' own results), plus patents in the
     CPC subtrees, citation neighbours of the seeds (both directions, up to H hops) and the
     seeds' family members.
  c. re-rank: weighted sum of normalised vector similarity, normalised BM25 score and graph
     features. With all graph weights at zero this ranks exactly like `HybridRetriever`.
  d. provenance: why each candidate was retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from patentrag.config import GraphCfg
from patentrag.graph.queries import GraphQueries
from patentrag.graph.schema import cpc_ancestor, normalise_applicant
from patentrag.index.bm25 import Bm25Index
from patentrag.index.vector_store import ChunkMatrix
from patentrag.retrieval.base import NO_EXCLUSION, Exclusion, Hit
from patentrag.retrieval.classify import vote_cpc
from patentrag.retrieval.hybrid import LexicalVectorScores
from patentrag.retrieval.rerank import Weights, citation_proximity, cpc_overlap, final_score
from patentrag.retrieval.vector import VectorRetriever


@dataclass
class Candidate:
    pub_id: str
    why: list[str] = field(default_factory=list)
    hop: int | None = None
    family: bool = False
    vote: float = 0.0  # similarity-weighted citation votes from the seeds (see candidates())


@dataclass
class GraphState:
    """Intermediate state of one GraphRAG query, so each step can be run and traced alone."""

    query: str
    qvec: np.ndarray
    excl: Exclusion
    seeds: list[Hit]
    cpc_ranked: list[tuple[str, float]]
    seed_cpcs: dict[str, list[str]]
    cands: dict[str, Candidate] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    final_excl: Exclusion = NO_EXCLUSION
    best: np.ndarray | None = None  # exact best-chunk similarity per patent (matrix order)
    arg: np.ndarray | None = None  # index of that best chunk
    lex: LexicalVectorScores | None = None  # raw and pool-normalised vector and BM25 scores


class GraphRagRetriever:
    name = "graphrag"

    def __init__(
        self,
        vector: VectorRetriever,
        matrix: ChunkMatrix,
        queries: GraphQueries,
        params: GraphCfg,
        applicants: set[str] | None = None,
        bm25: Bm25Index | None = None,
    ):
        self.vector = vector
        self.matrix = matrix
        self.q = queries
        self.p = params
        self.applicants = applicants or set()
        self.bm25 = bm25

    # -- step a ---------------------------------------------------------------------------
    def classify(self, seeds: list[Hit]) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
        cpcs = self.q.patent_cpcs([h.pub_id for h in seeds])
        ranked = vote_cpc(
            [(h.pub_id, h.score) for h in seeds], cpcs, self.p.cpc_level, self.p.cpc_top
        )
        return ranked, cpcs

    # -- step b ---------------------------------------------------------------------------
    def candidates(
        self, seeds: list[Hit], cpc_nodes: list[str], excl: Exclusion
    ) -> dict[str, Candidate]:
        cands: dict[str, Candidate] = {
            h.pub_id: Candidate(h.pub_id, ["vector-seed"]) for h in seeds
        }
        seed_ids = [h.pub_id for h in seeds]
        if self.p.use_cpc and cpc_nodes:
            for row in self.q.cpc_subtree_patents(cpc_nodes, self.p.cpc_depth, excl):
                c = cands.setdefault(row["pub_id"], Candidate(row["pub_id"]))
                tag = f"cpc:{row['via']}"
                if tag not in c.why:
                    c.why.append(tag)
        if self.p.use_cite and self.p.citation_hops > 0:
            # Citation vote: a candidate that similar seeds cite (weight 1) or that cites similar
            # seeds (weight 0.5) gathers the seeds' normalised similarity. Co-citation by several
            # similar patents is strong evidence of relevance.
            sims = [h.score for h in seeds]
            lo, hi = (min(sims), max(sims)) if sims else (0.0, 1.0)
            seed_w = {h.pub_id: ((h.score - lo) / (hi - lo) if hi > lo else 1.0) for h in seeds}
            for row in self.q.citation_hop(seed_ids, excl):
                c = cands.setdefault(row["dst"], Candidate(row["dst"]))
                dir_w = 1.0 if row["direction"] == "cites" else 0.5
                c.vote += dir_w * (0.1 + seed_w.get(row["src"], 0.0))
            nbrs = self.q.citation_neighbours(seed_ids, self.p.citation_hops, excl)
            for pid, info in nbrs.items():
                c = cands.setdefault(pid, Candidate(pid))
                c.hop = info["hop"]
                c.why.append(f"citation:{info['direction']}:{info['via']}:hop{info['hop']}")
        if self.p.use_family:
            for row in self.q.family_members(seed_ids, excl):
                c = cands.setdefault(row["dst"], Candidate(row["dst"]))
                c.family = True
                c.why.append(f"family:{row['src']}")
        for pid in list(cands):
            if pid in excl.exclude:
                del cands[pid]
        return cands

    # -- staged API (used by the agent so every step can be traced) ---------------------------
    def step_classify(
        self, query: str, excl: Exclusion = NO_EXCLUSION, extra_cpc: list[str] | None = None
    ) -> GraphState:
        """Embed the query, take vector seeds and vote for CPC nodes.

        `extra_cpc` are optional LLM-proposed CPC nodes at the voting level; the vote works
        without them and they only add nodes to the traversal set.
        """
        qvec = self.vector.embed_query(query)
        graph_excl = NO_EXCLUSION if excl.final_only else excl
        best, arg = self.matrix.patent_sims(qvec, self.vector.sections)
        lex = self.lexical_scores(query, best, graph_excl)
        if self.p.seed_source == "hybrid" and self.bm25 is not None:
            fused = sorted(
                ((p, lex.fused(p, self.p.w_sim, self.p.w_bm25)) for p in lex.pool),
                key=lambda t: (-t[1], t[0]),
            )
            seeds = [Hit(p, s, ["hybrid-seed"]) for p, s in fused[: self.p.seeds_n]]
        else:
            seeds = self.vector.search_vector(qvec, self.p.seeds_n, graph_excl)
        cpc_ranked, seed_cpcs = self.classify(seeds)
        for node in extra_cpc or []:
            if node not in dict(cpc_ranked):
                cpc_ranked.append((node, 0.0))
        st = GraphState(query, qvec, graph_excl, seeds, cpc_ranked, seed_cpcs, final_excl=excl)
        st.best, st.arg, st.lex = best, arg, lex
        return st

    def lexical_scores(self, query: str, best: np.ndarray, excl: Exclusion) -> LexicalVectorScores:
        """Vector and (when configured) BM25 scores, normalised on the text pool."""
        bm25 = self.bm25 if self.p.bm25_pool > 0 or self.p.w_bm25 > 0 else None
        return LexicalVectorScores.compute(
            self.matrix, best, bm25, query, excl, self.p.pool_n, self.p.bm25_pool
        )

    def step_traverse(self, st: GraphState) -> GraphState:
        st.cands = self.candidates(st.seeds, [c for c, _ in st.cpc_ranked], st.excl)
        if st.best is None or st.lex is None:
            st.best, st.arg = self.matrix.patent_sims(st.qvec, self.vector.sections)
            st.lex = self.lexical_scores(st.query, st.best, st.excl)
        self.add_text_pool(st)
        return st

    def add_text_pool(self, st: GraphState) -> None:
        """Add the vector top-`pool_n` (tag "vector") and BM25 top-`bm25_pool` (tag "bm25")."""
        if st.lex is None:
            return
        vec_n = min(self.p.pool_n, len(st.lex.pool))
        for n, pid in enumerate(st.lex.pool):
            tag = "vector" if n < vec_n else "bm25"
            c = st.cands.setdefault(pid, Candidate(pid))
            if not any(t in c.why for t in (tag, f"{tag}-seed", "hybrid-seed")):
                c.why.insert(0, tag)

    def step_rerank(self, st: GraphState, k: int) -> list[Hit]:
        votes = {c: w for c, w in st.cpc_ranked if w > 0}
        if st.best is None or st.arg is None or st.lex is None:
            st.best, st.arg = self.matrix.patent_sims(st.qvec, self.vector.sections)
            st.lex = self.lexical_scores(st.query, st.best, st.excl)
        arg, lex = st.arg, st.lex
        w = Weights(self.p.w_sim, self.p.w_cpc, self.p.w_cite, self.p.w_family, self.p.w_applicant)
        cand_cpcs = dict(st.seed_cpcs)
        missing = [p for p in st.cands if p not in cand_cpcs]
        cand_cpcs.update(self.q.patent_cpcs(missing))
        mentioned = self._mentioned_applicants(st.query)
        cand_apps = self.q.patent_applicants(list(st.cands)) if mentioned else {}

        if not lex.pool:
            lex.normalise_on(list(st.cands))
        max_vote = max((c.vote for c in st.cands.values()), default=0.0)
        scored: list[tuple[str, float, Candidate, dict[str, float]]] = []
        for pid, c in st.cands.items():
            # min-max normalised with the text pool's range: e5 cosines occupy a narrow band,
            # so raw values would be swamped by the graph features
            sim = lex.sim_n(pid)
            bm = lex.bm25_n(pid)
            overlap = cpc_overlap(cand_cpcs.get(pid, []), votes, self.p.cpc_level) if votes else 0.0
            if c.vote > 0 and max_vote > 0:
                cite = c.vote / max_vote
            else:
                cite = 0.3 * citation_proximity(c.hop) if (c.hop or 0) >= 2 else 0.0
            app = bool(mentioned & {normalise_applicant(a) for a in cand_apps.get(pid, [])})
            if self.p.use_rerank:
                feat = {"sim": sim, "bm25": bm, "cpc": overlap, "cite": cite, "app": float(app)}
                score = final_score(sim, overlap, cite, c.family, app, w) + self.p.w_bm25 * bm
            else:
                # traversal order only: graph features, no embedding similarity as a weight
                feat = {"cpc": overlap, "cite": cite, "app": float(app)}
                score = final_score(0.0, overlap, cite, c.family, app, w)
                score += 1e-3 * sim  # tie-break only
            scored.append((pid, score, c, feat))
        scored.sort(key=lambda t: (-t[1], t[0]))
        scored = [t for t in scored if t[0] not in st.final_excl.exclude]
        scored = scored[: self.p.max_candidates]
        hits = []
        for pid, score, c, feat in scored[:k]:
            i = self.matrix.pub_index.get(pid)
            chunk = self.matrix.texts[int(arg[i])] if i is not None and arg[i] >= 0 else ""
            hits.append(Hit(pid, float(score), c.why, {"features": feat, "best_chunk": chunk}))
        st.hits = hits
        return hits

    # -- steps c, d -------------------------------------------------------------------------
    def search(self, query: str, k: int, excl: Exclusion = NO_EXCLUSION) -> list[Hit]:
        return self.search_detailed(query, k, excl)["hits"]

    def search_detailed(self, query: str, k: int, excl: Exclusion = NO_EXCLUSION) -> dict[str, Any]:
        st = self.step_classify(query, excl)
        self.step_traverse(st)
        hits = self.step_rerank(st, k)
        return {
            "hits": hits,
            "cpc": st.cpc_ranked,
            "seeds": [h.pub_id for h in st.seeds],
            "n_candidates": len(st.cands),
        }

    def _mentioned_applicants(self, query: str) -> set[str]:
        if not self.applicants:
            return set()
        upper = re.sub(r"\s+", " ", query.upper())
        return {a for a in self.applicants if len(a) >= 4 and a in upper}


__all__ = ["GraphRagRetriever", "cpc_ancestor"]
