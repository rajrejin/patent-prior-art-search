# Evaluation protocol

How retrieval, the agent and the agent's claim charts are evaluated. Result values live in `results/`; this file names the files and contains no numbers.

## 1. Methods compared

| Method | Description |
|---|---|
| `bm25` | Lexical ranking (`rank_bm25`) over title, abstract and claims |
| `vector` | `intfloat/multilingual-e5-small` chunk embeddings, patent score = best chunk |
| `hybrid` | Weighted sum of min-max normalised vector and BM25 scores over the union of both top-100 lists |
| `rerank` | Hybrid top `depth` re-scored by a cross-encoder (best setting from tuning, evaluated even if it was not enabled for serving) |
| `graphrag` | Hybrid seeds, CPC vote, graph candidates (CPC subtree, citation neighbours, family) and graph features in the re-rank |
| `agent` | The tool-using agent in retrieval mode (no claim chart): its fused ranking over the whole-idea search and the feature searches it chose. With the validation-chosen feature weight of 0 this ranking is exactly the whole-idea search, which is `rerank`, so the reported test comparison scores `rerank` and does not repeat 300 agent runs (`docs/DECISIONS.md`, D20) |

## 2. Corpus and query patents

The corpus is citation-closed (`docs/DECISIONS.md`, D1): sampled EP and WO **seed** publications plus the documents they cite. Query patents are drawn only from seeds. A seed is eligible when it has an abstract and at least three cited documents inside the corpus after the leakage controls.

## 3. Ground truth and queries

* **Query text** = title plus abstract of the query patent. No claims, no numbers, no citation list.
* **Relevant set** = the document keys (country plus number) of the query patent's cited documents that exist in the corpus, minus the query patent's own document and its family.
* Retrieved patents are collapsed to document keys before scoring, so A1 and B1 of one document count once.

## 4. Leakage controls (each has a test)

1. **Self exclusion.** The query patent, every kind of it, is removed from results and cannot be traversed. Test: `test_leakage_controls_on_real_queries`.
2. **Hidden outgoing citations.** Every outgoing `CITES` edge of the query patent is invisible to graph queries, including the agent's `graph_neighbours` tool. This is a query-time Cypher filter; nothing is deleted. Test: `test_hiding_outgoing_edges_removes_them_from_traversal`.
3. **Family exclusion.** Family members are excluded from results and from the relevant set. Test: `test_leakage_controls_on_real_queries`.
4. **Clean query text.** The query contains no publication numbers. Test: same test.

**Demonstration of control 2.** GraphRAG runs once with control 2 switched off (`graphrag_leaky`) on validation queries, with the tuned parameters (`results/eval_<scope>_leak_demo.json`) and with the default parameters, where the citation feature is active (`results/eval_<scope>_leak_demo_default.json`). It only shows why the control is needed and is never used as a result.

## 5. Splits and tuning

Queries are split by a seeded, stratified (publication year by main CPC subclass) procedure into 100 validation and 300 test queries, stored in `results/testset_manifest_<scope>.json` (not committed: it contains patent text). All choices are made on validation queries (`evaluation/tuning.py`, `evaluation/agent_eval.py`), written to `results/tuned_params_<scope>.json`:

1. Hybrid BM25 weight from {0.25, 0.5, 1, 2, 4} (vector weight 1).
2. Graph signals added greedily on top of the chosen hybrid; the start setting ranks exactly like the hybrid, and a signal is kept only if validation nDCG@10 rises.
3. Cross-encoder model ({MiniLM-L6, bge-reranker-base}) by depth ({20, 50}) by blend weight ({1.0, 0.7, 0.4}); enabled for serving and for the agent only if it beats the hybrid.
4. The agent runs once on the validation queries; the feature-search fusion weight is chosen from {0, 0.25, 0.5, 1} on those runs.
5. The NLI threshold is chosen on 12 validation agent reports (balanced accuracy at separating a row's own evidence from a random sentence of another query).

The test queries are then scored once. **Second design round:** a first design (GraphRAG on vector search, free-text reports) was evaluated on the same test split before this design was made, so the test split is not completely unseen; settings were still chosen on validation queries only (`docs/DECISIONS.md`, D24).

## 6. Metrics

Precision and recall at 5, 10, 20; MRR; nDCG at 10 (binary gains). 95 percent percentile bootstrap intervals over queries (1000 resamples, seed 0). Every pair of methods is compared with a paired bootstrap of the per-query difference (difference, interval, bootstrap p-value). Breakdowns by number of relevant documents and publication year. Latency p50 and p95 per query after three warm-up queries; for the agent it is the wall time of a whole run including LLM planning.

## 7. Agent behaviour

From the retrieval runs (`agent_behaviour` in `results/eval_<scope>_test.json`) and the report runs (`results/faithfulness_<scope>.json`): mean steps, planner calls, share of planner actions used as chosen, actions replaced by the fallback policy, guardrail steps, planner outputs that were not valid JSON, feature searches per run, tool usage counts, seconds per run.

## 8. Grounding of the claim charts

The full agent runs on 100 test queries (seeded random sample). A row pairs a feature of the idea with a sentence of a charted patent. Measured on rows, for the first-ranked sentence of every pair and after retries (each pair's last judgement): support rate by the LLM judge, the NLI judge and both, judge agreement and Cohen's kappa; the share of evidence sentences found verbatim in the cited patent; and, separately, the support of the features against the idea. On the returned report: reports with at least one verified row, verified rows and patents per report, and the share of feature and patent pairs that got a verified row.

Judge reliability: a **negative control** judges every row's feature against a random sentence of another query (support must collapse), and an **independent audit** (`patentrag faith-audit`, `results/faithfulness_audit_<scope>.json`) re-judges every pair's final row with `qwen2.5:7b-instruct`, which the agent never uses, separately for verified rows and rows rejected after every try, plus its own negative control.

## 9. Ablations and robustness

Run once on the test queries with the tuned setting (`results/ablations_<scope>_test.json`): BM25 weight sweep (0 = vector only) and BM25 alone; each graph signal alone on top of the hybrid; each cross-encoder at depth 20 and 50; vector chunk aggregation (max, top-3 sum) and abstract-only chunks.

Robustness (`results/robustness_<scope>.json`), BM25, vector, hybrid and (if enabled) the re-ranker: query form (title only, abstract only, abstract plus first independent claim), language (German against English titles of the same patents), corpus size (25, 50, 100 percent of documents kept at query time; removed documents are excluded, not deleted).

## 10. Limitations

* Citations are made by examiners and applicants and contain noise; X, Y and A categories are not distinguished in the relevance labels.
* Incoming citations to the query patent are not used as ground truth.
* The citation-closed corpus is small compared with a real search database and over-represents cited documents, so absolute scores do not transfer to a real search task; comparisons between methods on the same corpus remain valid.
* Judges measure support of a feature by one sentence of retrieved text, not legal disclosure.
* One technology area; one evaluation run per configuration.
