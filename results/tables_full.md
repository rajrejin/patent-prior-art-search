### Comparison (test)

Test queries: 300. Cells: mean [95% bootstrap interval]. Latency is per query on the evaluation VM (results/environment_colab.json); this comparison ran without a GPU, which makes the cross-encoder slow. The agent has no row of its own: with the validation-chosen feature weight of 0 its ranking is its whole-idea search, the re-ranked hybrid (`rerank`).

| method | ndcg@10 | mrr | r@5 | r@10 | r@20 | p@5 | p@10 | latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.245 [0.214, 0.274] | 0.411 [0.362, 0.458] | 0.177 [0.153, 0.201] | 0.246 [0.217, 0.279] | 0.325 [0.293, 0.360] | 0.191 [0.163, 0.217] | 0.133 [0.116, 0.151] | 168 |
| vector | 0.189 [0.162, 0.215] | 0.347 [0.303, 0.392] | 0.133 [0.112, 0.154] | 0.184 [0.161, 0.211] | 0.260 [0.230, 0.289] | 0.146 [0.123, 0.168] | 0.104 [0.089, 0.119] | 306 |
| hybrid | 0.243 [0.214, 0.274] | 0.418 [0.372, 0.466] | 0.171 [0.148, 0.194] | 0.238 [0.208, 0.269] | 0.322 [0.290, 0.357] | 0.187 [0.161, 0.213] | 0.133 [0.116, 0.151] | 464 |
| rerank | 0.246 [0.217, 0.277] | 0.426 [0.379, 0.474] | 0.173 [0.151, 0.196] | 0.240 [0.212, 0.269] | 0.322 [0.290, 0.357] | 0.191 [0.165, 0.217] | 0.134 [0.117, 0.151] | 6513 |
| graphrag | 0.243 [0.214, 0.273] | 0.418 [0.371, 0.466] | 0.172 [0.149, 0.195] | 0.238 [0.208, 0.269] | 0.322 [0.290, 0.357] | 0.188 [0.161, 0.214] | 0.133 [0.116, 0.151] | 512 |

Paired difference (first minus second), 95% interval, bootstrap p-value:

| pair | nDCG@10 difference | interval | p |
|---|---|---|---|
| bm25-vector | +0.056 | [+0.032, +0.080] | 0.001 |
| bm25-hybrid | +0.002 | [-0.015, +0.019] | 0.812 |
| bm25-rerank | -0.002 | [-0.019, +0.017] | 0.934 |
| bm25-graphrag | +0.002 | [-0.015, +0.019] | 0.796 |
| vector-hybrid | -0.055 | [-0.068, -0.041] | 0.001 |
| vector-rerank | -0.058 | [-0.072, -0.044] | 0.001 |
| vector-graphrag | -0.054 | [-0.068, -0.041] | 0.001 |
| hybrid-rerank | -0.003 | [-0.011, +0.004] | 0.406 |
| hybrid-graphrag | +0.000 | [-0.000, +0.001] | 0.304 |
| rerank-graphrag | +0.003 | [-0.004, +0.011] | 0.384 |

### Agent behaviour

Full agent runs (with claim charts) on test queries: 100.

| measure | value |
|---|---|
| mean steps per run | 10.3 |
| planner calls | 953 |
| planner actions used as chosen | 0.609 |
| actions replaced by the fallback policy | 373 |
| steps forced by guardrails | 74 |
| planner outputs that were not valid JSON | 0 |
| mean feature searches per run | 4.00 |
| mean seconds per run (wall time, runs sharing one GPU) | 48.7 |

Tool calls in total: claim_chart 100, extract_features 100, finish 100, graph_neighbours 12, read_patent 215, search_feature 400, search_idea 100.

### Tuning (validation)

**Hybrid** (100 validation queries; vector weight 1, BM25 weight varied). Chosen BM25 weight: 0.5.

| BM25 weight | validation nDCG@10 |
|---|---|
| 0.25 | 0.250 |
| 0.5 | 0.264 |
| 1.0 | 0.261 |
| 2.0 | 0.261 |
| 4.0 | 0.252 |

**Graph signals on top of the hybrid** (greedy; a signal is kept only if it raises validation nDCG@10).

| step | value | validation nDCG@10 | kept |
|---|---|---|---|
| start (= hybrid) | None | 0.264 | yes |
| w_cite | 0.1 | 0.263 | no |
| w_cite | 0.2 | 0.260 | no |
| w_cite | 0.4 | 0.251 | no |
| w_cite | 0.8 | 0.201 | no |
| w_cpc | 0.05 | 0.254 | no |
| w_cpc | 0.1 | 0.252 | no |
| w_cpc | 0.2 | 0.245 | no |
| seeds_n | 5 | 0.264 | no |
| seeds_n | 20 | 0.264 | no |
| w_family | 0.1 | 0.265 | yes |
| w_applicant | 0.1 | 0.265 | no |

**Cross-encoder re-ranking** (hybrid alone: 0.264). Best: cross-encoder/ms-marco-MiniLM-L-6-v2, top 20, alpha 0.4: 0.267; enabled.

| model | depth | alpha | validation nDCG@10 |
|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | 20 | 1.0 | 0.223 |
| ms-marco-MiniLM-L-6-v2 | 20 | 0.7 | 0.251 |
| ms-marco-MiniLM-L-6-v2 | 20 | 0.4 | 0.267 |
| ms-marco-MiniLM-L-6-v2 | 50 | 1.0 | 0.172 |
| ms-marco-MiniLM-L-6-v2 | 50 | 0.7 | 0.232 |
| ms-marco-MiniLM-L-6-v2 | 50 | 0.4 | 0.264 |
| bge-reranker-base | 20 | 1.0 | 0.175 |
| bge-reranker-base | 20 | 0.7 | 0.221 |
| bge-reranker-base | 20 | 0.4 | 0.258 |
| bge-reranker-base | 50 | 1.0 | 0.141 |
| bge-reranker-base | 50 | 0.7 | 0.208 |
| bge-reranker-base | 50 | 0.4 | 0.255 |

**Agent feature-search weight** (100 validation queries; 0 means the ranking of the whole-idea search alone). Chosen: 0.0.

| feature weight | validation nDCG@10 | recall@10 |
|---|---|---|
| 0.0 | 0.267 | 0.265 |
| 0.25 | 0.236 | 0.261 |
| 0.5 | 0.232 | 0.262 |
| 1.0 | 0.223 | 0.244 |

### Leakage demonstration (validation)

GraphRAG with tuned parameters, validation queries: 100.

| run | nDCG@10 | recall@20 |
|---|---|---|
| graphrag | 0.265 [0.216, 0.320] | 0.345 [0.291, 0.404] |
| graphrag_leaky | 0.254 [0.207, 0.308] | 0.330 [0.278, 0.387] |

GraphRAG with default parameters (citation weight 0.2), validation queries: 100.

| run | nDCG@10 | recall@20 |
|---|---|---|
| graphrag | 0.201 [0.155, 0.254] | 0.268 [0.220, 0.318] |
| graphrag_leaky | 0.420 [0.362, 0.478] | 0.496 [0.444, 0.549] |


### Ablations (test)

Split: test, queries: 300.

**hybrid_weight**

| setting | nDCG@10 | recall@10 | MRR |
|---|---|---|---|
| w_bm25=0.0 | 0.189 [0.162, 0.215] | 0.184 [0.161, 0.211] | 0.347 [0.303, 0.392] |
| w_bm25=0.25 | 0.228 [0.199, 0.259] | 0.223 [0.195, 0.252] | 0.399 [0.354, 0.445] |
| w_bm25=0.5 (tuned) | 0.243 [0.214, 0.274] | 0.238 [0.208, 0.269] | 0.418 [0.372, 0.466] |
| w_bm25=1.0 | 0.254 [0.224, 0.284] | 0.253 [0.222, 0.285] | 0.418 [0.373, 0.464] |
| w_bm25=2.0 | 0.259 [0.228, 0.289] | 0.260 [0.230, 0.292] | 0.423 [0.374, 0.469] |
| w_bm25=4.0 | 0.254 [0.222, 0.285] | 0.252 [0.222, 0.284] | 0.423 [0.375, 0.471] |
| BM25 only | 0.245 [0.214, 0.274] | 0.246 [0.217, 0.279] | 0.411 [0.362, 0.458] |

**graph_signals**

| setting | nDCG@10 | recall@10 | MRR |
|---|---|---|---|
| none (= hybrid) | 0.243 [0.214, 0.274] | 0.238 [0.208, 0.269] | 0.418 [0.372, 0.466] |
| citation vote (w 0.2) | 0.239 [0.210, 0.270] | 0.234 [0.206, 0.264] | 0.407 [0.360, 0.456] |
| CPC overlap (w 0.1) | 0.237 [0.208, 0.268] | 0.235 [0.207, 0.265] | 0.406 [0.360, 0.453] |
| CPC subtree candidates + overlap | 0.237 [0.208, 0.268] | 0.235 [0.207, 0.265] | 0.406 [0.360, 0.453] |
| family (w 0.1) | 0.243 [0.214, 0.273] | 0.238 [0.208, 0.269] | 0.418 [0.371, 0.466] |
| all four | 0.232 [0.203, 0.263] | 0.230 [0.202, 0.260] | 0.395 [0.349, 0.444] |

**rerank**

| setting | nDCG@10 | recall@10 | MRR |
|---|---|---|---|
| ms-marco-MiniLM-L-6-v2, top 20, alpha 0.4 | 0.246 [0.217, 0.277] | 0.240 [0.212, 0.269] | 0.426 [0.379, 0.474] |
| ms-marco-MiniLM-L-6-v2, top 50, alpha 0.4 | 0.246 [0.217, 0.276] | 0.242 [0.214, 0.271] | 0.426 [0.379, 0.473] |
| bge-reranker-base, top 20, alpha 0.4 | 0.237 [0.207, 0.268] | 0.232 [0.203, 0.263] | 0.413 [0.366, 0.458] |
| bge-reranker-base, top 50, alpha 0.4 | 0.239 [0.210, 0.270] | 0.234 [0.205, 0.265] | 0.416 [0.370, 0.466] |

**vector**

| setting | nDCG@10 | recall@10 | MRR |
|---|---|---|---|
| aggregation max | 0.189 [0.162, 0.215] | 0.184 [0.161, 0.211] | 0.347 [0.303, 0.392] |
| aggregation top3sum | 0.025 [0.016, 0.034] | 0.022 [0.015, 0.031] | 0.056 [0.035, 0.078] |
| abstract-only chunks | 0.199 [0.173, 0.226] | 0.194 [0.170, 0.220] | 0.364 [0.320, 0.409] |


### Faithfulness of the agent's claim charts (test)

Full agent runs on 100 test queries; model qwen2.5:3b-instruct, NLI threshold 0.1 (chosen on validation reports). A row says "this sentence of the patent states this feature of the idea"; for each feature and charted patent the sentences are tried in embedding-similarity order, up to three.

| chart rows (feature and patent pairs) | first-ranked sentence | after up to 3 tries |
|---|---|---|
| rows judged | 1482 | 1482 |
| supported, LLM judge | 0.062 | 0.063 |
| supported, NLI judge | 0.206 | 0.149 |
| supported, both judges | 0.048 | 0.059 |
| agreement LLM vs NLI (Cohen's kappa) | 0.289 | 0.509 |

Evidence sentences found verbatim in the cited patent: 1.000. Features supported by the idea (both judges): 0.796 of 499.

| returned report (verified rows only) | value |
|---|---|
| reports with at least one verified row | 0.380 |
| mean verified patents per report | 0.58 |
| mean verified rows per report | 0.79 |
| feature and patent pairs with a verified row | 0.053 |
| mean features per idea | 3.97 |

Negative control (each row judged against a random sentence of another query):

| judge | matched evidence | mismatched sentence |
|---|---|---|
| LLM | 0.063 | 0.001 |
| NLI | 0.149 | 0.060 |

Rows in the control: 1482. A judge that tells support from non-support accepts far fewer rows against the unrelated sentence than against the row's own evidence; compare the two columns.

Independent audit by qwen2.5:7b-instruct (not used by the agent):

| rows | audited support rate | rows |
|---|---|---|
| verified (shown to the user) | 0.873 | 79 |
| rejected after every try | 0.016 | 1403 |
| mismatched control | 0.001 | 1482 |

Audited support of the extracted features against the idea: 0.916.

Agent in report mode: 10.3 steps and 49 s per run on average (wall time, runs sharing one GPU); planner actions used as chosen: 0.609.

### Robustness (test)

**Query form** (299 queries), nDCG@10

| form / method | nDCG@10 |
|---|---|
| title_only/bm25 | 0.143 [0.119, 0.168] |
| title_only/vector | 0.108 [0.089, 0.129] |
| title_only/hybrid | 0.147 [0.123, 0.170] |
| title_only/rerank | 0.159 [0.136, 0.182] |
| abstract_only/bm25 | 0.245 [0.216, 0.276] |
| abstract_only/vector | 0.173 [0.148, 0.199] |
| abstract_only/hybrid | 0.238 [0.207, 0.270] |
| abstract_only/rerank | 0.244 [0.215, 0.274] |
| abstract_plus_claim/bm25 | 0.234 [0.205, 0.264] |
| abstract_plus_claim/vector | 0.160 [0.138, 0.184] |
| abstract_plus_claim/hybrid | 0.222 [0.194, 0.251] |
| abstract_plus_claim/rerank | 0.229 [0.201, 0.259] |

**Language** (45 queries with both titles)

| title / method | nDCG@10 |
|---|---|
| title_en/bm25 | 0.146 [0.081, 0.230] |
| title_en/vector | 0.099 [0.048, 0.155] |
| title_en/hybrid | 0.137 [0.081, 0.209] |
| title_en/rerank | 0.150 [0.088, 0.222] |
| title_de/bm25 | 0.017 [0.002, 0.035] |
| title_de/vector | 0.055 [0.016, 0.114] |
| title_de/hybrid | 0.042 [0.007, 0.096] |
| title_de/rerank | 0.052 [0.014, 0.106] |

**Corpus size** (documents kept at query time), nDCG@10

| fraction | documents | queries | bm25 | vector | hybrid | rerank |
|---|---|---|---|---|---|---|
| 0.25 | 1089 | 53 | 0.269 | 0.226 | 0.262 | 0.267 |
| 0.5 | 2178 | 145 | 0.297 | 0.228 | 0.293 | 0.295 |
| 1.0 | 4357 | 300 | 0.245 | 0.189 | 0.243 | 0.246 |
