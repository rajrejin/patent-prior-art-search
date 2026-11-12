# Decisions

Each entry: decision, alternatives considered, reason.

## D1. Citation-closed corpus instead of a plain EP/WO area sample

* **Decision.** The corpus is two parts. Seeds: EP and WO publications sampled from the technology area (CPC `H01M10` or `B60L58`, seeded sample per month). Cited: every document the seeds cite (any authority) whose bibliographic record has an abstract or claims. Evaluation queries are built only from seed patents (`origin == "seed"`).
* **Evidence.** On the 300-publication June 2020 slice, the seeds cite 1289 distinct documents (4.3 per seed). By authority: JP 412, US 264, CN 207, KR 147, WO 143, DE 63, EP 58. Only 50 (3.9 percent) are EP or WO documents with a known date of 2015 or later (796 of the cited documents carry no date in OPS, so the true share lies between 3.9 and 15.6 percent). The seeds-only corpus contained 0 of 1289 cited documents and gave 0 eligible query patents (Section 7.1 needs at least 3). Source: `results/history/citation_composition_slice.json`, produced from cached responses by `scripts/citation_composition.py` (since removed with the pilot corpus; in git history).
* **Alternatives.** (a) EP/WO only: no usable ground truth. (b) Lower the minimum of 3 relevant documents: too few relevant documents make the metrics unstable. (c) Narrow the CPC area so 6000 documents cover it: the area would still be dominated by JP/US/CN/KR citations, which are outside the EP/WO scope.
* **Reason.** Citation ground truth needs the cited documents to exist in the corpus. The closure keeps the retrieval problem honest (the pool contains many real, on-topic documents that other queries cite) and the coverage is reported. Limitation: the candidate pool is far smaller than a real prior-art database, and cited documents are over-represented compared with a random sample of the area. This is stated in the README and `docs/EVALUATION_PROTOCOL.md`.
* **Consequence.** Seeds carry biblio, claims and INPADOC family. Cited documents carry biblio and claims; their family id comes from the biblio record.

## D2. OPS responses are requested as XML

JSON from OPS is a converted XML with single-element lists collapsing to objects, which makes parsers brittle. XML parsed with `lxml` (namespaces stripped) is regular. Raw responses are cached gzipped as returned.

## D3. Python version

The existing venv uses Python 3.13 (the project needs 3.11 or newer). CI uses 3.11. Kept.

## D4. CPC hierarchy

OPS returns CPC symbols without the indentation-based subgroup tree. The graph uses a five-level approximation: section, class, subclass, main group, full symbol, linked with `CHILD_OF`. Patents are `CLASSIFIED_AS` the full-symbol node.

## D5. Relevance at document level

A1 and B1 of the same EP number share a document key (country plus number). Relevance is defined on document keys. Retrieved patents are collapsed to document keys (first occurrence) before scoring, so two kinds of one document cannot both count.

## D6. Windows tooling

`make` is not installed. `scripts/*.ps1` wrappers mirror the Makefile. The `patentrag` CLI is the primary interface.

## D7. Neo4j in Docker

`neo4j:5.26-community`, heap 1 GB, page cache 512 MB, container limit 2200 MB, bound to 127.0.0.1, password taken from `.env` through `--env-file`. Ollama is not containerised (host install keeps RAM low).

## D8. Exclusion and leakage design

`Exclusion(exclude, hide_outgoing_from)` is a query-time object passed to every retriever and every Cypher query. No data is deleted. Traversal never passes through excluded nodes, and a CITES edge whose source is in `hide_outgoing_from` is invisible.

## D9. Re-rank features and normalisation

GraphRAG's score is `w_sim*sim + w_bm25*bm25 + w_cpc*cpc_overlap + w_cite*cite + w_family*family + w_applicant*applicant`. `sim` is the best-chunk cosine of the candidate, computed exactly from an in-memory matrix of all chunk vectors; `bm25` is the BM25 score of the whole patent. Both are min-max normalised with the range of the text pool (vector top 100 plus BM25 top 100), not of the candidate set, so the weights mean the same thing whatever the graph adds, and with every graph signal off GraphRAG ranks exactly like `hybrid` (tested). `cite` is a citation vote: each seed that cites the candidate adds its normalised score (weight 1), each seed cited by it adds half, normalised by the largest vote.

## D10. Tuning procedure

All settings are chosen on the 100 validation queries, in stages, each starting from the previous stage's choice (`evaluation/tuning.py`, `evaluation/agent_eval.py`): hybrid BM25 weight; graph signals added greedily on top of the hybrid (kept only if validation nDCG@10 rises); cross-encoder model, depth and blend weight (enabled only if it beats the hybrid); the agent's feature-search fusion weight (from one set of agent runs on the validation queries); the NLI threshold (on 12 validation agent reports). Everything is written to `results/tuned_params_<scope>.json` with its search log.

## D11. Compute on a GPU VM, serving on the laptop

The laptop (8 GB RAM, CPU only) serves the system. Batch work runs on a Google Colab VM (`scripts/colab_setup.sh`, `scripts/pipeline.sh`): hundreds of agent runs with LLM planning, chart drafting and judging take hours on a T4 GPU and days on the laptop CPU. A full agent run together with Neo4j and the API container exhausted the laptop's memory once, so the laptop runs agents only interactively. The pipeline has two parts: `retrieval` (no LLM: indexes, graph, tests, tuning, the comparison without the agent, leakage demonstrations, ablations, robustness; runs on a CPU runtime) and `agent` (the LLM parts; needs a GPU), so the first part could run while the account's free GPU quota was used up. Nothing secret is uploaded; the VM gets its own random Neo4j password.

Memory on the VM: processes started from the notebook share one memory group of about 12 GB. Running three model-heavy processes in parallel next to Ollama and Neo4j made the kernel kill Neo4j, so the pipeline runs one model-heavy process at a time, with Neo4j at 1 GB heap and 512 MB page cache (the graph has 4,357 patents). Throughput comes from concurrency inside one process instead: agent runs are evaluated on threads (`PATENTRAG_AGENT_WORKERS`) against an Ollama server that accepts as many parallel requests (`OLLAMA_NUM_PARALLEL`). The llama.cpp server behind Ollama keeps a prompt cache in host RAM of up to 8192 MiB (the "cache state" lines of its log); it grew until the kernel killed the server twice, each time taking the running evaluation step with it. `scripts/ollama_watchdog.sh` now unloads the model when the server passes about 6 GB (the next request reloads it with an empty cache; LLM calls are retried, so none is lost), and parallelism was lowered from three to two. Agent runs are saved one at a time and skipped when a step is rerun, so the reported results were completed by resuming, not by starting over. Concurrency of the reported runs: the validation runs used for tuning ran three at a time; of the 300 test runs, the first 153 ran three at a time and the rest two at a time; of the claim-chart reports, the 12 validation reports and the first 36 test reports ran three at a time and the rest two at a time. Runs share no state, so concurrency changes speed; outputs could differ only through batched GPU arithmetic, which is not bit-identical across batch sizes. The reported agent latencies are wall times while two or three runs shared the GPU.

## D12. Leakage demonstration uses default parameters as well

With tuned parameters the citation weight can be 0, in which case the "leaky" run shows no inflation. The demonstration is therefore also run with the untuned default parameters (citation weight 0.2), where the citation feature is active (`results/eval_full_leak_demo_default.json`). Both are reported.

## D13. First design round (history)

The first design built GraphRAG on vector search (vector seeds, CPC vote, graph traversal, graph features in the re-rank) and wrote free-text reports with quotes. Its test results (`results/history/round1_summary.json`, 300 queries): nDCG@10 BM25 0.245, vector 0.189, GraphRAG 0.167, with BM25 significantly ahead of both. Reports: with quotes copied by the model, 5 percent passed every check after retries; with evidence chosen by sentence id (D15), 29 percent passed and 71 percent still failed. These results motivated D18 to D22.

## D14. GraphRAG always includes the text retrievers' top results

The first GraphRAG built candidates only from a few seeds and their graph neighbourhood, so graph signals could only remove good candidates. Candidates now always include the vector top 100 and the BM25 top 100; the graph adds candidates and features on top.

## D15. Evidence is a sentence of the patent, never generated text

In round 1 the model copied quotes, and only 45 percent were verbatim. Letting it choose a sentence id from a JSON-schema enum instead raised verbatim evidence to 98 percent (`results/history/round1_summary.json`). Since D28 the sentence is chosen by embedding retrieval rather than by the model; either way the evidence is a sentence of the cited patent's own text, and the deterministic check confirms it.

## D16. The report cache stores the checked response

The full `ReportResponse` is cached (keyed by query, model and corpus hash) and returned unchanged on a hit, with `cached=true`, so a cache hit returns exactly what was verified. Evaluation calls with exclusions are never cached.

## D17. One stack for evaluation and serving

`stack.py` builds every retriever from `results/tuned_params_<scope>.json`; the evaluation commands, the agent, the service, the MCP server and the API all use it, so the served configuration is the evaluated one.

## D18. Hybrid retrieval as the base

* **Decision.** The default retriever is a weighted sum of min-max normalised embedding similarity and BM25 score (`retrieval/hybrid.py`); GraphRAG, the re-ranker and the agent all build on it.
* **Why.** In round 1 BM25 alone was the strongest method: patent text is full of exact technical terms that keyword search matches precisely, while embeddings capture paraphrase.
* **Alternatives.** Reciprocal rank fusion (drops score information; used instead to merge the agent's searches, where scores of different queries are not comparable), or vector search alone (the usual RAG default, weaker here).
* **Measured effect:** see the comparison table (`results/eval_full_test.json`) and the BM25-weight ablation (`results/ablations_full_test.json`).

## D19. Cross-encoder re-ranking only if it helps on validation

Two cross-encoders (MiniLM-L6 trained on MS MARCO; bge-reranker-base) by two depths by three blend weights were scored on validation queries. Off-the-shelf cross-encoders are trained on short web questions, while these queries are whole patent abstracts, so a gain was not assumed. The best setting is stored and evaluated on test either way; it is enabled for serving and inside the agent only if it beat the hybrid on validation (`results/tuned_params_full.json`, section `rerank`).

## D20. A tool-using agent with constrained actions

* **Decision.** The agent (`agent/loop.py`) is an LLM loop that chooses one tool per step (search the idea, extract features, search a feature, read a patent, look up graph neighbours, build the claim chart, finish). Each step's JSON schema offers only the tools that make sense at that moment and only ids that exist (Ollama constrained decoding), so actions are always well formed. Semantically pointless actions (a feature already searched, a patent already read) are replaced by a fixed fallback policy; guardrails enforce a step limit and a verified chart before finishing. Every step records whether it came from the planner, the fallback or a guardrail.
* **Why.** A fixed pipeline is not an agent; free-form function calling with a 3B model produces invented tool names and ids. The constrained loop leaves the real decisions to the model and makes its failure modes impossible or measurable.
* **Measured.** Agent behaviour (planner actions used as chosen, replacements, guardrail steps, tool use) on the 100 full test runs (`results/faithfulness_full.json`, `agent_behaviour`) and on the validation runs (`results/tuned_params_full.json`, section `agent`). Fusing the feature searches into the ranking lowered validation nDCG@10 at every weight tried, so the weight is 0 and the agent's ranking is its whole-idea search, the re-ranked hybrid (validation nDCG@10 identical: 0.2672 for both). The test comparison therefore scores that search (`rerank`); 175 retrieval-mode test runs of the agent were lost with a VM before they finished and were not repeated, because they cannot rank differently.

## D21. A claim chart instead of a free-text report

* **Problem.** Round 1 relevance sentences ("this patent is relevant because it involves X") mixed facts about the patent with claims about the idea, so no single text could support them; most failures in round 1 were such sentences.
* **Decision.** The model restates the idea as 2 to 5 features; each feature is then paired with a sentence of each charted patent (D28). Each feature is judged against the idea; each row against its own evidence sentence. Free text that cannot be checked is not generated; the summary and relevance lines are built by code from the verified rows.

## D22. Pairs without verified evidence are left out and counted

For each feature and patent pair, up to three sentences are tried in ranking order; a pair whose sentences are all rejected is left out of the chart, and the count is returned (`removed_rows`) and shown in the UI. A real claim chart also has empty cells: a cited patent rarely discloses every feature, and only title, abstract and first claim are available here. Returning rejected rows with a warning would leave the checking to the user. Metrics are reported for the first-ranked sentence, after retries, and on the returned reports.

## D23. Independent audit with a larger model

The two judges are small models, and the LLM judge is the same model that extracted the features. A larger model that the agent never uses (`qwen2.5:7b-instruct`) re-judges every pair's final row, separately for verified rows and rows rejected after every try, with its own negative control (`patentrag faith-audit`, `results/faithfulness_audit_full.json`). Verified rows should be supported far more often than rejected rows if the verification works.

## D24. Second design round on the same test split (disclosure)

Round 1 (D13) was evaluated on the same 300 test queries before this design was made, and its results motivated the redesign. All settings of this round were chosen on validation queries only and the test queries were scored once, but the test split is no longer completely unseen, so small differences on test should be read with that in mind. A fresh test split from newly fetched seeds would remove this caveat (`docs/KNOWN_ISSUES.md`, K5).

## D25. Agent runs as background jobs in the API

On the laptop CPU one agent run can take longer than any reasonable HTTP timeout (a smoke test hit the 15-minute limit while the laptop was under memory pressure). `/v1/report/jobs` starts the run in the background and `/v1/report/jobs/{id}` returns its state and result; the UI and `scripts/e2e_smoke.py` use it. The synchronous `/v1/report` stays for clients that can wait, with a configurable timeout.

## D26. Results leave the notebook as compact, checksummed text

The notebook cannot download files without interaction. `scripts/export_results.py` prints each result file as one line of compact JSON (per-query tables dropped, floats rounded to 4 decimals) with its length and CRC32; `scripts/import_results.py` writes only blocks whose length and CRC match. Per-query tables stay on the VM; the tables and the failure analysis that need them are computed there first.

## D27. Repository cleanup

The one-month pilot corpus (`slice` scope) and its results, the round 1 result files and the scripts of earlier notebook sessions were removed; the round 1 headline numbers are kept in `results/history/round1_summary.json` and the full files remain in git history. The raw EPO cache and the full corpus are kept (gitignored), because fetching them again costs weekly API quota.

## D28. Evidence retrieved by embeddings, not chosen by the LLM (third round for the chart)

* **Problem.** In the first run of this design on the 100 test reports, the 3B model chose each row's evidence sentence by id, and the judges rarely accepted its choices: 39 of 631 rows (6 percent) were supported by both judges, while 395 of 499 features were; only 24 percent of reports kept any verified row (`results/history/round2_llm_chosen_evidence.json`).
* **Decision.** For every feature and charted patent, the patent's sentences are ranked by cosine similarity with the retrieval embedding model (multilingual-e5, `agent/report.py: embedding_ranker`); the best sentence is judged, and a rejected row is retried with the next one, up to three. The judges, their threshold procedure and the checks are unchanged, and the judges never see the similarity scores. This is the sentence-selection step of fact-verification pipelines (retrieve documents, select sentences, verify).
* **Disclosure.** The change was made after seeing those test reports, so for the claim chart this is a third round on the same test split; the retrieval settings were not changed. Trying up to three sentences gives the judges more chances to accept a wrong row, which the negative control and the independent audit measure.
* **Measured effect.** On the same 100 test queries, reports with at least one verified row rose from 24 to 38 percent and verified rows per report from 0.33 to 0.79 (`results/faithfulness_full.json`, section `shipped`). The independent 7B audit finds 87 percent of the verified rows supported, against 1.6 percent of the rows rejected after every try and 0.1 percent of rows paired with an unrelated sentence (`results/faithfulness_audit_full.json`): few cells are filled, and those shown are almost always right.
