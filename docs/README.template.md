# patentrag: agentic RAG for patent prior-art search

A tool-using LLM agent that searches real European patents for prior art and returns a **verified claim chart**: every feature of an invention idea mapped to an evidence sentence in a retrieved patent, with every row checked by two judges. Built on 4,357 real documents from the EPO Open Patent Services API (rechargeable batteries and vehicle battery systems, CPC H01M10 and B60L58), with hybrid retrieval (BM25 plus multilingual embeddings, optional cross-encoder re-ranking), a Neo4j knowledge graph, a local LLM (qwen2.5 3B through Ollama), and an evaluation with leakage controls, bootstrap intervals, ablations and an independent audit.

Exposed as MCP tools (for Claude Desktop), a FastAPI service and a Streamlit app.

## Headline results

{{HEADLINE}}

All numbers on this page are copied from files in `results/` by `scripts/build_readme.py`.

## How it works

```
idea ──► agent (qwen2.5 3B, JSON-schema tool calls, guardrails, trace)
           ├─ search_idea / search_feature ──► hybrid retrieval: BM25 + e5 embeddings (+ cross-encoder)
           ├─ read_patent                   ──► patent text
           ├─ graph_neighbours              ──► Neo4j: citations, family, CPC (leakage controls applied)
           └─ claim_chart ──► features ──► best sentence per feature and patent ──► checks ──► verified chart
                                                       evidence from the cited patent? LLM judge? NLI judge?
```

* **Retrieval.** `hybrid` adds min-max normalised embedding similarity and BM25 scores (weight chosen on validation queries). A cross-encoder re-ranks the top results if that helped on validation. `graphrag` adds graph candidates and features (citation vote, CPC overlap, family) on top of the hybrid and is kept as a measured alternative.
* **Agent.** The planner LLM chooses one tool per step. Its JSON schema offers only the tools allowed at that moment and only ids that exist, so every action is well formed; pointless actions are replaced by a fallback policy and recorded; guardrails enforce a step limit and a verified chart before finishing. Feature searches can be fused into the ranking by weighted reciprocal rank, with the weight tuned on validation queries; whatever the weight, they supply candidates and evidence for the claim chart.
* **Claim chart.** The model never writes patent numbers or quotes: for each feature and charted patent the evidence is the patent's most similar sentence (embedding retrieval), so it is always real text. Each row is checked deterministically (the sentence belongs to the cited patent) and by an LLM judge and an NLI cross-encoder. A rejected row is retried with the next most similar sentence (up to three); pairs with no verified sentence are left out and counted.
* **Evaluation.** 300 held-out test queries built from real patents (query = title plus abstract, relevant = its cited documents), four leakage controls with tests, every setting tuned on 100 separate validation queries, paired bootstrap comparisons, ablations, robustness, a negative control for the judges and an independent 7B audit.

Diagrams and module details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Protocol: [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md). Design decisions with measured effects: [docs/DECISIONS.md](docs/DECISIONS.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,ml,serve]"         # Linux: .venv/bin/python; or `make setup`
copy .env.example .env                                            # fill in OPS credentials and a Neo4j password
docker compose --env-file .env -f docker/docker-compose.yml up -d neo4j
ollama pull qwen2.5:3b-instruct

.venv\Scripts\python -m patentrag.cli fetch --scope full          # resumable, quota guarded
.venv\Scripts\python -m patentrag.cli build-corpus --scope full
.venv\Scripts\python -m patentrag.cli build-index --scope full
.venv\Scripts\python -m patentrag.cli build-graph --scope full --reset
bash scripts/pipeline.sh                                          # tuning, evaluation, agent runs (GPU recommended)

.venv\Scripts\python -m patentrag.cli report "A battery pack with a cooling plate ..."   # run the agent
.venv\Scripts\python -m patentrag.cli serve-api                   # http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py                                 # UI
```

The reported results were computed on a Google Colab T4 GPU with `colab/patentrag_colab.ipynb` (`scripts/colab_setup.sh`, then `scripts/pipeline.sh`). Quality gate: `scripts/check.ps1` or `make check` (ruff, ruff format, mypy, pytest with an 80 percent coverage gate).

## MCP usage (Claude Desktop)

```json
{
  "mcpServers": {
    "patent-prior-art": {
      "command": "D:\\Projects\\patent-graphrag\\.venv\\Scripts\\python.exe",
      "args": ["-m", "patentrag.cli", "serve-mcp"],
      "env": { "NEO4J_PASSWORD": "<your Neo4j password>" }
    }
  }
}
```

Tools: `search_patents` (query, k, method: auto, hybrid, rerank, graphrag, vector, bm25; results carry provenance), `run_prior_art_agent` (verified claim chart plus every agent step), `get_agent_trace`.

## API and Docker

```bash
docker compose --env-file .env -f docker/docker-compose.yml --profile app up -d --build
```

Endpoints: `/v1/health`, `/v1/ready` (503 with readable reasons when Neo4j, the indexes or the LLM are missing), `/v1/search`, `/v1/report` (runs the agent and waits), `/v1/report/jobs` (submits an agent run in the background; poll `/v1/report/jobs/{id}`), `/v1/trace/{id}`, `/metrics` (Prometheus). The API and UI containers run as a non-root user; the LLM runs on the host through Ollama. `scripts/e2e_smoke.py` exercises a running API end to end.

## Data

* **Source:** EPO Open Patent Services, used under the EPO terms. Raw responses, the corpus and any file containing patent text are not redistributed (`data/` and those result files are gitignored). The client honours the weekly fair-use quota (stops at 80 percent), `Retry-After` and the per-service throttling header.
* **Scope:** seeds are EP and WO publications 2015 to 2024 in CPC H01M10 or B60L58 (7 per month, fixed seed), plus every document they cite, from any office (citation-closed corpus; reason in [docs/DECISIONS.md](docs/DECISIONS.md), D1).
* **Size and coverage:** {{CORPUS}}

## Results

{{TABLES}}

## Limitations

* Citation-based ground truth is noisy (examiner and applicant citations; X, Y and A categories are not distinguished) and recall is capped by what was cited.
* One technology area; the citation-closed corpus is far smaller than a real patent database, so absolute scores do not transfer to a real search task. Comparisons between methods on the same corpus remain valid.
* The test split was used once before, by the first design; this second design was tuned on validation queries only (see [docs/DECISIONS.md](docs/DECISIONS.md), D24).
* The 3B planner is small: most of its actions are used as chosen, but the fallback policy and guardrails do part of the work, and the share is reported rather than hidden.
* Judges measure whether a sentence of retrieved text states a feature, not legal novelty. This is not legal advice and not a freedom-to-operate tool.

## Repository layout

`src/patentrag/` (ingest, graph, index, retrieval, agent, faithfulness, evaluation, service, MCP), `api/`, `app/`, `tests/` (unit, integration on real data, API and MCP contract tests), `scripts/` (pipeline, Colab setup, result export and import, tables, README builder), `docker/`, `docs/`, `results/` (every reported number).
