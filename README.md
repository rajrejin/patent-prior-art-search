# Patent prior-art search with an LLM agent

You describe an invention idea in a few sentences. An agent searches real patents for prior art
and returns a **claim chart**: each feature of your idea next to the sentence in an existing
patent that already describes it. Every row is checked by two independent judges before it is
shown, and rows that fail are dropped rather than displayed.

The corpus is 4,357 real patent documents on rechargeable batteries and vehicle battery systems
(CPC H01M10 and B60L58), fetched from the EPO Open Patent Services API. The LLM is a 3B model
(Qwen2.5) running locally through Ollama. You can use the system as an MCP server from Claude
Desktop, through a FastAPI service, or in a Streamlit app.

## What I found

I started this project expecting GraphRAG (retrieval helped by a Neo4j knowledge graph of
citations, patent families and classifications) to clearly beat plain retrieval. It did not.

Retrieval quality on 300 held-out test queries, nDCG@10:

| Method | nDCG@10 | Median latency |
|---|---|---|
| BM25 (keyword search) | 0.245 | 0.2 s |
| Vector search (e5 embeddings) | 0.189 | 0.3 s |
| Hybrid (BM25 + vectors) | 0.243 | 0.5 s |
| Hybrid + cross-encoder rerank | 0.246 | 6.5 s |
| GraphRAG (hybrid + graph signals) | 0.243 | 0.5 s |

- BM25, hybrid, rerank and GraphRAG are statistically indistinguishable. Only pure vector search
  is clearly worse. Patent language is so specific that keyword matching is hard to beat.
- The graph signals added nothing on top of the hybrid once I removed the leakage. This matters:
  the test queries are patents and the "correct answers" are the documents they cite, so a graph
  that contains those citation edges can simply look the answer up. With that leak left open and a
  default citation weight, GraphRAG scored 0.42 on validation instead of 0.20. The leakage controls and a demonstration of the effect are documented in
  [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
- The agent is cautious to a fault. Over 100 full runs only 38% of reports contained at least one
  verified row (0.8 rows per report on average). The rows that do survive are good: an
  independent audit with a larger model that the agent never used judged 87% of them as supported.
- A 3B model is a weak planner. Only 61% of its chosen actions were used as they were. The rest
  were replaced by a fallback policy, and every replacement is logged.

All tables (tuning, ablations, robustness, failure analysis) are in
[results/tables_full.md](results/tables_full.md).

## How it works

```
idea ──► agent (Qwen2.5 3B, JSON-schema tool calls, guardrails, full trace)
           ├─ search_idea / search_feature ──► hybrid retrieval (BM25 + e5, optional cross-encoder)
           ├─ read_patent                  ──► patent text
           ├─ graph_neighbours             ──► Neo4j: citations, family, CPC
           └─ claim_chart ──► features ──► best sentence per feature and patent ──► two judges
```

- **The model never writes patent numbers or quotes.** Evidence is always a real sentence pulled
  from the patent by embedding similarity, so the chart cannot contain an invented citation.
- **Two judges per row.** An LLM judge and an NLI cross-encoder both have to agree that the
  sentence states the feature. If not, the next best sentence is tried, up to three times.
- **Constrained tool calls.** At every step the JSON schema only offers tools that make sense at
  that moment and only patent ids that exist. There were no malformed tool calls in 953 planner
  calls.
- **Everything is tuned on validation queries**, separate from the 300 test queries.

## Running it

You need Python 3.11+, Docker (for Neo4j), [Ollama](https://ollama.com), and free EPO OPS
credentials.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml,serve]"
cp .env.example .env                                  # OPS credentials and a Neo4j password
docker compose --env-file .env -f docker/docker-compose.yml up -d neo4j
ollama pull qwen2.5:3b-instruct

python -m patentrag.cli fetch --scope full            # resumable, respects the OPS weekly quota
python -m patentrag.cli build-corpus --scope full
python -m patentrag.cli build-index --scope full
python -m patentrag.cli build-graph --scope full --reset

python -m patentrag.cli report "A battery pack with a cooling plate between the cells ..."
python -m patentrag.cli serve-api                     # http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py
```

The full evaluation (`bash scripts/pipeline.sh`) wants a GPU. I ran it on a free Colab T4 using
`colab/patentrag_colab.ipynb`. On my 8 GB laptop a single agent run takes minutes on CPU.

Tests and lint: `make check`.

### As an MCP server in Claude Desktop

```json
{
  "mcpServers": {
    "patent-prior-art": {
      "command": "C:\\path\\to\\patent-rag\\.venv\\Scripts\\python.exe",
      "args": ["-m", "patentrag.cli", "serve-mcp"],
      "env": { "NEO4J_PASSWORD": "<your Neo4j password>" }
    }
  }
}
```

Tools: `search_patents`, `run_prior_art_agent`, `get_agent_trace`.

### API in Docker

```bash
docker compose --env-file .env -f docker/docker-compose.yml --profile app up -d --build
```

## Limitations

- Ground truth is citations. They are noisy, and a relevant patent that nobody cited counts as
  wrong.
- One technology area and a small corpus, so the absolute scores say little about real patent
  search. The comparison between methods is the useful part.
- The test split was looked at once by an earlier design of this project. Everything in the
  current design was tuned on validation queries only, but the test set is not perfectly unseen.
- The judges decide whether a sentence states a feature. That is not a legal opinion on novelty,
  and this is not a freedom-to-operate tool.

More: [architecture](docs/ARCHITECTURE.md), [evaluation protocol](docs/EVALUATION_PROTOCOL.md),
[design decisions](docs/DECISIONS.md), [known issues](docs/KNOWN_ISSUES.md).

## Data

EPO Open Patent Services, used under the EPO terms. Patent text is not redistributed here: raw
responses, the corpus and any result file containing patent text are gitignored.
