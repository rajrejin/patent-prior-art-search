# Known issues

| # | Issue | Status |
|---|---|---|
| K1 | An agent run on the laptop CPU takes minutes (local 3B LLM for planning, chart and judging). The evaluation therefore ran on a Colab T4 GPU, where a retrieval-mode run takes seconds and a full run about a minute. | Accepted; GPU recommended for batch use |
| K2 | The 3B planner sometimes chooses pointless actions (for example reading a patent twice). They are replaced by the fallback policy and counted; the share of planner actions used as chosen is reported with the results. | Mitigated and measured |
| K3 | Cited documents from US, JP, CN and KR often lack claims in OPS, so claim chunks and first-claim evidence exist mainly for EP and WO documents. | Open, data limitation |
| K4 | OPS answers with HTTP 403 `ThrottlingControlQuota` when a service budget is exceeded and sends `Retry-After` in milliseconds. The client paces requests from the throttling header, but long fetches still wait out occasional penalties. | Mitigated |
| K5 | The test split was used once by the first design before this one was made (see `docs/DECISIONS.md`, D24). Settings were chosen on validation queries only, but the test split is not completely unseen. | Disclosed |
| K6 | CPC hierarchy is an approximation (section, class, subclass, group, symbol); the real subgroup nesting is not available from OPS. | Open, by design |
| K7 | Windows: the venv is Python 3.13 while CI and Docker use 3.11. | Accepted |
| K8 | On this Windows machine an Application Control policy blocked numpy 2.5.4 `_mt19937.pyd` after a restart; numpy is pinned below 2.4 (2.3.5 loads). | Mitigated |
| K9 | Results are copied from the GPU notebook as compact, checksummed text (`scripts/export_results.py`): per-query score tables stay on the VM; the failure analysis and tables are computed there before export. | Accepted |
| K10 | The first API call after start-up loads the embedding model and all chunk vectors (about a minute on CPU); later searches take a few seconds. | Accepted |
| K11 | Judges decide whether one sentence of retrieved text states a feature; this is not legal disclosure, and both judges are small models. Their reliability is measured (agreement, negative control, independent 7B audit) but not against human labels. | Open |
| K12 | The llama.cpp server behind Ollama keeps a prompt cache of up to 8 GiB in host RAM, more than a 12 GB Colab VM can spare next to Neo4j and the evaluation; it got the server killed twice. `scripts/ollama_watchdog.sh` unloads the model when the server passes about 6 GB, and LLM calls are retried across the reload. A cache-size setting, if Ollama exposes one, would be the cleaner fix. | Mitigated |
