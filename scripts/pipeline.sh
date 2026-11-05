#!/usr/bin/env bash
# Compute pipeline: indexes, graph, checks, tuning on validation queries, one test evaluation,
# leakage demonstrations, ablations, robustness, the agent's runs and claim-chart reports, an
# end-to-end check of the API, and the independent audit. Needs Neo4j and Ollama running (see
# scripts/colab_setup.sh); a GPU is strongly recommended.
#
#   bash scripts/pipeline.sh            everything
#   bash scripts/pipeline.sh retrieval  only the parts that need no LLM (runs on a CPU machine)
#   bash scripts/pipeline.sh agent      only the LLM parts; needs results/tuned_params_*.json
#   bash scripts/pipeline.sh reports    tests, test comparison and the agent's reports, API check
#                                       and audit, reusing results/tuned_params_*.json
# The agent's ranking is its whole-idea search when the tuned feature weight is 0 (it was), so
# the test comparison scores that search (`rerank`) instead of repeating 300 agent runs.
# Agent runs are saved one at a time (results/agent_*.jsonl) and skipped when a step is run
# again, so after an interruption `bash scripts/pipeline.sh agent` continues where it stopped.
# Progress goes to logs/progress.txt; each step's output to logs/<step>.log.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
export PATENTRAG_MAX_QUERIES=0
export PATENTRAG_AGENT_WORKERS=${PATENTRAG_AGENT_WORKERS:-2}  # concurrent agent runs (needs OLLAMA_NUM_PARALLEL)
PART=${1:-all}
PR="python -m patentrag.cli"
step() {
  local name=$1; shift
  echo "$(date -u +%T) START $name" | tee -a logs/progress.txt
  "$@" > "logs/$name.log" 2>&1
  local rc=$?  # captured before any other command runs
  echo "$(date -u +%T) END $name rc=$rc" | tee -a logs/progress.txt
}

retrieval_studies() {
  step leak_tuned   $PR evaluate --scope full --split val --methods graphrag,graphrag_leaky --name eval_full_leak_demo
  step leak_default $PR evaluate --scope full --split val --methods graphrag,graphrag_leaky --default-params --name eval_full_leak_demo_default
  step ablate_test  $PR ablate --scope full --split test
  step robustness   $PR robustness --scope full
}

api_check() {
  # serve the API as deployed, then one search per method and one agent job over HTTP
  $PR serve-api > logs/api.log 2>&1 &
  local api=$!
  for _ in $(seq 1 90); do curl -sf http://localhost:8000/v1/ready > /dev/null && break; sleep 5; done
  step e2e python scripts/e2e_smoke.py
  kill "$api"
}

step build_index  $PR build-index --scope full
step build_graph  $PR build-graph --scope full --reset

if [ "$PART" = "retrieval" ]; then
  step checks     python -m pytest tests -q -rs -p no:cacheprovider --cov=src/patentrag --cov=api --cov-fail-under=80
  step tune       $PR tune --scope full
  step eval_text  $PR evaluate --scope full --split test --methods bm25,vector,hybrid,rerank,graphrag --name eval_full_test
  retrieval_studies
elif [ "$PART" = "reports" ]; then
  step checks       python -m pytest tests -q -rs -p no:cacheprovider --cov=src/patentrag --cov=api --cov-fail-under=80
  step eval_test    $PR evaluate --scope full --split test --methods bm25,vector,hybrid,rerank,graphrag --name eval_full_test
  step faithfulness $PR faithfulness --scope full --n-test 100 --n-cal 12
elif [ "$PART" = "agent" ]; then
  step tune_agent   $PR tune-agent --scope full
  step eval_test    $PR evaluate --scope full --split test --methods bm25,vector,hybrid,rerank,graphrag --name eval_full_test
  step faithfulness $PR faithfulness --scope full --n-test 100 --n-cal 12
else
  step checks     python -m pytest tests -q -rs -p no:cacheprovider --cov=src/patentrag --cov=api --cov-fail-under=80
  step tune       $PR tune --scope full
  # one model-heavy process at a time: on a 12 GB VM, Ollama, Neo4j and two such processes
  # exceed the notebook's memory group and the kernel kills Neo4j (docs/DECISIONS.md, D11)
  retrieval_studies
  step tune_agent   $PR tune-agent --scope full
  step eval_test    $PR evaluate --scope full --split test --methods bm25,vector,hybrid,rerank,graphrag --name eval_full_test
  step faithfulness $PR faithfulness --scope full --n-test 100 --n-cal 12
fi

step failures     python scripts/failure_analysis.py full
if [ "$PART" != "retrieval" ]; then
  step example    python scripts/example_run.py full
  api_check
  # the audit needs only the LLM: free memory for the 7B judge
  pkill -f "org.neo4j"
  curl -s http://localhost:11434/api/generate -d '{"model":"qwen2.5:3b-instruct","keep_alive":0}' > /dev/null
  step pull_judge ollama pull qwen2.5:7b-instruct
  step audit      $PR faith-audit --scope full --model qwen2.5:7b-instruct
fi
step tables       python scripts/render_tables.py full
echo "$(date -u +%T) PIPELINE $PART DONE" | tee -a logs/progress.txt
