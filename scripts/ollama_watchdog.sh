#!/usr/bin/env bash
# Keep Ollama's model server inside the VM's memory. The llama.cpp server behind Ollama keeps a
# prompt cache in host RAM with a limit of 8192 MiB ("cache state" lines in the Ollama log). On
# a 12 GB Colab VM, next to Neo4j and the evaluation process, that cache got the server killed
# by the kernel twice, and the running evaluation with it. When the server's resident memory
# passes the limit, this asks Ollama to unload the model; the next request loads it again (a few
# seconds) with an empty cache. Agent runs retry failed LLM calls, so a request caught in the
# unload is repeated, not lost.
# Usage: nohup bash scripts/ollama_watchdog.sh [limit_kb] > /dev/null 2>&1 &
LIMIT=${1:-6000000}
LOG=${WATCHDOG_LOG:-/content/watchdog.log}
while true; do
  rss=$(ps -C llama-server -o rss= | sort -n | tail -1)
  if [ "${rss:-0}" -gt "$LIMIT" ]; then
    for m in $(ollama ps | awk 'NR > 1 {print $1}'); do
      curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null
    done
    echo "$(date -u +%T) unloaded the model at ${rss} kB" >> "$LOG"
    sleep 40
  fi
  sleep 5
done
