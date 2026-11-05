#!/usr/bin/env bash
# One-time setup of a fresh Colab GPU VM: the Python package, Neo4j 5.26 (Java 17) and Ollama
# with the agent's model. Writes a VM-local .env with a random Neo4j password.
set -e
cd "$(dirname "$0")/.."
APP=$(pwd)
pip install -q -e ".[ml,serve]" pytest pytest-cov
apt-get -qq update && apt-get -qq install -y openjdk-17-jre-headless zstd > /dev/null
cd /content
[ -d neo4j-community-5.26.0 ] || (wget -q https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz && tar xzf neo4j-community-5.26.0-unix.tar.gz)
PW=$(python -c "import secrets; print(secrets.token_hex(12))")
cd neo4j-community-5.26.0
cat >> conf/neo4j.conf <<EOF
server.memory.heap.initial_size=1g
server.memory.heap.max_size=1g
server.memory.pagecache.size=512m
EOF
bin/neo4j-admin dbms set-initial-password "$PW" > /dev/null
nohup bin/neo4j console > /content/neo4j.log 2>&1 &
for _ in $(seq 1 60); do (echo > /dev/tcp/127.0.0.1/7687) 2>/dev/null && break; sleep 2; done
curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
# two requests at a time; the watchdog keeps the server's prompt cache within the VM's memory
OLLAMA_NUM_PARALLEL=2 nohup ollama serve > /content/ollama.log 2>&1 &
nohup bash "$APP/scripts/ollama_watchdog.sh" > /dev/null 2>&1 &
sleep 8
ollama pull qwen2.5:3b-instruct > /dev/null 2>&1
cd "$APP"
printf 'NEO4J_URI=bolt://localhost:7687\nNEO4J_USER=neo4j\nNEO4J_PASSWORD=%s\nOLLAMA_HOST=http://localhost:11434\n' "$PW" > .env
echo "$(date -u +%T) SETUP DONE"
