PY ?= python
SCOPE ?= full

.PHONY: setup test check lint data pipeline up down serve-api ui

setup:
	$(PY) -m venv .venv
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -e ".[dev,ml,serve]"

test:
	$(PY) -m pytest -rs

check: lint
	$(PY) -m pytest -rs --cov=src/patentrag --cov=api --cov-fail-under=80

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

# fetch from EPO OPS (needs credentials in .env) and build the corpus
data:
	$(PY) -m patentrag.cli fetch --scope $(SCOPE)
	$(PY) -m patentrag.cli build-corpus --scope $(SCOPE)

# indexes, graph, tuning, evaluation, agent reports and audit (a GPU is strongly recommended)
pipeline:
	bash scripts/pipeline.sh

up:
	docker compose --env-file .env -f docker/docker-compose.yml up -d neo4j

down:
	docker compose --env-file .env -f docker/docker-compose.yml down

serve-api:
	$(PY) -m patentrag.cli serve-api

ui:
	streamlit run app/streamlit_app.py
