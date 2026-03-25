.PHONY: help install sync dev ingest api run ui build up down up-prod down-prod ingest-prod milvus-up milvus-down test lint format

help:
	@echo 'Targets:'
	@echo '  install      Sync runtime dependencies only'
	@echo '  sync         Sync all dependencies including dev'
	@echo '  dev          Start LangGraph + frontend dev servers in parallel (:2024 + :3000)'
	@echo '  ingest       Ingest aapl_10k.json into Milvus Lite (./milvus.db) — stop stack first'
	@echo '  api          Start FastAPI upload service on :8080'
	@echo '  run          Start LangGraph dev server only (:2024)'
	@echo '  ui           Start frontend dev server only (:3000)'
	@echo '  build        Build LangGraph server Docker image (research-agent:latest)'
	@echo '  up           Start dev stack (Milvus Lite) — requires prior make ingest'
	@echo '  down         Stop dev stack'
	@echo '  up-prod      Start prod stack (full Milvus standalone + BM25 hybrid search)'
	@echo '  down-prod    Stop prod stack'
	@echo '  ingest-prod  Ingest aapl_10k.json into running prod Milvus (stack must be up)'
	@echo '  milvus-up    Start Milvus only (etcd + minio + standalone + attu :8888)'
	@echo '  milvus-down  Stop Milvus only'
	@echo '  test         Run unit tests'
	@echo '  lint         Run Ruff checks'
	@echo '  format       Format with Ruff'

install:
	uv sync --no-dev

sync:
	uv sync

dev:
	overmind start -f Procfile.dev

ingest:
	uv run python -m agent.ingest --data ./aapl_10k.json $(ARGS)

api:
	uv run uvicorn api.main:app --reload --port 8080

run:
	uv run langgraph dev

ui-install:
	cd src/agent-chat-ui && bun install

ui:
	cd src/agent-chat-ui && bun run dev

build:
	uv run langgraph build -t research-agent:latest

up:
	docker compose -f docker-compose.yml up -d

down:
	docker compose -f docker-compose.yml down

up-prod:
	docker compose -f docker-compose.prod.yml up -d

down-prod:
	docker compose -f docker-compose.prod.yml down

ingest-prod:
	APP_MILVUS_URI=http://localhost:19530 uv run python -m agent.ingest --data ./aapl_10k.json $(ARGS)

milvus-up:
	docker compose -f docker-compose.prod.yml up -d etcd minio milvus-standalone attu

milvus-down:
	docker compose -f docker-compose.prod.yml stop etcd minio milvus-standalone attu

test:
	uv run python -m pytest tests/ -q

lint:
	uv run python -m ruff check src tests

format:
	uv run python -m ruff format src tests
