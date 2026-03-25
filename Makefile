.PHONY: help install dev ingest api run test lint format

help:
	@echo 'Targets:'
	@echo '  install      Sync runtime dependencies with uv'
	@echo '  dev          Sync project + dev dependencies with uv'
	@echo '  ingest       Ingest aapl_10k.json into Milvus Lite (./milvus.db)'
	@echo '  api          Start FastAPI upload service on :8080'
	@echo '  run          Start the LangGraph dev server'
	@echo '  test         Run unit tests'
	@echo '  lint         Run Ruff checks'
	@echo '  format       Format with Ruff'

install:
	uv sync --no-dev

dev:
	uv sync

ingest:
	uv run python -m agent.ingest --data ./aapl_10k.json $(ARGS)

api:
	uv run uvicorn api.main:app --reload --port 8080

run:
	uv run langgraph dev

test:
	uv run python -m pytest tests/ -q

lint:
	uv run python -m ruff check src tests

format:
	uv run python -m ruff format src tests
