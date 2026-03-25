# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LangGraph-based AI research assistant specialized in analyzing Apple Inc. SEC 10-K filings. Uses hybrid retrieval (keyword search + semantic reranking) over a Milvus vector store.

## Commands

```bash
# Setup
make dev          # uv sync (install all deps including dev)
make install      # uv sync --no-dev (runtime only)

# Run
make run          # langgraph dev server at http://localhost:8000
make api          # FastAPI upload service at http://localhost:8080
make ingest       # python -m agent.ingest --data ./aapl_10k.json

# Test
make test         # pytest tests/ -q
make lint         # ruff check src tests
make format       # ruff format src tests

# Single test
uv run pytest tests/unit_tests/test_configuration.py -q
```

Integration tests (`tests/integration_tests/`) are skipped unless `ANTHROPIC_API_KEY` is set.

## Architecture

The agent runs as a LangGraph graph (`agent.graph:graph`) with a single tool:

**Retrieve tool** (`src/agent/graph.py`) — multi-step pipeline:
1. Parallel keyword group search via `ThreadPoolExecutor` (one thread per keyword group)
2. Deduplication by `page_content`
3. Per-perspective reranking using Qwen3-Reranker-8B
4. Returns structured markdown grouped by perspective

**Vector store** (`src/agent/vectorstore.py`):
- Milvus Lite (embedded `.db` file) auto-started via TCP on `APP_MILVUS_LITE_PORT` (default 19530)
- Embeddings: Qwen3-Embedding-8B via Gitee OpenAI-compatible API (chunk limit: 16 texts/request)
- Chunks: 1500 chars, 150 overlap, RecursiveCharacterTextSplitter
- Metadata fields: `symbol`, `fiscal_year`, `section_title`, `section_id`, `form_type`
- `build_expr(filters)` converts Python dicts to Milvus filter expressions

**FastAPI service** (`src/api/main.py`):
- `POST /upload` — standard 10-K JSON format (nested dict keyed by SQL query string)
- `POST /upload/raw` — generic JSON array with configurable `text_field`
- `GET /health`

## Environment

Copy `.env.example` to `.env`. Key variables:

| Variable | Purpose |
|---|---|
| `APP_MILVUS_URI` | `./milvus.db` (Lite) or `http://localhost:19530` (Docker) |
| `APP_MILVUS_COLLECTION` | Collection name (default: `aapl_10k`) |
| `EMBEDDING_OPENAI_API_KEY` / `BASE_URL` | Gitee API for Qwen3-Embedding-8B |
| `RERANKER_OPENAI_API_KEY` / `BASE_URL` / `MODEL` | Gitee API for Qwen3-Reranker-8B |
| `CHAT_OPENAI_API_KEY` / `BASE_URL` / `CHAT_MODEL` | Chat LLM (default: MiniMax-M2.7) |
| `LANGSMITH_API_KEY` | LangSmith tracing (optional) |

All three model providers (embedding, reranking, chat) use OpenAI-compatible APIs with separate keys/base URLs.

## LangGraph Deployment

`langgraph.json` points to `agent.graph:graph` with Python 3.13 and loads `.env`. Run `make run` for local dev via `langgraph dev`.

For full Milvus (non-Lite), use `docker-compose.yml` which brings up etcd, MinIO, Milvus standalone, and Attu UI.

## 10-K JSON Format

The ingest script and `/upload` endpoint expect:
```json
{
  "<any_key>": [
    {
      "symbol": "AAPL",
      "file_fiscal_year": 2024,
      "section_title": "Item 1. Business",
      "section_id": 1,
      "section_text": "...",
      "form_type": "10-K"
    }
  ]
}
```
