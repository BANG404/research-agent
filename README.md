# Research Agent (Apple 10-K Analysis Assistant)

English | [[简体中文](./README_CN.md)](./README.md)

An AI research assistant built with LangGraph for analyzing Apple Inc. SEC 10-K filings. 
It uses hybrid retrieval (keyword search + semantic reranking) with Milvus vector store for high-precision financial Q&A.

## Architecture

- **Agent Core**: LangGraph-based agent (`agent.graph:graph`) with multi-step reasoning.
- **RAG Pipeline**: 
  - Vector Store: Milvus (Standalone or Lite) with hybrid search.
  - Models: OpenAI-compatible APIs for Embedding and Reranking (e.g., Qwen3).
- **Backend**: FastAPI upload service (`make api`).
- **Frontend**: Next.js interactive UI (`make ui`).

## Quick Start (Production/Full-Stack)

### 1. Environment Setup

Configure your API keys in the `.env` file:
```bash
cp .env.example .env
```

### 2. Build & Launch Infrastructure

Build the LangGraph server image before starting Docker:

```bash
make sync        # Install dependencies
make build       # Build production image via langgraph cli
```

Start the production stack (Milvus, etc.):

```bash
make up-prod
```

### 3. Data Ingestion

Once the stack is healthy, ingest the sample Apple 10-K data:

```bash
make ingest-prod # Process and upload aapl_10k.json to Milvus
```

### 4. Run & Interact

**Start the UI and API servers:**
- Frontend:
  ```bash
  make ui-install
  make ui          # Starts at :3000
  ```
- Backend/Agent:
  ```bash
  make api         # Starts at :8080
  make run         # Starts LangGraph dev server at :2024
  ```

## Common Commands

| Command | Description |
|---|---|
| `make up-prod` | Start production Docker stack (Milvus standalone) |
| `make ingest-prod`| Ingest Apple 10-K JSON into production Milvus |
| `make dev` | Start local dev flow (LangGraph + Frontend) |

---
*See [`CLAUDE.md`](./CLAUDE.md) for more technical details.*
