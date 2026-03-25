# Research Agent (Apple 10-K Analysis Assistant)

English | [[简体中文](./README_CN.md)](./README.md)

An AI research assistant built with LangGraph for analyzing Apple Inc. SEC 10-K filings. 
It uses hybrid retrieval (keyword search + semantic reranking) with Milvus vector store for high-precision financial Q&A.

## Agent and Retrieval Architecture Design

The project leverages LangGraph to design a multi-step retrieval and analysis pipeline specifically tailored for complex long texts like financial reports. It primarily involves three core components:

### 1. Data Pre-processing & Indexing
- **Text Chunking**: Structured Apple 10-K financial report JSON data is parsed and sliced using `RecursiveCharacterTextSplitter` (default 1500 chars limit, 150 chars overlap) to ensure coherent contextual semantics.
- **Embedded Vectorization**: The application utilizes Qwen3-Embedding-8B (via an OpenAI-compatible API) to bundle text chunks and calculate vector embeddings.
- **Storage & Metadata**: Vectors are stored in the Milvus vector database (handling Lite or Standalone environments). Comprehensive metadata attributes (such as `symbol`, `fiscal_year`, `section_title`, and `form_type`) are preserved alongside the vectors, establishing a foundation for precise structured condition filtering.

### 2. Text Retrieval & Information Augmentation (RAG Pipeline)
- **Multi-dimensional Concurrent Retrieval**: Based on the user's questions, inquiries are divided into different evaluation perspectives and keyword groups. Concurrent retrieval is executed across these keyword groups utilizing a `ThreadPoolExecutor`. This parallel design significantly enhances the recall efficiency of complex financial report contexts.
- **Content Deduplication & Cleansing**: The retrieval results from various threads are aggregated and strictly deduplicated via their `page_content` to prevent redundant information from interfering with large language model inference.
- **Semantic Reranking**: The recalled preliminary materials are distributed to the Qwen3-Reranker-8B model based on distinct evaluation perspectives (Perspectives) for deep semantic reranking and scoring. The high-matching structured Markdown content is then filtered out to complete the multi-dimensional information augmentation.

### 3. Answer Generation & Brief Analysis Reports
- **Core Inference and Output**: After obtaining highly condensed and cleansed background context from the reranking mechanism, the Agent's decision-making brain (Graph) integrates with powerful conversational LLMs (e.g., MiniMax-M2.7). Based on the user's questions and the financial report snippets, it conducts comprehensive analysis and tabular summaries. Ultimately, it can not only handle common Q&A but also automatically draft logically rigorous stage-by-stage brief financial analysis reports.

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
