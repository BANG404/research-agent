"""Shared vectorstore, embeddings, and reranker utilities.

Uses langchain_milvus.Milvus backed by either:
  - Milvus Lite  (APP_MILVUS_URI=./milvus.db, no Docker required)
  - Milvus server (APP_MILVUS_URI=http://localhost:19530)

NOTE: APP_MILVUS_URI avoids collision with pymilvus's own MILVUS_URI env var,
which pymilvus reads at import time and rejects non-http URIs.
"""

from __future__ import annotations

import os

import httpx
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MILVUS_URI = os.getenv("APP_MILVUS_URI", "./milvus.db")
COLLECTION_NAME = os.getenv("APP_MILVUS_COLLECTION", "aapl_10k")

EMBEDDING_API_KEY = os.getenv("EMBEDDING_OPENAI_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_OPENAI_BASE_URL", "https://ai.gitee.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")

RERANKER_API_KEY = os.getenv("RERANKER_OPENAI_API_KEY", "")
RERANKER_BASE_URL = os.getenv("RERANKER_OPENAI_BASE_URL", "https://ai.gitee.com/v1")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen3-Reranker-8B")

# Chunk long sections so each embedding stays within model context limits
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ---------------------------------------------------------------------------
# Embeddings (Qwen3-Embedding-8B via Gitee OpenAI-compatible API)
# ---------------------------------------------------------------------------


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,  # type: ignore[arg-type]
        base_url=EMBEDDING_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def get_vectorstore(collection_name: str = COLLECTION_NAME) -> Milvus:
    """Return a Milvus vector store connected to the configured backend.

    The collection is created automatically on first use.
    Metadata fields (symbol, fiscal_year, section_title, etc.) are stored via
    dynamic fields and can be used in Milvus filter expressions.
    """
    return Milvus(
        embedding_function=get_embeddings(),
        connection_args={"uri": MILVUS_URI},
        collection_name=collection_name,
        auto_id=True,
        enable_dynamic_field=True,
    )


def split_and_add(docs: list[Document], collection_name: str = COLLECTION_NAME) -> int:
    """Chunk documents with RecursiveCharacterTextSplitter then upsert into Milvus.

    Returns the number of chunks inserted.
    """
    chunks = _splitter.split_documents(docs)
    vs = get_vectorstore(collection_name)
    ids = vs.add_documents(chunks)
    return len(ids)


# ---------------------------------------------------------------------------
# Reranker (Qwen3-Reranker-8B via Gitee /v1/rerank — Cohere-compatible)
# ---------------------------------------------------------------------------


def rerank(query: str, documents: list[str], top_n: int = 5) -> list[tuple[int, float]]:
    """Rerank documents with Qwen3-Reranker-8B.

    Returns list of (original_index, relevance_score) sorted descending.
    Falls back to identity ordering on any error.
    """
    if not documents:
        return []

    url = RERANKER_BASE_URL.rstrip("/") + "/rerank"
    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {RERANKER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANKER_MODEL,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        items = resp.json()["results"]
        ranked = [(item["index"], item["relevance_score"]) for item in items]
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[reranker] failed, falling back to original order: {exc}")
        return [(i, 0.0) for i in range(min(top_n, len(documents)))]
