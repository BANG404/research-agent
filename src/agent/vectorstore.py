"""Shared vectorstore, embeddings, and reranker utilities.

Uses langchain_milvus.Milvus backed by either:
  - Milvus Lite  (APP_MILVUS_URI=./milvus.db, no Docker required)
  - Milvus server (APP_MILVUS_URI=http://localhost:19530)

NOTE: APP_MILVUS_URI avoids collision with pymilvus's own MILVUS_URI env var,
which pymilvus reads at import time and rejects non-http URIs.
"""

from __future__ import annotations

import os
import threading

import httpx
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MILVUS_URI = os.getenv("APP_MILVUS_URI", "./milvus.db")


# ---------------------------------------------------------------------------
# Milvus Lite TCP bootstrap (UDS is unreliable on some Linux/WSL2 environments)
# ---------------------------------------------------------------------------

_milvus_lite_server = None  # keep reference so process stays alive


def _port_listening(host: str, port: int) -> bool:
    """Return True if something is already accepting connections on host:port."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _resolve_milvus_uri(uri: str) -> str:
    """If *uri* is a local .db file, start milvus-lite on TCP and return an
    http://localhost:PORT URI.  Otherwise return *uri* unchanged.

    If another process (e.g. the LangGraph dev server) has already started
    milvus-lite on the configured port, the existing server is reused instead
    of attempting a second start (which would fail because the .db file is
    locked).
    """
    global _milvus_lite_server
    if uri.startswith("http://") or uri.startswith("https://"):
        return uri
    # Local file path → start milvus-lite with TCP
    import pathlib
    from milvus_lite.server import Server

    port = int(os.getenv("APP_MILVUS_LITE_PORT", "19530"))
    tcp_addr = f"localhost:{port}"
    http_uri = f"http://{tcp_addr}"

    if _port_listening("localhost", port):
        # Another process already started milvus-lite — reuse it.
        return http_uri

    db_path = str(pathlib.Path(uri).absolute())
    if _milvus_lite_server is None:
        import time

        s = Server(db_path, address=tcp_addr)
        if not s.init() or not s.start():
            raise RuntimeError(f"Failed to start milvus-lite for {db_path}")
        time.sleep(2)  # wait for gRPC server to be ready
        _milvus_lite_server = s
    return http_uri
COLLECTION_NAME = os.getenv("APP_MILVUS_COLLECTION", "aapl_10k")

# BM25BuiltInFunction requires Milvus Standalone/Cluster (not Milvus Lite).
# Enable only when APP_MILVUS_URI points to a running Milvus server.
_USE_BM25 = MILVUS_URI.startswith("http://") or MILVUS_URI.startswith("https://")

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
        check_embedding_ctx_length=False,  # send plain text, not tiktoken IDs
        chunk_size=16,  # API limit: max 20 texts per request
    )


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

_vs_cache: dict[str, Milvus] = {}
_vs_lock = threading.Lock()


def _make_vectorstore(collection_name: str, uri: str) -> Milvus:
    """Instantiate a Milvus vector store.

    When APP_MILVUS_URI points to a full Milvus server, both vector_field and
    builtin_function are always passed so langchain_milvus can handle hybrid
    search (dense + BM25) on both create and connect paths.
    """
    kwargs: dict = {}
    if _USE_BM25:
        kwargs["vector_field"] = ["vector", "sparse"]
        kwargs["builtin_function"] = BM25BuiltInFunction(
            analyzer_params={"type": "english"},
        )
    return Milvus(
        embedding_function=get_embeddings(),
        connection_args={"uri": uri},
        collection_name=collection_name,
        auto_id=True,
        enable_dynamic_field=True,
        **kwargs,
    )


def get_vectorstore(collection_name: str = COLLECTION_NAME) -> Milvus:
    """Return a Milvus vector store connected to the configured backend.

    Metadata fields (symbol, fiscal_year, section_title, etc.) are stored via
    dynamic fields and can be used in Milvus filter expressions.
    """
    if collection_name in _vs_cache:
        return _vs_cache[collection_name]
    with _vs_lock:
        if collection_name in _vs_cache:  # re-check after acquiring lock
            return _vs_cache[collection_name]
        uri = _resolve_milvus_uri(MILVUS_URI)
        vs = _make_vectorstore(collection_name, uri)
        _vs_cache[collection_name] = vs
        return vs


def build_expr(filters: dict) -> str | None:
    """Convert a metadata filter dict to a Milvus filter expression."""
    if not filters:
        return None
    parts = []
    for key, value in filters.items():
        if isinstance(value, str):
            parts.append(f'{key} == "{value}"')
        else:
            parts.append(f"{key} == {value}")
    return " && ".join(parts)


def search(query: str, metadata_filters: dict | None = None, k: int = 10) -> list[Document]:
    """Hybrid search (dense + BM25) with optional metadata filtering.

    When APP_MILVUS_URI points to a full Milvus server, the collection is created
    with a BM25 sparse field and langchain_milvus automatically uses RRF hybrid
    ranking — no explicit ranker kwarg needed.
    """
    kwargs: dict = {"k": k}
    if metadata_filters:
        expr = build_expr(metadata_filters)
        if expr:
            kwargs["expr"] = expr
    for attempt in range(2):
        try:
            return get_vectorstore().similarity_search(query, **kwargs)
        except Exception:
            if attempt == 0:
                with _vs_lock:
                    _vs_cache.pop(COLLECTION_NAME, None)
            else:
                raise


def split_and_add(docs: list[Document], collection_name: str = COLLECTION_NAME) -> int:
    """Chunk documents with RecursiveCharacterTextSplitter then upsert into Milvus.

    Uses builtin_function=BM25BuiltInFunction so the sparse field is created on
    first insert; subsequent calls reuse the same Milvus instance.
    Returns the number of chunks inserted.
    """
    chunks = [c for c in _splitter.split_documents(docs) if len(c.page_content.strip()) >= 100]
    # Use the write-path vectorstore (with BM25 builtin_function for collection creation)
    with _vs_lock:
        write_key = f"__write__{collection_name}"
        if write_key not in _vs_cache:
            uri = _resolve_milvus_uri(MILVUS_URI)
            _vs_cache[write_key] = _make_vectorstore(collection_name, uri)
        vs = _vs_cache[write_key]
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
