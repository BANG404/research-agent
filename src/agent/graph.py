"""LangGraph 1.0 research agent with Milvus retrieval.

Pipeline (Plan2.md):
  1. Query Analysis   – LLM decomposes question into 3-5 sub-queries.
  2. Multi-Query Retrieval – parallel similarity_search per sub-query.
  3. Reranking        – merged + deduplicated hits reranked by Qwen3-Reranker.
  4. Answer           – top-K docs returned as context; LLM synthesises answer.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Chat model (MiniMax-M2.7 via OpenAI-compatible API)
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "MiniMax-M2.7"),
    api_key=os.getenv("CHAT_OPENAI_API_KEY", ""),  # type: ignore[arg-type]
    base_url=os.getenv("CHAT_OPENAI_BASE_URL", "https://api.minimaxi.com/v1"),
    temperature=0,
)

# Separate instance for query analysis — slightly higher temperature
_analysis_llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "MiniMax-M2.7"),
    api_key=os.getenv("CHAT_OPENAI_API_KEY", ""),  # type: ignore[arg-type]
    base_url=os.getenv("CHAT_OPENAI_BASE_URL", "https://api.minimaxi.com/v1"),
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# Internal retrieval helpers
# ---------------------------------------------------------------------------


def _decompose_query(question: str) -> list[str]:
    """Ask the LLM to split the question into 3-5 independent sub-queries."""
    prompt = (
        "You are a search query analyst. "
        "Break the following question into 3 to 5 independent sub-queries that together "
        "cover all information needed to answer it. "
        "Return ONLY a JSON array of strings with no explanation.\n\n"
        f"Question: {question}"
    )
    content = str(_analysis_llm.invoke(prompt).content).strip()
    # Strip markdown fences
    if content.startswith("```"):
        content = content.split("```")[1].lstrip("json").strip()
    try:
        sub_queries: list[str] = json.loads(content)
        if not isinstance(sub_queries, list):
            raise ValueError
    except Exception:
        sub_queries = [question]
    return sub_queries[:5]


def _search_one(query: str, fiscal_year: int | None, k: int) -> list[Document]:
    """Similarity search for a single sub-query with optional year filter."""
    from agent.vectorstore import get_vectorstore

    vs = get_vectorstore()
    kwargs: dict = {"k": k}
    if fiscal_year is not None:
        kwargs["expr"] = f"fiscal_year == {fiscal_year}"
    return vs.similarity_search(query, **kwargs)


def _deduplicate(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            out.append(doc)
    return out


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------


@tool
def retrieve(question: str, fiscal_year: int | None = None, top_k: int = 5) -> str:
    """Search the AAPL 10-K knowledge base for sections relevant to the question.

    Args:
        question:    The user's question or information need.
        fiscal_year: Optional fiscal year filter (e.g. 2024).
        top_k:       Number of chunks to return after reranking (default 5).

    Returns:
        Formatted string of the most relevant 10-K sections with metadata.
    """
    from agent.vectorstore import rerank

    # Step 1 — query decomposition
    sub_queries = _decompose_query(question)

    # Step 2 — parallel retrieval
    per_query_k = max(top_k * 3, 15)
    all_docs: list[Document] = []
    with ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
        futures = {pool.submit(_search_one, q, fiscal_year, per_query_k): q for q in sub_queries}
        for future in as_completed(futures):
            try:
                all_docs.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"[retrieve] '{futures[future]}' failed: {exc}")

    if not all_docs:
        return "No relevant documents found in the knowledge base."

    # Step 3 — deduplicate + rerank
    unique = _deduplicate(all_docs)
    texts = [d.page_content for d in unique]
    ranked = rerank(question, texts, top_n=top_k)
    top_docs = [unique[idx] for idx, _ in ranked]

    # Step 4 — format
    parts = [
        f"[{i}] {doc.metadata.get('section_title', 'N/A')} | "
        f"{doc.metadata.get('symbol', 'N/A')} | "
        f"FY{doc.metadata.get('fiscal_year', 'N/A')}\n"
        f"{doc.page_content[:2000]}"
        for i, doc in enumerate(top_docs, 1)
    ]
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# LangChain agent (langchain.agents.create_agent — LangChain 1.0 LTS)
# ---------------------------------------------------------------------------

graph = create_agent(
    model=_llm,
    tools=[retrieve],
    system_prompt=(
        "You are a financial research assistant specialising in Apple Inc. SEC filings. "
        "Always call the retrieve tool before answering factual questions about Apple's "
        "business, financials, risks, or strategy. "
        "Cite the section title and fiscal year from the retrieved context. "
        "Be concise and accurate."
    ),
)
