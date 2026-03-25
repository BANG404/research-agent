"""LangGraph 1.0 research agent with Milvus retrieval.

Pipeline:
  1. Retrieve  – per keyword-group hybrid search + metadata filter (parallel).
  2. Reranking – per perspective reranking of deduplicated results.
  3. Answer    – all perspectives returned to LLM for synthesis.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any

from pydantic import BeforeValidator

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

# ---------------------------------------------------------------------------
# Internal retrieval helpers
# ---------------------------------------------------------------------------


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
def retrieve(
    question: str,
    perspectives: list[str],
    keyword_groups: list[list[str]],
    metadata_filters: Annotated[dict | None, BeforeValidator(lambda v: None if (not isinstance(v, str) or v.strip().lower() in ("none", "null", "")) else json.loads(v) if v.strip().startswith("{") else None)] = None,
    top_k: int = 5,
) -> str:
    """Search the AAPL 10-K knowledge base for sections relevant to the question.

    Args:
        question:         The original user question.
        perspectives:     List of retrieval angles used for reranking (e.g. ["revenue trend",
                          "risk factors"]). Each perspective produces its own ranked result set.
        keyword_groups:   List of keyword groups; each group is searched independently.
                          Example: [["revenue", "sales"], ["operating expenses", "cost"]].
        metadata_filters: Optional metadata field filters, e.g. {"fiscal_year": 2024}.
        top_k:            Number of chunks per perspective after reranking (default 5).

    Returns:
        Formatted string with one result section per perspective.
    """
    from agent.vectorstore import rerank, search


    per_query_k = max(top_k * 3, 15)

    # Step 1 — parallel search per keyword group with metadata filtering
    all_docs: list[Document] = []
    with ThreadPoolExecutor(max_workers=len(keyword_groups)) as pool:
        futures = {
            pool.submit(search, " ".join(group), metadata_filters, per_query_k): group
            for group in keyword_groups
        }
        for future in as_completed(futures):
            try:
                all_docs.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"[retrieve] keyword group '{futures[future]}' failed: {exc}")

    if not all_docs:
        return "No relevant documents found in the knowledge base."

    # Step 2 — deduplicate
    unique = _deduplicate(all_docs)
    texts = [d.page_content for d in unique]

    # Step 3 — rerank per perspective and collect results
    def _fmt_doc(i: int, doc: Document) -> str:
        return (
            f"  [{i}] {doc.metadata.get('section_title', 'N/A')} | "
            f"{doc.metadata.get('symbol', 'N/A')} | "
            f"FY{doc.metadata.get('fiscal_year', 'N/A')}\n"
            f"  {doc.page_content[:2000]}"
        )

    sections: list[str] = []
    for perspective in perspectives:
        ranked = rerank(perspective, texts, top_n=top_k)
        top_docs = [unique[idx] for idx, _ in ranked]
        body = "\n\n---\n\n".join(_fmt_doc(i, doc) for i, doc in enumerate(top_docs, 1))
        sections.append(f"## Perspective: {perspective}\n\n{body}")

    return "\n\n===\n\n".join(sections)


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
        "When calling retrieve, you must supply:\n"
        "  • perspectives – 2-4 distinct angles from which to evaluate the results "
        "(e.g. 'revenue growth', 'risk factors', 'segment breakdown').\n"
        "  • keyword_groups – 2-4 groups of keywords, each group targeting a different "
        "aspect of the question (e.g. [['revenue', 'net sales'], ['operating income']]).\n"
        "  • metadata_filters – set {'fiscal_year': YYYY} when a specific year is mentioned.\n"
        "Cite the section title and fiscal year from the retrieved context. "
        "Be concise and accurate."
    ),
)
