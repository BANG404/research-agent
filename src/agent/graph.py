"""LangGraph 1.0 general-purpose research agent with Milvus retrieval.

Pipeline:
  1. Retrieve  – per keyword-group hybrid search + metadata filter (parallel).
  2. Reranking – per perspective reranking of deduplicated results.
  3. Answer    – all perspectives returned to LLM for synthesis.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
    metadata_filters: Annotated[dict | None, BeforeValidator(lambda v: v if isinstance(v, dict) else None if (not isinstance(v, str) or v.strip().lower() in ("none", "null", "")) else json.loads(v) if v.strip().startswith("{") else None)] = None,
    top_k: int = 5,
) -> str:
    """Search the knowledge base for chunks most relevant to the question.

    Args:
        question:         The verbatim user question.
        perspectives:     1-3 reranking angles — each re-ranks the full candidate pool
                          independently. Use more angles only when the question has clearly
                          distinct sub-questions; a simple question needs just one.
        keyword_groups:   2-4 search queries, each a list of ~3 synonyms/variants targeting
                          one theme. Groups run in parallel; results are merged and deduplicated.
        metadata_filters: Optional metadata filters, e.g. {"year": 2026}.
        top_k:            Chunks returned per perspective after reranking (default 5).

    Returns:
        Markdown string with one titled section per perspective.
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
        meta = doc.metadata
        tags = " | ".join(
            str(v) for k, v in meta.items() if v and k not in ("id", "pk", "section_title")
        )
        header = f"  [{i}] {tags}" if tags else f"  [{i}]"
        return f"{header}\n  {doc.page_content[:2000]}"

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

def _system_prompt() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""\
Current date and time: {now}

You are a general-purpose research assistant backed by a private knowledge base.
Always call the `retrieve` tool before answering any factual question.

## How to build `retrieve` arguments

**keyword_groups** — what to fetch (breadth)
  Decompose the question into 2-4 distinct themes. Each group is ~3 synonyms or
  variants of one theme; all groups run in parallel and results are deduplicated.

**perspectives** — how to rank (depth)
  1-3 focused analytical angles. Use 1 for a simple question, up to 3 only when the
  question has clearly distinct sub-questions. Each angle independently re-ranks the
  full candidate pool.

### Example
Question: "How does Ubuntu compare to Windows today? Is switching feasible?"

keyword_groups:
  [
    ["Ubuntu market share 2026", "desktop Linux market share", "Linux usage statistics"],
    ["Windows market share 2026", "Windows desktop usage", "Windows vs Linux adoption"],
    ["Ubuntu developer usage", "Linux developer survey 2026", "developer OS preference"],
    ["switch Windows to Ubuntu", "migrate to Linux pros cons", "Ubuntu migration guide developers"],
  ]

perspectives:
  [
    "Ubuntu vs Windows market share and usage trends 2026",
    "migration feasibility from Windows to Ubuntu for developers",
  ]

## After retrieval
- Synthesise across all perspective sections; do not just repeat them.
- Cite the source metadata shown in each chunk header.
- If the knowledge base has no relevant content, say so and answer from general
  knowledge, noting the limitation.
- Be concise and accurate.
"""


graph = create_agent(
    model=_llm,
    tools=[retrieve],
    system_prompt=_system_prompt(),
)
