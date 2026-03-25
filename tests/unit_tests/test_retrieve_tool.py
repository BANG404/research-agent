"""Unit tests for the retrieve tool.

All Milvus/reranker I/O is mocked — no real vector store or API calls needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agent.graph import retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(*texts: str, **meta_overrides) -> list[Document]:
    base_meta = {
        "symbol": "AAPL",
        "fiscal_year": 2024,
        "section_title": "Item 1. Business",
        "section_id": 1,
        "form_type": "10-K",
    }
    return [
        Document(page_content=t, metadata={**base_meta, **meta_overrides})
        for t in texts
    ]


def _identity_rerank(query: str, documents: list[str], top_n: int = 5):
    """Reranker that returns documents in original order."""
    return [(i, 1.0 - i * 0.1) for i in range(min(top_n, len(documents)))]


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------

class TestRetrieveBasic:
    def test_returns_markdown_with_perspective_header(self):
        docs = _make_docs("Apple revenue grew 6% in fiscal 2024.")

        with patch("agent.vectorstore.search", return_value=docs), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            result = retrieve.invoke({
                "question": "What was Apple's revenue growth?",
                "perspectives": ["Apple revenue growth"],
                "keyword_groups": [["Apple revenue 2024"]],
            })

        assert "## Perspective: Apple revenue growth" in result
        assert "Apple revenue grew 6%" in result

    def test_multiple_perspectives_separated_by_equals(self):
        docs = _make_docs("iPhone sales increased.", "Services revenue up 14%.")

        with patch("agent.vectorstore.search", return_value=docs), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            result = retrieve.invoke({
                "question": "iPhone and Services performance?",
                "perspectives": ["iPhone performance", "Services performance"],
                "keyword_groups": [["iPhone sales"], ["Services revenue"]],
            })

        assert "## Perspective: iPhone performance" in result
        assert "## Perspective: Services performance" in result
        assert "===" in result

    def test_top_k_limits_docs_per_perspective(self):
        docs = _make_docs(*[f"Doc {i}" for i in range(20)])

        captured: list[int] = []

        def rerank_capture(query, documents, top_n=5):
            captured.append(top_n)
            return [(i, 1.0) for i in range(min(top_n, len(documents)))]

        with patch("agent.vectorstore.search", return_value=docs), \
             patch("agent.vectorstore.rerank", side_effect=rerank_capture):
            retrieve.invoke({
                "question": "test",
                "perspectives": ["p1"],
                "keyword_groups": [["kw"]],
                "top_k": 3,
            })

        assert captured[0] == 3


# ---------------------------------------------------------------------------
# Empty / no results
# ---------------------------------------------------------------------------

class TestRetrieveEmpty:
    def test_no_docs_returns_not_found_message(self):
        with patch("agent.vectorstore.search", return_value=[]):
            result = retrieve.invoke({
                "question": "irrelevant query",
                "perspectives": ["p1"],
                "keyword_groups": [["nothing"]],
            })

        assert "No relevant documents found" in result

    def test_all_keyword_groups_fail_returns_not_found(self):
        def failing_search(*args, **kwargs):
            raise RuntimeError("connection error")

        with patch("agent.vectorstore.search", side_effect=failing_search):
            result = retrieve.invoke({
                "question": "test",
                "perspectives": ["p1"],
                "keyword_groups": [["kw1"], ["kw2"]],
            })

        assert "No relevant documents found" in result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestRetrieveDeduplicate:
    def test_duplicate_page_content_removed(self):
        """Same text returned by two keyword groups should appear only once."""
        dup_text = "Apple Inc. designs, manufactures, and markets smartphones."
        docs = _make_docs(dup_text)

        search_calls: list[str] = []

        def search_returning_dup(query, filters, k):
            search_calls.append(query)
            return docs  # both groups return the same doc

        rerank_calls: list[list[str]] = []

        def rerank_capture(query, documents, top_n=5):
            rerank_calls.append(documents)
            return [(i, 1.0) for i in range(min(top_n, len(documents)))]

        with patch("agent.vectorstore.search", side_effect=search_returning_dup), \
             patch("agent.vectorstore.rerank", side_effect=rerank_capture):
            retrieve.invoke({
                "question": "Apple business",
                "perspectives": ["Apple business overview"],
                "keyword_groups": [["Apple business"], ["Apple products"]],
            })

        # After dedup, reranker should only receive one unique doc
        assert len(rerank_calls[0]) == 1


# ---------------------------------------------------------------------------
# metadata_filters handling
# ---------------------------------------------------------------------------

class TestRetrieveMetadataFilters:
    def test_dict_filter_passed_to_search(self):
        captured_filters: list = []

        def search_capture(query, filters, k):
            captured_filters.append(filters)
            return _make_docs("some text")

        with patch("agent.vectorstore.search", side_effect=search_capture), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            retrieve.invoke({
                "question": "test",
                "perspectives": ["p1"],
                "keyword_groups": [["kw"]],
                "metadata_filters": {"fiscal_year": 2024},
            })

        assert captured_filters[0] == {"fiscal_year": 2024}

    def test_none_filter_passed_as_none(self):
        captured_filters: list = []

        def search_capture(query, filters, k):
            captured_filters.append(filters)
            return _make_docs("some text")

        with patch("agent.vectorstore.search", side_effect=search_capture), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            retrieve.invoke({
                "question": "test",
                "perspectives": ["p1"],
                "keyword_groups": [["kw"]],
                "metadata_filters": None,
            })

        assert captured_filters[0] is None

    @pytest.mark.parametrize("bad_value", ["none", "null", "", "  "])
    def test_string_none_variants_normalised_to_none(self, bad_value):
        """LLM sometimes passes "none" or "null" as a string — should be normalised."""
        captured_filters: list = []

        def search_capture(query, filters, k):
            captured_filters.append(filters)
            return _make_docs("some text")

        with patch("agent.vectorstore.search", side_effect=search_capture), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            retrieve.invoke({
                "question": "test",
                "perspectives": ["p1"],
                "keyword_groups": [["kw"]],
                "metadata_filters": bad_value,
            })

        assert captured_filters[0] is None

    def test_json_string_filter_parsed(self):
        """LLM may pass filters as a JSON string instead of a dict."""
        captured_filters: list = []

        def search_capture(query, filters, k):
            captured_filters.append(filters)
            return _make_docs("some text")

        with patch("agent.vectorstore.search", side_effect=search_capture), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            retrieve.invoke({
                "question": "test",
                "perspectives": ["p1"],
                "keyword_groups": [["kw"]],
                "metadata_filters": '{"fiscal_year": 2025}',
            })

        assert captured_filters[0] == {"fiscal_year": 2025}


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

class TestRetrieveOutputFormat:
    def test_doc_header_contains_metadata(self):
        docs = [Document(
            page_content="Revenue details here.",
            metadata={
                "symbol": "AAPL",
                "fiscal_year": 2024,
                "section_title": "Item 7. MD&A",
                "section_id": 7,
                "form_type": "10-K",
            },
        )]

        with patch("agent.vectorstore.search", return_value=docs), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            result = retrieve.invoke({
                "question": "revenue",
                "perspectives": ["revenue analysis"],
                "keyword_groups": [["revenue"]],
            })

        assert "AAPL" in result
        assert "2024" in result
        assert "Item 7. MD&A" in result

    def test_doc_content_truncated_at_2000_chars(self):
        long_text = "x" * 3000
        docs = _make_docs(long_text)

        with patch("agent.vectorstore.search", return_value=docs), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            result = retrieve.invoke({
                "question": "test",
                "perspectives": ["p1"],
                "keyword_groups": [["kw"]],
            })

        # 2000 x's + some overhead, should not contain 3000 x's
        assert "x" * 2001 not in result
        assert "x" * 2000 in result

    def test_chunks_separated_by_horizontal_rule(self):
        docs = _make_docs("First chunk.", "Second chunk.")

        with patch("agent.vectorstore.search", return_value=docs), \
             patch("agent.vectorstore.rerank", side_effect=_identity_rerank):
            result = retrieve.invoke({
                "question": "test",
                "perspectives": ["p1"],
                "keyword_groups": [["kw"]],
                "top_k": 2,
            })

        assert "---" in result
