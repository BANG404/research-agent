"""Unit tests for agent.vectorstore.

Milvus connections and HTTP calls are fully mocked — no real server needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import agent.vectorstore as vs_module
from agent.vectorstore import build_expr, rerank, search, split_and_add


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, **meta) -> Document:
    return Document(
        page_content=text,
        metadata={"symbol": "AAPL", "fiscal_year": 2024, **meta},
    )


def _mock_vs(docs: list[Document] | None = None) -> MagicMock:
    """Return a mock Milvus instance whose similarity_search returns *docs*."""
    mv = MagicMock()
    mv.similarity_search.return_value = docs or []
    mv.add_documents.return_value = ["id1", "id2"]
    return mv


# ---------------------------------------------------------------------------
# build_expr
# ---------------------------------------------------------------------------

class TestBuildExpr:
    def test_empty_dict_returns_none(self):
        assert build_expr({}) is None

    def test_single_int_field(self):
        assert build_expr({"fiscal_year": 2024}) == "fiscal_year == 2024"

    def test_single_str_field(self):
        assert build_expr({"symbol": "AAPL"}) == 'symbol == "AAPL"'

    def test_multiple_fields_joined_with_and(self):
        expr = build_expr({"fiscal_year": 2024, "symbol": "AAPL"})
        assert "fiscal_year == 2024" in expr
        assert 'symbol == "AAPL"' in expr
        assert "&&" in expr

    def test_float_value(self):
        assert build_expr({"score": 0.9}) == "score == 0.9"


# ---------------------------------------------------------------------------
# search — Milvus interaction
# ---------------------------------------------------------------------------

class TestSearch:
    def setup_method(self):
        # Clear cache before each test so get_vectorstore() creates fresh mock
        vs_module._vs_cache.clear()

    def test_calls_similarity_search_with_query_and_k(self):
        mock_vs = _mock_vs([_make_doc("Revenue details.")])

        with patch("agent.vectorstore.get_vectorstore", return_value=mock_vs):
            result = search("Apple revenue", k=7)

        mock_vs.similarity_search.assert_called_once()
        call_kwargs = mock_vs.similarity_search.call_args
        assert call_kwargs.args[0] == "Apple revenue"
        assert call_kwargs.kwargs["k"] == 7

    def test_returns_documents_from_milvus(self):
        docs = [_make_doc("iPhone sales up 4%."), _make_doc("Services up 14%.")]
        mock_vs = _mock_vs(docs)

        with patch("agent.vectorstore.get_vectorstore", return_value=mock_vs):
            result = search("Apple results")

        assert result == docs

    def test_no_filter_omits_expr_kwarg(self):
        mock_vs = _mock_vs()

        with patch("agent.vectorstore.get_vectorstore", return_value=mock_vs):
            search("query", metadata_filters=None)

        kwargs = mock_vs.similarity_search.call_args.kwargs
        assert "expr" not in kwargs

    def test_filter_translated_to_expr(self):
        mock_vs = _mock_vs()

        with patch("agent.vectorstore.get_vectorstore", return_value=mock_vs):
            search("query", metadata_filters={"fiscal_year": 2024})

        kwargs = mock_vs.similarity_search.call_args.kwargs
        assert kwargs.get("expr") == "fiscal_year == 2024"

    def test_connection_error_retries_with_fresh_vectorstore(self):
        """On first failure the cache is cleared and a new instance is tried."""
        failing_vs = MagicMock()
        failing_vs.similarity_search.side_effect = Exception("closed channel")

        ok_vs = _mock_vs([_make_doc("Recovered result.")])

        call_count = 0

        def get_vs_side_effect():
            nonlocal call_count
            call_count += 1
            return failing_vs if call_count == 1 else ok_vs

        with patch("agent.vectorstore.get_vectorstore", side_effect=get_vs_side_effect):
            result = search("query")

        assert call_count == 2
        assert result[0].page_content == "Recovered result."

    def test_second_failure_raises(self):
        """If retry also fails, the exception propagates."""
        bad_vs = MagicMock()
        bad_vs.similarity_search.side_effect = Exception("still broken")

        with patch("agent.vectorstore.get_vectorstore", return_value=bad_vs):
            with pytest.raises(Exception, match="still broken"):
                search("query")


# ---------------------------------------------------------------------------
# split_and_add
# ---------------------------------------------------------------------------

class TestSplitAndAdd:
    def setup_method(self):
        vs_module._vs_cache.clear()

    def test_returns_number_of_chunks_inserted(self):
        mock_vs = MagicMock()
        mock_vs.add_documents.return_value = ["a", "b", "c"]

        doc = _make_doc("word " * 50)  # short enough to be one chunk
        with patch("agent.vectorstore.get_vectorstore", return_value=mock_vs):
            count = split_and_add([doc])

        assert count == 3

    def test_long_doc_split_into_multiple_chunks(self):
        captured_chunks: list[list[Document]] = []

        mock_vs = MagicMock()

        def capture_add(chunks):
            captured_chunks.append(chunks)
            return [f"id{i}" for i in range(len(chunks))]

        mock_vs.add_documents.side_effect = capture_add

        # ~4500 chars → should produce at least 3 chunks (chunk_size=1500)
        long_text = "Apple reported strong results. " * 150
        doc = _make_doc(long_text)

        with patch("agent.vectorstore.get_vectorstore", return_value=mock_vs):
            split_and_add([doc])

        chunks = captured_chunks[0]
        assert len(chunks) >= 3
        for c in chunks:
            assert len(c.page_content) <= 1500

    def test_short_chunks_below_100_chars_filtered(self):
        """Page-footer artifacts shorter than 100 chars must be dropped."""
        captured_chunks: list[list[Document]] = []

        mock_vs = MagicMock()

        def capture_add(chunks):
            captured_chunks.append(chunks)
            return [f"id{i}" for i in range(len(chunks))]

        mock_vs.add_documents.side_effect = capture_add

        # Mix of real content and a page-footer stub
        real_text = "Apple Inc. reported total net sales of $416 billion. " * 30
        footer_text = "Apple Inc. | 2024 Form 10-K | 5"  # 31 chars — should be dropped
        docs = [_make_doc(real_text), _make_doc(footer_text)]

        with patch("agent.vectorstore.get_vectorstore", return_value=mock_vs):
            split_and_add(docs)

        chunks = captured_chunks[0]
        for c in chunks:
            assert len(c.page_content.strip()) >= 100

    def test_metadata_preserved_in_chunks(self):
        captured_chunks: list[list[Document]] = []

        mock_vs = MagicMock()

        def capture_add(chunks):
            captured_chunks.append(chunks)
            return [f"id{i}" for i in range(len(chunks))]

        mock_vs.add_documents.side_effect = capture_add

        doc = Document(
            page_content="Apple revenue details. " * 30,
            metadata={"symbol": "AAPL", "fiscal_year": 2025, "section_title": "Item 7. MD&A"},
        )

        with patch("agent.vectorstore.get_vectorstore", return_value=mock_vs):
            split_and_add([doc])

        for chunk in captured_chunks[0]:
            assert chunk.metadata["symbol"] == "AAPL"
            assert chunk.metadata["fiscal_year"] == 2025


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------

class TestRerank:
    def test_empty_documents_returns_empty(self):
        assert rerank("query", []) == []

    def test_returns_sorted_by_score_descending(self):
        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.3},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.6},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response

        with patch("httpx.post", return_value=mock_resp):
            result = rerank("query", ["doc0", "doc1", "doc2"], top_n=3)

        assert result[0] == (1, 0.9)
        assert result[1] == (2, 0.6)
        assert result[2] == (0, 0.3)

    def test_top_n_capped_at_document_count(self):
        """top_n larger than number of documents should not cause an error."""
        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.5},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            rerank("query", ["doc0", "doc1"], top_n=10)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["top_n"] == 2  # capped at len(documents)

    def test_http_error_falls_back_to_identity_order(self):
        with patch("httpx.post", side_effect=Exception("network error")):
            result = rerank("query", ["a", "b", "c"], top_n=3)

        # fallback: identity order with score 0.0
        assert [idx for idx, _ in result] == [0, 1, 2]
        assert all(score == 0.0 for _, score in result)

    def test_non_200_response_falls_back(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")

        with patch("httpx.post", return_value=mock_resp):
            result = rerank("query", ["a", "b"], top_n=2)

        assert len(result) == 2

    def test_request_payload_contains_required_fields(self):
        mock_response = {"results": [{"index": 0, "relevance_score": 1.0}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            rerank("what is Apple revenue", ["Apple revenue grew 6%"], top_n=1)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["query"] == "what is Apple revenue"
        assert payload["documents"] == ["Apple revenue grew 6%"]
        assert "model" in payload
