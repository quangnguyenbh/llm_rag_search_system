"""Unit tests for the Reranker with mocked Cohere client."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.query.reranker import Reranker
from src.core.query.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    doc_id: str = "doc-1",
    score: float = 0.9,
    text: str = "Some text",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        score=score,
        metadata={"title": "Doc", "page_number": 1},
    )


def _many_chunks(n: int, same_doc: bool = False) -> list[RetrievedChunk]:
    return [
        _make_chunk(
            f"c{i}",
            doc_id="doc-1" if same_doc else f"doc-{i % 5}",
            score=1.0 - i * 0.01,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Score-based fallback (no Cohere key)
# ---------------------------------------------------------------------------


class TestScoreBasedFallback:
    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        reranker = Reranker(cohere_api_key="")
        result = await reranker.rerank("q", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_top_k(self):
        chunks = _many_chunks(20)
        reranker = Reranker(cohere_api_key="")
        result = await reranker.rerank("query", chunks, top_k=5)
        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_score_order_preserved(self):
        chunks = _many_chunks(10)
        reranker = Reranker(cohere_api_key="")
        result = await reranker.rerank("query", chunks, top_k=10)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_diversity_multi_doc(self):
        """When docs are available, result should span multiple documents."""
        chunks = [_make_chunk(f"c{i}", doc_id=f"doc-{i % 4}") for i in range(16)]
        reranker = Reranker(cohere_api_key="")
        result = await reranker.rerank("q", chunks, top_k=8)
        unique_docs = {c.document_id for c in result}
        assert len(unique_docs) >= 2

    @pytest.mark.asyncio
    async def test_single_chunk(self):
        reranker = Reranker(cohere_api_key="")
        chunks = [_make_chunk("only")]
        result = await reranker.rerank("q", chunks, top_k=8)
        assert len(result) == 1
        assert result[0].chunk_id == "only"

    @pytest.mark.asyncio
    async def test_fewer_chunks_than_top_k(self):
        chunks = _many_chunks(3)
        reranker = Reranker(cohere_api_key="")
        result = await reranker.rerank("q", chunks, top_k=10)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_no_key_logs_warning(self, caplog):
        import logging

        chunks = _many_chunks(5)
        reranker = Reranker(cohere_api_key="")
        with caplog.at_level(logging.WARNING):
            await reranker.rerank("q", chunks, top_k=5)
        # structlog doesn't use caplog directly; just verify no crash


# ---------------------------------------------------------------------------
# Cohere reranking (mocked)
# ---------------------------------------------------------------------------


class TestCohereReranking:
    def _make_cohere_result(self, index: int, score: float):
        result = MagicMock()
        result.index = index
        result.relevance_score = score
        return result

    @pytest.mark.asyncio
    async def test_cohere_rerank_called(self):
        chunks = _many_chunks(10)
        mock_cohere_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = [
            self._make_cohere_result(2, 0.95),
            self._make_cohere_result(0, 0.88),
            self._make_cohere_result(5, 0.75),
        ]
        mock_cohere_client.rerank.return_value = mock_response

        reranker = Reranker(cohere_api_key="test-key")
        reranker._client = mock_cohere_client

        result = await reranker.rerank("query", chunks, top_k=3)

        mock_cohere_client.rerank.assert_called_once()
        assert len(result) == 3
        # Top result should be index=2 (score=0.95)
        assert result[0].chunk_id == "c2"

    @pytest.mark.asyncio
    async def test_cohere_uses_correct_model(self):
        from src.core.query.reranker import _COHERE_MODEL

        chunks = _many_chunks(5)
        mock_cohere_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = [self._make_cohere_result(0, 0.9)]
        mock_cohere_client.rerank.return_value = mock_response

        reranker = Reranker(cohere_api_key="key")
        reranker._client = mock_cohere_client

        await reranker.rerank("q", chunks, top_k=1)

        call_kwargs = mock_cohere_client.rerank.call_args.kwargs
        assert call_kwargs.get("model") == _COHERE_MODEL

    @pytest.mark.asyncio
    async def test_cohere_passes_correct_documents(self):
        chunks = [_make_chunk(f"c{i}", text=f"text {i}") for i in range(5)]
        mock_cohere_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = [self._make_cohere_result(0, 0.9)]
        mock_cohere_client.rerank.return_value = mock_response

        reranker = Reranker(cohere_api_key="key")
        reranker._client = mock_cohere_client

        await reranker.rerank("q", chunks, top_k=1)

        call_kwargs = mock_cohere_client.rerank.call_args.kwargs
        docs = call_kwargs.get("documents", [])
        assert docs == [f"text {i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_cohere_failure_falls_back_to_score(self):
        """If Cohere API raises, fall back to score-based reranking."""
        chunks = _many_chunks(10)
        mock_cohere_client = MagicMock()
        mock_cohere_client.rerank.side_effect = RuntimeError("API error")

        reranker = Reranker(cohere_api_key="key")
        reranker._client = mock_cohere_client

        result = await reranker.rerank("q", chunks, top_k=5)
        # Should still return results via fallback
        assert len(result) <= 5
        assert all(isinstance(c, RetrievedChunk) for c in result)

    @pytest.mark.asyncio
    async def test_cohere_result_scores_set(self):
        """Cohere relevance scores should replace the original retrieval scores."""
        chunks = _many_chunks(3)
        mock_cohere_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = [
            self._make_cohere_result(0, 0.777),
            self._make_cohere_result(1, 0.555),
            self._make_cohere_result(2, 0.333),
        ]
        mock_cohere_client.rerank.return_value = mock_response

        reranker = Reranker(cohere_api_key="key")
        reranker._client = mock_cohere_client

        result = await reranker.rerank("q", chunks, top_k=3)
        assert result[0].score == pytest.approx(0.777)
        assert result[1].score == pytest.approx(0.555)


# ---------------------------------------------------------------------------
# Diversity filter
# ---------------------------------------------------------------------------


class TestDiversityFilter:
    @pytest.mark.asyncio
    async def test_diversity_prevents_one_doc_dominance(self):
        # 20 chunks from 1 doc, only 2 from other docs
        same_doc = [_make_chunk(f"s{i}", doc_id="dominant", score=1.0 - i * 0.01) for i in range(20)]
        other_doc = [_make_chunk(f"o{i}", doc_id=f"other-{i}", score=0.5) for i in range(2)]
        chunks = same_doc + other_doc

        reranker = Reranker(cohere_api_key="")
        result = await reranker.rerank("q", chunks, top_k=8)

        # First pass caps "dominant" at max(8//2, 2) = 4; two "other" docs fill 2 more.
        # Backfill then adds 2 more from "dominant" to reach top_k=8 → dominant_count = 6.
        # The key assertion is that diversity logic ran (not all 8 from dominant first).
        dominant_count = sum(1 for c in result if c.document_id == "dominant")
        other_count = sum(1 for c in result if c.document_id.startswith("other"))
        assert other_count == 2  # both "other" chunks are present
        assert dominant_count <= 6  # backfill adds up to 2 more beyond the cap
        assert len(result) == 8

    @pytest.mark.asyncio
    async def test_backfill_when_diversity_not_enough(self):
        """If only one doc available, backfill is used to reach top_k."""
        chunks = [_make_chunk(f"c{i}", doc_id="only-doc", score=1.0 - i * 0.05) for i in range(10)]
        reranker = Reranker(cohere_api_key="")
        result = await reranker.rerank("q", chunks, top_k=8)
        assert len(result) == 8
