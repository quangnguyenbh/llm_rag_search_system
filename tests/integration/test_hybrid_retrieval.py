"""Integration tests for hybrid retrieval and RRF fusion."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.query.retriever import RetrievedChunk, Retriever, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, doc_id: str = "doc-1", score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"Text for {chunk_id}",
        score=score,
        metadata={"title": "Test Doc"},
    )


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_single_list(self):
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        fused = reciprocal_rank_fusion([chunks])
        # All chunks from the single list, same relative order
        assert [c.chunk_id for c in fused] == [c.chunk_id for c in chunks]

    def test_two_lists_deduplication(self):
        list_a = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
        list_b = [_make_chunk("c2"), _make_chunk("c4"), _make_chunk("c1")]
        fused = reciprocal_rank_fusion([list_a, list_b])
        # All unique ids present
        ids = [c.chunk_id for c in fused]
        assert sorted(ids) == ["c1", "c2", "c3", "c4"]
        # c1 appears in rank 0 of list_a and rank 2 of list_b
        # c2 appears in rank 1 of list_a and rank 0 of list_b
        # Both have high RRF scores; exact order depends on k=60
        assert len(ids) == 4

    def test_chunk_in_all_lists_gets_highest_score(self):
        list_a = [_make_chunk("winner"), _make_chunk("c2"), _make_chunk("c3")]
        list_b = [_make_chunk("winner"), _make_chunk("c4"), _make_chunk("c5")]
        list_c = [_make_chunk("winner"), _make_chunk("c6"), _make_chunk("c7")]
        fused = reciprocal_rank_fusion([list_a, list_b, list_c])
        assert fused[0].chunk_id == "winner"

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_rrf_k_parameter(self):
        list_a = [_make_chunk("c1")]
        fused_k1 = reciprocal_rank_fusion([list_a], k=1)
        fused_k100 = reciprocal_rank_fusion([list_a], k=100)
        # k=1 → score = 1/(1+0+1) = 0.5;  k=100 → score = 1/(100+0+1) ≈ 0.0099
        assert fused_k1[0].score > fused_k100[0].score

    def test_scores_are_positive(self):
        chunks = [_make_chunk(f"c{i}") for i in range(10)]
        fused = reciprocal_rank_fusion([chunks, list(reversed(chunks))])
        for c in fused:
            assert c.score > 0

    def test_ordering_is_descending(self):
        list_a = [_make_chunk(f"a{i}") for i in range(5)]
        list_b = [_make_chunk(f"b{i}") for i in range(5)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        scores = [c.score for c in fused]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


class TestRetriever:
    @pytest.fixture
    def mock_qdrant(self):
        client = MagicMock()
        # Return empty collections list so table collection check passes gracefully
        client.get_collections.return_value = MagicMock(collections=[])
        return client

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed_query = AsyncMock(return_value=[0.1] * 4)
        return embedder

    @pytest.fixture
    def mock_sparse_encoder(self):
        enc = MagicMock()
        enc.encode.return_value = [{0: 0.5, 1: 0.3}]
        return enc

    def _make_qdrant_hit(self, chunk_id: str, score: float = 0.9):
        hit = MagicMock()
        hit.id = chunk_id
        hit.score = score
        hit.payload = {
            "document_id": "doc-1",
            "text": f"Text {chunk_id}",
            "title": "Test",
            "page_number": 1,
            "section_path": "Intro",
            "source_file": "test.pdf",
        }
        return hit

    @pytest.mark.asyncio
    async def test_dense_only_search(self, mock_qdrant, mock_embedder):
        """Dense-only mode returns chunks from the main collection."""
        with patch("src.config.settings.hybrid_retrieval_enabled", False):
            hit = self._make_qdrant_hit("chunk-1")
            mock_qdrant.query_points.return_value = MagicMock(points=[hit])

            retriever = Retriever(qdrant_client=mock_qdrant, embedder=mock_embedder)
            results = await retriever.search("test query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"

    @pytest.mark.asyncio
    async def test_hybrid_search_merges_results(
        self, mock_qdrant, mock_embedder, mock_sparse_encoder
    ):
        """Hybrid mode calls both dense and sparse search, then RRF-fuses."""
        # Dense returns chunk-1; sparse returns chunk-2
        dense_hit = self._make_qdrant_hit("chunk-1", score=0.9)
        sparse_hit = self._make_qdrant_hit("chunk-2", score=0.8)

        call_count = {"n": 0}

        def query_points(**kwargs):
            call_count["n"] += 1
            using = kwargs.get("using", "dense")
            if using == "sparse":
                return MagicMock(points=[sparse_hit])
            return MagicMock(points=[dense_hit])

        mock_qdrant.query_points.side_effect = query_points

        with patch("src.config.settings.hybrid_retrieval_enabled", True):
            retriever = Retriever(
                qdrant_client=mock_qdrant,
                embedder=mock_embedder,
                sparse_encoder=mock_sparse_encoder,
            )
            results = await retriever.search("test query", top_k=10)

        # Both chunk-1 and chunk-2 present in fused results
        ids = [r.chunk_id for r in results]
        assert "chunk-1" in ids
        assert "chunk-2" in ids

    @pytest.mark.asyncio
    async def test_hybrid_search_sparse_failure_graceful(
        self, mock_qdrant, mock_embedder
    ):
        """If sparse search fails, hybrid falls back to dense-only results."""
        dense_hit = self._make_qdrant_hit("chunk-dense")

        def query_points(**kwargs):
            if kwargs.get("using") == "sparse":
                raise RuntimeError("sparse not supported")
            return MagicMock(points=[dense_hit])

        mock_qdrant.query_points.side_effect = query_points

        enc = MagicMock()
        enc.encode.return_value = [{0: 0.5}]

        with patch("src.config.settings.hybrid_retrieval_enabled", True):
            retriever = Retriever(
                qdrant_client=mock_qdrant,
                embedder=mock_embedder,
                sparse_encoder=enc,
            )
            results = await retriever.search("test query")

        assert any(r.chunk_id == "chunk-dense" for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_returns_at_most_top_k(
        self, mock_qdrant, mock_embedder, mock_sparse_encoder
    ):
        """Fused results are capped at top_k."""
        hits = [self._make_qdrant_hit(f"chunk-{i}", score=1.0 - i * 0.05) for i in range(20)]

        mock_qdrant.query_points.return_value = MagicMock(points=hits)

        with patch("src.config.settings.hybrid_retrieval_enabled", True):
            retriever = Retriever(
                qdrant_client=mock_qdrant,
                embedder=mock_embedder,
                sparse_encoder=mock_sparse_encoder,
            )
            results = await retriever.search("query", top_k=5)

        assert len(results) <= 5
