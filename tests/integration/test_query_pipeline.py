"""Integration tests for the full query pipeline."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.query.analyzer import QueryAnalyzer
from src.core.query.context_builder import ContextBuilder
from src.core.query.pipeline import QueryPipeline, QueryResult
from src.core.query.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "c1",
    doc_id: str = "doc-1",
    text: str = "Sample text",
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        score=score,
        metadata={"title": "Test Doc", "page_number": 1, "section_path": "Intro"},
    )


def _build_pipeline(
    chunks: list[RetrievedChunk] | None = None,
    answer: str = "The answer is 42.",
) -> QueryPipeline:
    chunks = chunks or [_make_chunk()]

    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=chunks)

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=chunks[:8])

    mock_generator = MagicMock()
    from src.core.query.generator import GenerationResult

    mock_generator.generate = AsyncMock(
        return_value=GenerationResult(answer=answer, model="nova-lite")
    )

    async def _stream_gen(*args, **kwargs):
        for word in answer.split():
            yield word

    mock_generator.generate_stream = _stream_gen

    from src.core.query.citation import CitationVerifier
    from src.core.query.model_router import ModelRouter

    mock_verifier = MagicMock(spec=CitationVerifier)
    mock_verifier.verify = MagicMock(
        return_value=MagicMock(
            answer=answer, citations=[{"source": "doc-1"}], confidence=0.9
        )
    )

    return QueryPipeline(
        analyzer=QueryAnalyzer(),
        retriever=mock_retriever,
        reranker=mock_reranker,
        context_builder=ContextBuilder(max_tokens=1000),
        generator=mock_generator,
        citation_verifier=mock_verifier,
        model_router=ModelRouter(),
    )


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestQueryPipeline:
    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        pipeline = _build_pipeline(answer="The answer is 42.")
        result = await pipeline.execute("What is the answer?")

        assert isinstance(result, QueryResult)
        assert result.answer == "The answer is 42."
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_execute_with_filters(self):
        pipeline = _build_pipeline()
        result = await pipeline.execute("Query", filters={"document_id": "doc-1"})
        assert result.answer

    @pytest.mark.asyncio
    async def test_search_only_returns_chunks(self):
        chunks = [_make_chunk(f"c{i}", score=1.0 - i * 0.1) for i in range(5)]
        pipeline = _build_pipeline(chunks=chunks)
        results = await pipeline.search_only("test query", top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert "chunk_id" in results[0]
        assert "text" in results[0]
        assert "score" in results[0]

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        pipeline = _build_pipeline(answer="Hello world response")
        events = []
        async for event in pipeline.stream("Hello?"):
            events.append(event)

        types = [e["type"] for e in events]
        assert "token" in types
        assert "sources" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_stream_sources_first(self):
        pipeline = _build_pipeline()
        events = []
        async for event in pipeline.stream("What?"):
            events.append(event)

        assert events[0]["type"] == "sources"

    @pytest.mark.asyncio
    async def test_stream_done_last(self):
        pipeline = _build_pipeline(answer="Final answer")
        events = []
        async for event in pipeline.stream("?"):
            events.append(event)

        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_pipeline_calls_retriever_with_filters(self):
        pipeline = _build_pipeline()
        await pipeline.execute("q", filters={"document_id": "d1"})
        pipeline.retriever.search.assert_called_once()
        _, call_kwargs = pipeline.retriever.search.call_args
        assert call_kwargs.get("filters") == {"document_id": "d1"}


# ---------------------------------------------------------------------------
# SSE streaming endpoint
# ---------------------------------------------------------------------------


class TestSSEEndpoint:
    @pytest.mark.asyncio
    async def test_stream_endpoint_content_type(self):
        """The /stream route should return text/event-stream."""
        from fastapi.testclient import TestClient
        from src.main import app

        # Mock the pipeline
        async def fake_stream(*args, **kwargs):
            yield {"type": "sources", "data": []}
            yield {"type": "token", "data": "hello"}
            yield {"type": "done", "data": ""}

        mock_pipeline = MagicMock()
        mock_pipeline.stream = fake_stream

        with patch("src.api.routes.query._build_pipeline", return_value=mock_pipeline):
            client = TestClient(app)
            response = client.post(
                "/v1/query/stream",
                json={"question": "test"},
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# Redis cache
# ---------------------------------------------------------------------------


class TestQueryCache:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        from src.db.cache.redis_client import QueryCache

        cache = QueryCache()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=json.dumps({"answer": "cached"}))
        mock_client.setex = AsyncMock()
        cache._client = mock_client

        result = await cache.get_query("what is X?")
        assert result == {"answer": "cached"}

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        from src.db.cache.redis_client import QueryCache

        cache = QueryCache()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=None)
        cache._client = mock_client

        result = await cache.get_query("unseen query")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set(self):
        from src.db.cache.redis_client import QueryCache

        cache = QueryCache()
        mock_client = MagicMock()
        mock_client.setex = AsyncMock()
        cache._client = mock_client

        await cache.set_query("test query", {"answer": "42"})
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        stored = json.loads(call_args[0][2])
        assert stored["answer"] == "42"

    @pytest.mark.asyncio
    async def test_cache_ttl_respected(self):
        from src.db.cache.redis_client import QueryCache

        cache = QueryCache(query_ttl=999)
        mock_client = MagicMock()
        mock_client.setex = AsyncMock()
        cache._client = mock_client

        await cache.set_query("q", {"data": "x"})
        call_args = mock_client.setex.call_args
        assert call_args[0][1] == 999

    @pytest.mark.asyncio
    async def test_cache_graceful_on_redis_unavailable(self):
        from src.db.cache.redis_client import QueryCache

        cache = QueryCache()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        cache._client = mock_client

        # Should not raise
        result = await cache.get_query("anything")
        assert result is None

    @pytest.mark.asyncio
    async def test_embedding_cache_round_trip(self):
        from src.db.cache.redis_client import QueryCache

        cache = QueryCache()
        mock_client = MagicMock()
        stored: dict = {}

        async def setex(key, ttl, value):
            stored[key] = value

        async def get(key):
            return stored.get(key)

        mock_client.setex = setex
        mock_client.get = get
        cache._client = mock_client

        vector = [0.1, 0.2, 0.3]
        await cache.set_embedding("hello world", vector)
        result = await cache.get_embedding("hello world")

        assert result == vector

    @pytest.mark.asyncio
    async def test_embedding_cache_key_is_deterministic(self):
        from src.db.cache.redis_client import QueryCache, _embed_cache_key

        key1 = _embed_cache_key("same text")
        key2 = _embed_cache_key("same text")
        assert key1 == key2
        assert key1.startswith("embed:")

    @pytest.mark.asyncio
    async def test_query_cache_key_includes_filters(self):
        from src.db.cache.redis_client import _query_cache_key

        key_no_filter = _query_cache_key("q")
        key_with_filter = _query_cache_key("q", {"document_id": "doc-1"})
        assert key_no_filter != key_with_filter
