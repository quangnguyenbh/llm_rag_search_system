"""Unit tests for Qdrant client functions with mocked QdrantClient."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.core.ingestion.chunker import Chunk
from src.db.vector.qdrant_client import (
    init_collection,
    upsert_chunks,
    search_chunks,
    delete_by_document_id,
)


@pytest.fixture
def mock_qdrant():
    return MagicMock()


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk-1",
            text="Document: Test | Section: Intro\n\nHello world.",
            document_id="doc-123",
            page_number=1,
            section_path="Intro",
            heading_hierarchy=["Intro"],
            token_count=10,
        ),
        Chunk(
            chunk_id="chunk-2",
            text="Document: Test | Section: Details\n\nMore content here.",
            document_id="doc-123",
            page_number=2,
            section_path="Details",
            heading_hierarchy=["Details"],
            token_count=15,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


class TestInitCollection:
    def test_creates_collection_when_missing(self, mock_qdrant):
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])

        init_collection(mock_qdrant)

        mock_qdrant.create_collection.assert_called_once()
        call_kwargs = mock_qdrant.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "manual_chunks"

    def test_skips_when_collection_exists(self, mock_qdrant):
        existing = MagicMock()
        existing.name = "manual_chunks"
        mock_qdrant.get_collections.return_value = MagicMock(collections=[existing])

        init_collection(mock_qdrant)

        mock_qdrant.create_collection.assert_not_called()


class TestUpsertChunks:
    def test_upserts_correct_count(self, mock_qdrant, sample_chunks, sample_embeddings):
        count = upsert_chunks(
            client=mock_qdrant,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            source_file="test.pdf",
            title="Test Doc",
        )

        assert count == 2
        mock_qdrant.upsert.assert_called_once()

    def test_point_payload_structure(self, mock_qdrant, sample_chunks, sample_embeddings):
        upsert_chunks(
            client=mock_qdrant,
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            source_file="test.pdf",
            title="Test Doc",
        )

        call_kwargs = mock_qdrant.upsert.call_args.kwargs
        points = call_kwargs["points"]
        assert len(points) == 2

        p = points[0]
        assert p.id == "chunk-1"
        assert p.vector == [0.1, 0.2, 0.3]
        assert p.payload["document_id"] == "doc-123"
        assert p.payload["page_number"] == 1
        assert p.payload["section_path"] == "Intro"
        assert p.payload["title"] == "Test Doc"
        assert p.payload["source_file"] == "test.pdf"
        assert "text" in p.payload

    def test_empty_chunks(self, mock_qdrant):
        count = upsert_chunks(
            client=mock_qdrant,
            chunks=[],
            embeddings=[],
        )
        assert count == 0


class TestSearchChunks:
    def test_basic_search(self, mock_qdrant):
        hit = MagicMock()
        hit.id = "chunk-1"
        hit.score = 0.95
        hit.payload = {"text": "Hello", "document_id": "doc-1"}
        mock_qdrant.query_points.return_value = MagicMock(points=[hit])

        results = search_chunks(mock_qdrant, query_vector=[0.1, 0.2, 0.3], limit=5)

        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["text"] == "Hello"

    def test_search_with_document_filter(self, mock_qdrant):
        mock_qdrant.query_points.return_value = MagicMock(points=[])

        search_chunks(mock_qdrant, query_vector=[0.1], document_id="doc-filter")

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    def test_search_no_filter(self, mock_qdrant):
        mock_qdrant.query_points.return_value = MagicMock(points=[])

        search_chunks(mock_qdrant, query_vector=[0.1])

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None


class TestDeleteByDocumentId:
    def test_calls_delete(self, mock_qdrant):
        delete_by_document_id(mock_qdrant, "doc-to-remove")

        mock_qdrant.delete.assert_called_once()
        call_kwargs = mock_qdrant.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == "manual_chunks"
