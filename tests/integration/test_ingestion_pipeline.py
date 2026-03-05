"""Integration test: full ingestion pipeline with mocked embedder + Qdrant."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from src.core.ingestion.pipeline import IngestionPipeline, IngestionResult
from src.core.ingestion.chunker import SemanticChunker
from src.core.ingestion.metadata import MetadataExtractor
from src.core.ingestion.embedder import BatchEmbedder

PDF_PATH = Path("data/raw/huggingface/kaizen9_finepdfs_en/BTTM-803.pdf")


@pytest.fixture
def mock_embedder():
    embedder = BatchEmbedder()
    embedder.embed_batch = AsyncMock(side_effect=_fake_embed_batch)
    return embedder


@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    client.upsert = MagicMock()
    return client


async def _fake_embed_batch(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake embeddings (dimension 4) for each text."""
    return [[float(i), 0.1, 0.2, 0.3] for i in range(len(texts))]


@pytest.fixture
def pipeline(mock_embedder, mock_qdrant):
    return IngestionPipeline(
        chunker=SemanticChunker(),
        metadata_extractor=MetadataExtractor(),
        embedder=mock_embedder,
        qdrant=mock_qdrant,
    )


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_ingest_real_pdf(self, pipeline, mock_embedder, mock_qdrant):
        """Parse a real PDF → chunk → mock embed → mock upsert."""
        if not PDF_PATH.exists():
            pytest.skip("Test PDF not available")

        result = await pipeline.ingest(PDF_PATH)

        assert isinstance(result, IngestionResult)
        assert result.chunks_count > 0
        assert result.vectors_upserted == result.chunks_count
        assert result.title != ""
        assert result.document_id != ""

        # Embedder was called with the right number of texts
        embed_call_args = mock_embedder.embed_batch.call_args
        assert len(embed_call_args[0][0]) == result.chunks_count

        # Qdrant upsert was called
        mock_qdrant.upsert.assert_called_once()
        points = mock_qdrant.upsert.call_args.kwargs["points"]
        assert len(points) == result.chunks_count

    @pytest.mark.asyncio
    async def test_ingest_unsupported_format(self, pipeline):
        with pytest.raises(ValueError, match="Unsupported file type"):
            await pipeline.ingest(Path("test.xyz"))

    @pytest.mark.asyncio
    async def test_upserted_payload_has_required_fields(self, pipeline, mock_qdrant):
        """Verify each upserted point has all required payload fields."""
        if not PDF_PATH.exists():
            pytest.skip("Test PDF not available")

        await pipeline.ingest(PDF_PATH)

        points = mock_qdrant.upsert.call_args.kwargs["points"]
        required_fields = {
            "document_id", "text", "page_number", "section_path",
            "heading_hierarchy", "token_count", "title", "source_file",
        }
        for point in points:
            assert required_fields.issubset(point.payload.keys()), (
                f"Missing fields: {required_fields - point.payload.keys()}"
            )

    @pytest.mark.asyncio
    async def test_ingestion_result_fields(self, pipeline):
        if not PDF_PATH.exists():
            pytest.skip("Test PDF not available")

        result = await pipeline.ingest(PDF_PATH, doc_metadata={"source": "test"})

        assert result.file_path == str(PDF_PATH)
        assert result.chunks_count == result.vectors_upserted
