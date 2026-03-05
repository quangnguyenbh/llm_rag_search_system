"""E2E test: real PDF → real Bedrock embeddings → real Qdrant Cloud → search.

Uses a dedicated test collection (manual_chunks_test) so data persists
for inspection in the Qdrant Cloud dashboard.

Requires:
    - AWS credentials configured (for Bedrock)
    - QDRANT_URL and QDRANT_API_KEY set in .env
    - Test PDF present at the expected path

Run with:
    PYTHONPATH=. pytest tests/e2e/test_ingest_and_search.py -v -s
"""

import pytest
from pathlib import Path

from src.config import settings
from src.core.ingestion.pipeline import IngestionPipeline
from src.core.ingestion.chunker import SemanticChunker
from src.core.ingestion.metadata import MetadataExtractor
from src.core.ingestion.embedder import BatchEmbedder
from src.db.vector.qdrant_client import (
    get_qdrant_client,
    search_chunks,
)
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

PDF_PATH = Path("data/raw/huggingface/kaizen9_finepdfs_en/BTTM-803.pdf")
TEST_COLLECTION = "manual_chunks_test"

# Skip entire module if infra not configured
pytestmark = pytest.mark.skipif(
    not settings.qdrant_url or not settings.aws_bedrock_region,
    reason="E2E requires QDRANT_URL and AWS Bedrock credentials",
)


@pytest.fixture(scope="module")
def qdrant():
    client = get_qdrant_client()

    # Create test collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if TEST_COLLECTION not in existing:
        client.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=TEST_COLLECTION,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # Point settings to test collection for the duration of the test
    settings.qdrant_collection = TEST_COLLECTION
    return client


@pytest.fixture(scope="module")
def embedder():
    return BatchEmbedder()


@pytest.fixture(scope="module")
def pipeline(qdrant, embedder):
    return IngestionPipeline(
        chunker=SemanticChunker(),
        metadata_extractor=MetadataExtractor(),
        embedder=embedder,
        qdrant=qdrant,
    )


class TestIngestAndSearch:
    _document_id: str | None = None

    @pytest.mark.asyncio
    async def test_01_ingest_pdf(self, pipeline):
        """Ingest a real PDF through the full pipeline."""
        if not PDF_PATH.exists():
            pytest.skip("Test PDF not available")

        result = await pipeline.ingest(PDF_PATH)

        assert result.chunks_count > 0
        assert result.vectors_upserted == result.chunks_count
        TestIngestAndSearch._document_id = result.document_id
        print(f"\nIngested: {result.chunks_count} chunks, doc_id={result.document_id}")

    @pytest.mark.asyncio
    async def test_02_search_returns_relevant_results(self, qdrant, embedder):
        """Search for a query and verify relevant chunks are returned."""
        doc_id = TestIngestAndSearch._document_id
        if not doc_id:
            pytest.skip("Ingest test must run first")

        query_vector = await embedder.embed_query("What is pilgrimage tourism in India?")
        assert len(query_vector) == settings.embedding_dimensions

        results = search_chunks(
            client=qdrant,
            query_vector=query_vector,
            limit=5,
            document_id=doc_id,
        )

        assert len(results) > 0
        top_result = results[0]
        assert top_result["score"] > 0.0
        assert "text" in top_result["payload"]
        assert top_result["payload"]["document_id"] == doc_id
        print(f"\nTop result (score={top_result['score']:.4f}): {top_result['payload']['text'][:100]}...")

    @pytest.mark.asyncio
    async def test_03_search_with_different_query(self, qdrant, embedder):
        """Search for a different query to verify vector diversity."""
        doc_id = TestIngestAndSearch._document_id
        if not doc_id:
            pytest.skip("Ingest test must run first")

        query_vector = await embedder.embed_query("Buddhism main teachings")
        results = search_chunks(qdrant, query_vector, limit=3, document_id=doc_id)

        assert len(results) > 0
        # The text should mention Buddhism
        texts = " ".join(r["payload"]["text"] for r in results)
        assert "buddhism" in texts.lower() or "unit" in texts.lower()
