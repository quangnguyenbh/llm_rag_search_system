"""Qdrant vector database client — supports Qdrant Cloud and local."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

import structlog

from src.config import settings
from src.core.ingestion.chunker import Chunk

logger = structlog.get_logger()


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant Cloud (URL+key) or fall back to local host:port."""
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def init_collection(client: QdrantClient) -> None:
    """Create the collection if it doesn't exist."""
    collection = settings.qdrant_collection
    existing = [c.name for c in client.get_collections().collections]

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=collection,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("qdrant.collection_created", collection=collection)
    else:
        logger.info("qdrant.collection_exists", collection=collection)


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    source_file: str = "",
    title: str = "",
) -> int:
    """Upsert chunk vectors + payload into Qdrant. Returns count upserted."""
    collection = settings.qdrant_collection

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "section_path": chunk.section_path,
                    "heading_hierarchy": chunk.heading_hierarchy,
                    "token_count": chunk.token_count,
                    "title": title,
                    "source_file": source_file,
                },
            )
        )

    # Qdrant client handles batching internally for large sets
    client.upsert(collection_name=collection, points=points)

    logger.info("qdrant.upserted", collection=collection, count=len(points))
    return len(points)


def search_chunks(
    client: QdrantClient,
    query_vector: list[float],
    limit: int = 10,
    document_id: str | None = None,
) -> list[dict]:
    """Search for similar chunks. Returns list of {id, score, payload}."""
    collection = settings.qdrant_collection

    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results.points
    ]


def delete_by_document_id(client: QdrantClient, document_id: str) -> None:
    """Delete all points for a given document (for re-ingestion or cleanup)."""
    collection = settings.qdrant_collection
    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )
    logger.info("qdrant.deleted", collection=collection, document_id=document_id)
