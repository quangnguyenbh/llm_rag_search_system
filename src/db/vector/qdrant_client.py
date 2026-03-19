"""Qdrant vector database client — supports Qdrant Cloud and local."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import settings

if TYPE_CHECKING:
    from src.core.ingestion.chunker import Chunk

logger = structlog.get_logger()

# Name used for the sparse vector field inside the main collection
_SPARSE_VECTOR_NAME = "sparse"


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant Cloud (URL+key) or fall back to local host:port."""
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def init_collection(client: QdrantClient) -> None:
    """Create the main collection (dense + sparse vectors) if it doesn't exist."""
    collection = settings.qdrant_collection
    existing = [c.name for c in client.get_collections().collections]

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dimensions,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        client.create_payload_index(
            collection_name=collection,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("qdrant.collection_created", collection=collection)
    else:
        logger.info("qdrant.collection_exists", collection=collection)


def init_tables_collection(client: QdrantClient) -> None:
    """Create the tables collection if it doesn't exist."""
    collection = settings.qdrant_tables_collection
    existing = [c.name for c in client.get_collections().collections]

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
        for field in ("table_id", "doc_id"):
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info("qdrant.tables_collection_created", collection=collection)
    else:
        logger.info("qdrant.tables_collection_exists", collection=collection)


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    source_file: str = "",
    title: str = "",
    sparse_vectors: list[dict[int, float]] | None = None,
) -> int:
    """Upsert chunk vectors + payload into Qdrant. Returns count upserted."""
    collection = settings.qdrant_collection

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        vector_payload: dict = {
            "dense": embedding,
        }
        if sparse_vectors and i < len(sparse_vectors) and sparse_vectors[i]:
            sv = sparse_vectors[i]
            vector_payload[_SPARSE_VECTOR_NAME] = SparseVector(
                indices=list(sv.keys()),
                values=list(sv.values()),
            )
        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector=vector_payload,
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
    """Search for similar chunks using the dense vector. Returns list of {id, score, payload}."""
    collection = settings.qdrant_collection

    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        using="dense",
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


def search_chunks_sparse(
    client: QdrantClient,
    sparse_vector: dict[int, float],
    limit: int = 10,
    document_id: str | None = None,
) -> list[dict]:
    """Search using the sparse (SPLADE) vector. Returns list of {id, score, payload}."""
    collection = settings.qdrant_collection

    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    results = client.query_points(
        collection_name=collection,
        query=SparseVector(
            indices=list(sparse_vector.keys()),
            values=list(sparse_vector.values()),
        ),
        using=_SPARSE_VECTOR_NAME,
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


def search_table_chunks(
    client: QdrantClient,
    query_vector: list[float],
    limit: int = 10,
    document_id: str | None = None,
) -> list[dict]:
    """Search the tables collection. Returns list of {id, score, payload}."""
    collection = settings.qdrant_tables_collection

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        return []

    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=document_id))]
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
