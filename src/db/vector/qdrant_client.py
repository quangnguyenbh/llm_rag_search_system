"""Qdrant vector database client."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


DENSE_COLLECTION = "manual_chunks_dense"
SPARSE_COLLECTION = "manual_chunks_sparse"
TABLE_COLLECTION = "manual_tables"

DENSE_VECTOR_SIZE = 3072  # text-embedding-3-large


def init_collections(client: QdrantClient) -> None:
    """Initialize Qdrant collections if they don't exist."""
    existing = [c.name for c in client.get_collections().collections]

    if DENSE_COLLECTION not in existing:
        client.create_collection(
            collection_name=DENSE_COLLECTION,
            vectors_config=VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),
        )

    if TABLE_COLLECTION not in existing:
        client.create_collection(
            collection_name=TABLE_COLLECTION,
            vectors_config=VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),
        )
