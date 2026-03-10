"""Dense vector retrieval via Qdrant."""

from dataclasses import dataclass

import structlog

from src.core.ingestion.embedder import BatchEmbedder
from src.db.vector.qdrant_client import search_chunks

logger = structlog.get_logger()


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict
    chunk_type: str = "text"


class Retriever:
    def __init__(self, qdrant_client, embedder: BatchEmbedder):
        self.qdrant = qdrant_client
        self.embedder = embedder

    async def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Embed query and search Qdrant for similar chunks."""
        query_vector = await self.embedder.embed_query(query)

        document_id = filters.get("document_id") if filters else None

        hits = search_chunks(
            client=self.qdrant,
            query_vector=query_vector,
            limit=top_k,
            document_id=document_id,
        )

        results = []
        for hit in hits:
            payload = hit.get("payload", {})
            results.append(
                RetrievedChunk(
                    chunk_id=hit["id"],
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    score=hit["score"],
                    metadata={
                        "title": payload.get("title", ""),
                        "page_number": payload.get("page_number"),
                        "section_path": payload.get("section_path", ""),
                        "source_file": payload.get("source_file", ""),
                    },
                )
            )

        logger.info("retriever.search_complete", query_len=len(query), results=len(results))
        return results
