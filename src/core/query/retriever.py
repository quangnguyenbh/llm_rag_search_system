"""Hybrid retrieval: dense vector search + sparse (SPLADE/BM25) with RRF fusion."""

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict
    chunk_type: str = "text"  # "text" or "table"


class HybridRetriever:
    def __init__(self, qdrant_client, dense_collection: str, sparse_collection: str | None = None):
        self.qdrant = qdrant_client
        self.dense_collection = dense_collection
        self.sparse_collection = sparse_collection

    async def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 50,
    ) -> list[RetrievedChunk]:
        """Execute hybrid search and fuse results with RRF."""
        # TODO: Embed query (dense)
        # TODO: Encode query (sparse / SPLADE)
        # TODO: Dense search in Qdrant
        # TODO: Sparse search in Qdrant
        # TODO: Table collection search
        # TODO: Apply metadata filters
        # TODO: Reciprocal Rank Fusion
        return []

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[RetrievedChunk]],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """Fuse multiple ranked lists using RRF(k)."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for results in result_lists:
            for rank, chunk in enumerate(results):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
                chunk_map[chunk.chunk_id] = chunk

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=cid,
                document_id=chunk_map[cid].document_id,
                text=chunk_map[cid].text,
                score=scores[cid],
                metadata=chunk_map[cid].metadata,
                chunk_type=chunk_map[cid].chunk_type,
            )
            for cid in sorted_ids
        ]
