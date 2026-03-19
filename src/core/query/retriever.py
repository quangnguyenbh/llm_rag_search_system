"""Dense + sparse hybrid retrieval via Qdrant with RRF fusion."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.config import settings
from src.core.ingestion.embedder import BatchEmbedder
from src.core.ingestion.sparse_encoder import SparseEncoder
from src.db.vector.qdrant_client import search_chunks, search_chunks_sparse, search_table_chunks

logger = structlog.get_logger()


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict
    chunk_type: str = "text"


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    result_lists:
        Each inner list is a ranked list of :class:`RetrievedChunk` objects.
    k:
        RRF smoothing constant (default 60 per the original paper).

    Returns
    -------
    list[RetrievedChunk]
        Deduplicated chunks ordered by RRF score descending.  The ``score``
        field on each chunk is replaced with the RRF score.
    """
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                k + rank + 1
            )
            if chunk.chunk_id not in chunk_by_id:
                chunk_by_id[chunk.chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    result: list[RetrievedChunk] = []
    for cid in sorted_ids:
        chunk = chunk_by_id[cid]
        result.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=rrf_scores[cid],
                metadata=chunk.metadata,
                chunk_type=chunk.chunk_type,
            )
        )
    return result


def _hits_to_chunks(hits: list[dict], chunk_type: str = "text") -> list[RetrievedChunk]:
    results: list[RetrievedChunk] = []
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
                chunk_type=chunk_type,
            )
        )
    return results


class Retriever:
    """Retriever supporting dense-only and hybrid (dense + sparse + table) modes.

    Hybrid mode is enabled when ``settings.hybrid_retrieval_enabled`` is ``True``
    (the default).  In hybrid mode, results from dense search, SPLADE sparse
    search, and the tables collection are combined via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        qdrant_client,
        embedder: BatchEmbedder,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        self.qdrant = qdrant_client
        self.embedder = embedder
        self._sparse_encoder = sparse_encoder or SparseEncoder()

    async def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Search for relevant chunks.

        When ``settings.hybrid_retrieval_enabled`` is ``True``, runs dense +
        sparse + table searches and merges them with RRF.  Otherwise, falls
        back to dense-only search (original behaviour).
        """
        if settings.hybrid_retrieval_enabled:
            return await self._hybrid_search(query, filters=filters, top_k=top_k)
        return await self._dense_search(query, filters=filters, top_k=top_k)

    # ------------------------------------------------------------------
    # Dense-only (original behaviour)
    # ------------------------------------------------------------------

    async def _dense_search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        query_vector = await self.embedder.embed_query(query)
        document_id = filters.get("document_id") if filters else None
        hits = search_chunks(
            client=self.qdrant,
            query_vector=query_vector,
            limit=top_k,
            document_id=document_id,
        )
        results = _hits_to_chunks(hits)
        logger.info("retriever.dense_search", query_len=len(query), results=len(results))
        return results

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    async def _hybrid_search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Run dense + sparse + table searches and merge with RRF."""
        document_id = filters.get("document_id") if filters else None

        # 1. Dense search
        query_vector = await self.embedder.embed_query(query)
        dense_hits = search_chunks(
            client=self.qdrant,
            query_vector=query_vector,
            limit=top_k,
            document_id=document_id,
        )
        dense_chunks = _hits_to_chunks(dense_hits, chunk_type="text")

        # 2. Sparse (SPLADE) search
        sparse_vectors = self._sparse_encoder.encode([query])
        sparse_vec = sparse_vectors[0] if sparse_vectors else {}
        sparse_chunks: list[RetrievedChunk] = []
        if sparse_vec:
            try:
                sparse_hits = search_chunks_sparse(
                    client=self.qdrant,
                    sparse_vector=sparse_vec,
                    limit=top_k,
                    document_id=document_id,
                )
                sparse_chunks = _hits_to_chunks(sparse_hits, chunk_type="text")
            except Exception as exc:  # noqa: BLE001
                logger.warning("retriever.sparse_search_failed", error=str(exc))

        # 3. Table search
        table_chunks: list[RetrievedChunk] = []
        try:
            table_hits = search_table_chunks(
                client=self.qdrant,
                query_vector=query_vector,
                limit=top_k,
                document_id=document_id,
            )
            table_chunks = _hits_to_chunks(table_hits, chunk_type="table")
        except Exception as exc:  # noqa: BLE001
            logger.warning("retriever.table_search_failed", error=str(exc))

        # 4. RRF fusion
        result_lists = [lst for lst in [dense_chunks, sparse_chunks, table_chunks] if lst]
        if not result_lists:
            return []

        fused = reciprocal_rank_fusion(result_lists)[:top_k]
        logger.info(
            "retriever.hybrid_search",
            query_len=len(query),
            dense=len(dense_chunks),
            sparse=len(sparse_chunks),
            table=len(table_chunks),
            fused=len(fused),
        )
        return fused
