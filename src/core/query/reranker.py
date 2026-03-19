"""Reranking for retrieval precision — Cohere Rerank API with score-based fallback."""

from __future__ import annotations

import asyncio
import os

import structlog

from src.core.query.retriever import RetrievedChunk

logger = structlog.get_logger()

_COHERE_MODEL = "rerank-english-v3.0"
_RERANK_TOP_N = 50  # candidates sent to Cohere
_DIVERSITY_MIN_DOCS = 3  # target minimum unique documents in final results


class Reranker:
    """Cross-encoder reranker using the Cohere Rerank API.

    If ``COHERE_API_KEY`` is not set, falls back to score-based reranking
    with the same diversity filter.
    """

    def __init__(self, cohere_api_key: str | None = None) -> None:
        self._api_key = cohere_api_key or os.environ.get("COHERE_API_KEY", "")
        self._client = None

    def _get_cohere_client(self):
        if self._client is None:
            try:
                import cohere  # type: ignore[import]

                self._client = cohere.Client(self._api_key)
            except ImportError:
                logger.warning("reranker.cohere_not_installed")
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """Rerank *candidates* and return the top *top_k* chunks.

        Uses the Cohere Rerank API when an API key is available, otherwise
        falls back to score-based ordering.
        """
        if not candidates:
            return []

        if self._api_key:
            try:
                return await self._cohere_rerank(query, candidates, top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "reranker.cohere_failed_fallback",
                    error=str(exc),
                )

        logger.warning("reranker.no_cohere_key_fallback")
        return self._score_based_rerank(candidates, top_k)

    # ------------------------------------------------------------------
    # Cohere reranking
    # ------------------------------------------------------------------

    async def _cohere_rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Call the Cohere Rerank API asynchronously."""
        client = self._get_cohere_client()
        if client is None:
            return self._score_based_rerank(candidates, top_k)

        # Send at most _RERANK_TOP_N candidates to Cohere
        pool = candidates[:_RERANK_TOP_N]
        docs = [c.text for c in pool]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.rerank(
                query=query,
                documents=docs,
                model=_COHERE_MODEL,
                top_n=min(_RERANK_TOP_N, len(docs)),
            ),
        )

        # Build score-annotated list preserving original chunk metadata
        reranked: list[RetrievedChunk] = []
        for result in response.results:
            chunk = pool[result.index]
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=result.relevance_score,
                    metadata=chunk.metadata,
                    chunk_type=chunk.chunk_type,
                )
            )

        return self._apply_diversity_filter(reranked, top_k)

    # ------------------------------------------------------------------
    # Score-based fallback
    # ------------------------------------------------------------------

    def _score_based_rerank(
        self,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank candidates by score, ensuring multi-document diversity."""
        sorted_chunks = sorted(candidates, key=lambda c: c.score, reverse=True)
        return self._apply_diversity_filter(sorted_chunks, top_k)

    def _apply_diversity_filter(
        self,
        sorted_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Select top_k chunks ensuring results span multiple documents where possible."""
        selected: list[RetrievedChunk] = []
        doc_counts: dict[str, int] = {}
        max_per_doc = max(top_k // 2, 2)

        # First pass: enforce per-document cap to ensure diversity
        for chunk in sorted_chunks:
            doc_id = chunk.document_id
            if doc_counts.get(doc_id, 0) < max_per_doc:
                selected.append(chunk)
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            if len(selected) >= top_k:
                break

        # Second pass: backfill if diversity pass didn't fill top_k
        if len(selected) < top_k:
            seen_ids = {c.chunk_id for c in selected}
            for chunk in sorted_chunks:
                if chunk.chunk_id not in seen_ids:
                    selected.append(chunk)
                    if len(selected) >= top_k:
                        break

        # Log if diversity goal not met
        unique_docs = len({c.document_id for c in selected})
        if unique_docs < _DIVERSITY_MIN_DOCS and len(selected) >= _DIVERSITY_MIN_DOCS:
            logger.warning(
                "reranker.low_diversity",
                unique_docs=unique_docs,
                target=_DIVERSITY_MIN_DOCS,
            )

        return selected
