"""Reranking for retrieval precision — score-based with diversity."""

from src.core.query.retriever import RetrievedChunk


class Reranker:
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """Rerank candidates by score, ensuring multi-document diversity."""
        if not candidates:
            return []

        # Sort by retrieval score descending
        sorted_chunks = sorted(candidates, key=lambda c: c.score, reverse=True)

        # Ensure diversity: don't let one document dominate
        selected: list[RetrievedChunk] = []
        doc_counts: dict[str, int] = {}
        max_per_doc = max(top_k // 2, 2)

        for chunk in sorted_chunks:
            doc_id = chunk.document_id
            if doc_counts.get(doc_id, 0) < max_per_doc:
                selected.append(chunk)
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            if len(selected) >= top_k:
                break

        # If we didn't fill top_k due to diversity, backfill
        if len(selected) < top_k:
            seen_ids = {c.chunk_id for c in selected}
            for chunk in sorted_chunks:
                if chunk.chunk_id not in seen_ids:
                    selected.append(chunk)
                    if len(selected) >= top_k:
                        break

        return selected
