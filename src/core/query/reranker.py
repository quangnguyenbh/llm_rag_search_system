"""Cross-encoder reranking for retrieval precision."""

from src.core.query.retriever import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """Rerank candidates using a cross-encoder model."""
        if not candidates:
            return []
        # TODO: Score with cross-encoder / Cohere Rerank API
        # TODO: Apply diversity filter (ensure multi-doc coverage)
        return candidates[:top_k]
