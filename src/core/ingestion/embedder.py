"""Batch embedding using OpenAI or self-hosted models."""

import structlog

logger = structlog.get_logger()


class BatchEmbedder:
    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 2048):
        self.model = model
        self.batch_size = batch_size

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, handling rate limiting and retries."""
        if not texts:
            return []

        # TODO: OpenAI embeddings API call with batching
        # TODO: Rate limiting + exponential backoff
        # TODO: Sparse encoding (SPLADE) as separate method
        logger.info("embedder.batch", count=len(texts), model=self.model)
        return []

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        results = await self.embed_batch([text])
        return results[0] if results else []
