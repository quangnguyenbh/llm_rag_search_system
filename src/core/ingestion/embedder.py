"""Batch embedding with configurable provider (AWS Bedrock / OpenAI)."""

import json
import asyncio
import random

import boto3
import structlog

from src.config import settings

logger = structlog.get_logger()


class BatchEmbedder:
    def __init__(
        self,
        provider: str | None = None,
        model_id: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
    ):
        self.provider = provider or settings.embedding_provider
        self.model_id = model_id or settings.embedding_model_id
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_retries = max_retries or settings.embedding_max_retries
        self._client = None

    def _get_bedrock_client(self):
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=settings.aws_bedrock_region,
            )
        return self._client

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, handling batching and retries."""
        if not texts:
            return []

        logger.info("embedder.batch_start", count=len(texts), model=self.model_id)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        logger.info("embedder.batch_complete", count=len(all_embeddings))
        return all_embeddings

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with per-text retry."""
        if self.provider == "bedrock":
            return await self._embed_bedrock(texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    async def _embed_bedrock(self, texts: list[str]) -> list[list[float]]:
        """Call Amazon Titan Embed V2 via Bedrock for a batch of texts.

        Each text is embedded individually with per-call retry + jitter
        to handle Bedrock throttling gracefully.
        """
        client = self._get_bedrock_client()
        loop = asyncio.get_event_loop()
        embeddings: list[list[float]] = []

        for text in texts:
            body = json.dumps({
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": True,
            })
            embedding = await self._invoke_with_retry(client, loop, body)
            embeddings.append(embedding)

        return embeddings

    async def _invoke_with_retry(self, client, loop, body: str) -> list[float]:
        """Single invoke_model call with exponential backoff + jitter."""
        for attempt in range(self.max_retries):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda b=body: client.invoke_model(
                        modelId=self.model_id,
                        body=b,
                        contentType="application/json",
                        accept="application/json",
                    ),
                )
                result = json.loads(response["body"].read())
                return result["embedding"]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "embedder.retry",
                    attempt=attempt + 1,
                    wait=round(wait, 1),
                    error=str(e),
                )
                await asyncio.sleep(wait)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        results = await self.embed_batch([text])
        return results[0] if results else []
