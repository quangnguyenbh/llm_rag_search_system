"""Redis cache client with query result and embedding caching."""

from __future__ import annotations

import hashlib
import json

import structlog

from src.config import settings

logger = structlog.get_logger()

_DEFAULT_QUERY_TTL = 3600  # 1 hour
_DEFAULT_EMBED_TTL = 86400  # 24 hours


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _query_cache_key(query_text: str, filters: dict | None = None) -> str:
    """Deterministic cache key for a query + optional filters."""
    filter_str = json.dumps(filters, sort_keys=True) if filters else ""
    return f"query:{_sha256(query_text + filter_str)}"


def _embed_cache_key(text: str) -> str:
    return f"embed:{_sha256(text)}"


def get_redis_client():
    """Return an async Redis client connected to ``settings.redis_url``."""
    import redis.asyncio as redis  # type: ignore[import]

    return redis.from_url(settings.redis_url, decode_responses=True)


class QueryCache:
    """Async Redis-backed cache for query results and embedding vectors.

    Designed for graceful degradation: if Redis is unavailable (e.g. in
    tests), all operations log a warning and return ``None`` / no-op.
    """

    def __init__(
        self,
        query_ttl: int | None = None,
        embed_ttl: int | None = None,
    ) -> None:
        self._query_ttl = query_ttl or settings.query_cache_ttl
        self._embed_ttl = embed_ttl or _DEFAULT_EMBED_TTL
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = get_redis_client()
        return self._client

    # ------------------------------------------------------------------
    # Query result cache
    # ------------------------------------------------------------------

    async def get_query(
        self, query_text: str, filters: dict | None = None
    ) -> dict | None:
        """Return cached query result or ``None`` on miss / error."""
        key = _query_cache_key(query_text, filters)
        try:
            client = self._get_client()
            raw = await client.get(key)
            if raw is None:
                return None
            logger.debug("query_cache.hit", key=key)
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_cache.get_failed", error=str(exc))
            return None

    async def set_query(
        self,
        query_text: str,
        result: dict,
        filters: dict | None = None,
    ) -> None:
        """Cache a query result. Silently swallows errors."""
        key = _query_cache_key(query_text, filters)
        try:
            client = self._get_client()
            await client.setex(key, self._query_ttl, json.dumps(result))
            logger.debug("query_cache.set", key=key, ttl=self._query_ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_cache.set_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    async def get_embedding(self, text: str) -> list[float] | None:
        """Return cached embedding vector or ``None`` on miss / error."""
        key = _embed_cache_key(text)
        try:
            client = self._get_client()
            raw = await client.get(key)
            if raw is None:
                return None
            logger.debug("embed_cache.hit", key=key)
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embed_cache.get_failed", error=str(exc))
            return None

    async def set_embedding(self, text: str, vector: list[float]) -> None:
        """Cache an embedding vector. Silently swallows errors."""
        key = _embed_cache_key(text)
        try:
            client = self._get_client()
            await client.setex(key, self._embed_ttl, json.dumps(vector))
            logger.debug("embed_cache.set", key=key, ttl=self._embed_ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embed_cache.set_failed", error=str(exc))


# Module-level singleton — import and use directly in route handlers
query_cache = QueryCache()
