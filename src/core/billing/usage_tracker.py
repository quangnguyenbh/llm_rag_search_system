"""Usage tracking for metered billing — backed by Redis."""

from datetime import UTC, datetime

from src.db.cache.redis_client import get_redis_client

# TTL for usage keys: 35 days to cover any billing period overlap
_USAGE_KEY_TTL_SECONDS = 60 * 60 * 24 * 35


def _period_key(user_id: str) -> str:
    """Return a Redis key scoped to the current UTC month."""
    period = datetime.now(UTC).strftime("%Y-%m")
    return f"usage:{user_id}:{period}"


class UsageTracker:
    def __init__(self):
        self._redis = None

    def _get_client(self):
        if self._redis is None:
            self._redis = get_redis_client()
        return self._redis

    async def increment(self, user_id: str, query_count: int = 1) -> None:
        """Record query usage for billing."""
        client = self._get_client()
        key = _period_key(user_id)
        await client.incrby(key, query_count)
        await client.expire(key, _USAGE_KEY_TTL_SECONDS)

    async def get_usage(self, user_id: str) -> dict:
        """Get current billing period usage."""
        client = self._get_client()
        key = _period_key(user_id)
        raw = await client.get(key)
        count = int(raw) if raw else 0
        period = datetime.now(UTC).strftime("%Y-%m")
        return {"user_id": user_id, "period": period, "query_count": count}

    async def check_quota(self, user_id: str, limit: int = 100) -> bool:
        """Return True if the user is within their monthly query quota."""
        usage = await self.get_usage(user_id)
        return usage["query_count"] < limit
