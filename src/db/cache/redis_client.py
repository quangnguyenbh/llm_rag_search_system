"""Redis cache client."""

import redis.asyncio as redis

from src.config import settings


def get_redis_client() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)
