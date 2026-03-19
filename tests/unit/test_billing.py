"""Unit tests for UsageTracker."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.billing.usage_tracker import UsageTracker, _period_key


class TestPeriodKey:
    def test_key_format(self):
        key = _period_key("user-123")
        assert key.startswith("usage:user-123:")
        # Should contain year-month part like "2026-03"
        parts = key.split(":")
        assert len(parts) == 3
        assert len(parts[2]) == 7  # "YYYY-MM"


class TestUsageTracker:
    def _make_tracker(self):
        tracker = UsageTracker()
        mock_redis = MagicMock()
        mock_redis.incrby = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=None)
        tracker._redis = mock_redis
        return tracker, mock_redis

    @pytest.mark.asyncio
    async def test_increment_calls_incrby(self):
        tracker, redis = self._make_tracker()
        await tracker.increment("user-1")
        redis.incrby.assert_awaited_once()
        call_args = redis.incrby.call_args
        assert call_args[0][1] == 1  # query_count=1

    @pytest.mark.asyncio
    async def test_increment_custom_count(self):
        tracker, redis = self._make_tracker()
        await tracker.increment("user-1", query_count=5)
        call_args = redis.incrby.call_args
        assert call_args[0][1] == 5

    @pytest.mark.asyncio
    async def test_increment_sets_expiry(self):
        tracker, redis = self._make_tracker()
        await tracker.increment("user-1")
        redis.expire.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_usage_zero_when_no_data(self):
        tracker, redis = self._make_tracker()
        redis.get = AsyncMock(return_value=None)
        usage = await tracker.get_usage("user-1")
        assert usage["query_count"] == 0
        assert usage["user_id"] == "user-1"
        assert "period" in usage

    @pytest.mark.asyncio
    async def test_get_usage_returns_count(self):
        tracker, redis = self._make_tracker()
        redis.get = AsyncMock(return_value="42")
        usage = await tracker.get_usage("user-1")
        assert usage["query_count"] == 42

    @pytest.mark.asyncio
    async def test_check_quota_within_limit(self):
        tracker, redis = self._make_tracker()
        redis.get = AsyncMock(return_value="50")
        result = await tracker.check_quota("user-1", limit=100)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_quota_exceeded(self):
        tracker, redis = self._make_tracker()
        redis.get = AsyncMock(return_value="150")
        result = await tracker.check_quota("user-1", limit=100)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_quota_at_exact_limit_is_exceeded(self):
        tracker, redis = self._make_tracker()
        redis.get = AsyncMock(return_value="100")
        result = await tracker.check_quota("user-1", limit=100)
        assert result is False
