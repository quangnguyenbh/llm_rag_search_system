"""Usage tracking for metered billing."""


class UsageTracker:
    async def increment(self, user_id: str, query_count: int = 1) -> None:
        """Record query usage for billing."""
        raise NotImplementedError

    async def get_usage(self, user_id: str) -> dict:
        """Get current period usage."""
        raise NotImplementedError

    async def check_quota(self, user_id: str) -> bool:
        """Check if user is within their quota."""
        raise NotImplementedError
