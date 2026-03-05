"""Billing service: subscription management, usage tracking."""


class BillingService:
    async def get_subscription(self, user_id: str) -> dict:
        raise NotImplementedError

    async def create_checkout(self, user_id: str, plan: str) -> str:
        raise NotImplementedError

    async def handle_webhook(self, event_type: str, data: dict) -> None:
        raise NotImplementedError
