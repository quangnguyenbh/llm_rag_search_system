"""Stripe API client wrapper."""


class StripeClient:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    async def create_customer(self, email: str, name: str) -> str:
        raise NotImplementedError

    async def create_checkout_session(self, customer_id: str, price_id: str) -> str:
        raise NotImplementedError

    async def cancel_subscription(self, subscription_id: str) -> None:
        raise NotImplementedError
