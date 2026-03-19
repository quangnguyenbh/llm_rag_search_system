"""Stripe API client wrapper."""

import structlog

try:
    import stripe as _stripe_lib  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _stripe_lib = None  # type: ignore[assignment]

logger = structlog.get_logger()

_SUPPORTED_PLANS = frozenset({"starter", "pro", "enterprise"})


class StripeClient:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def _get_stripe(self):
        if _stripe_lib is None:  # pragma: no cover
            raise ImportError("stripe package is not installed")
        _stripe_lib.api_key = self.secret_key
        return _stripe_lib

    async def create_customer(self, email: str, name: str) -> str:
        """Create a Stripe customer and return the customer ID."""
        stripe = self._get_stripe()
        customer = stripe.Customer.create(email=email, name=name)
        logger.info("stripe.customer_created", customer_id=customer["id"])
        return customer["id"]

    async def create_checkout_session(self, customer_id: str, price_id: str) -> str:
        """Create a Stripe checkout session and return the session URL."""
        stripe = self._get_stripe()
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url="https://app.manualai.com/billing/success",
            cancel_url="https://app.manualai.com/billing/cancel",
        )
        return session["url"]

    async def cancel_subscription(self, subscription_id: str) -> None:
        """Cancel a Stripe subscription immediately."""
        stripe = self._get_stripe()
        stripe.Subscription.cancel(subscription_id)
        logger.info("stripe.subscription_canceled", subscription_id=subscription_id)
