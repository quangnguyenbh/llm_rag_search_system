"""Billing service: subscription management, usage tracking."""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.core.billing.stripe_client import StripeClient
from src.core.billing.usage_tracker import UsageTracker
from src.db.postgres.repositories.user import OrganizationRepository, UserRepository

logger = structlog.get_logger()


class BillingService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.user_repo = UserRepository(session)
        self.org_repo = OrganizationRepository(session)
        self.usage = UsageTracker()
        self._stripe: StripeClient | None = None

    def _get_stripe(self) -> StripeClient:
        if self._stripe is None:
            self._stripe = StripeClient(secret_key=settings.stripe_secret_key)
        return self._stripe

    async def get_subscription(self, user_id: str) -> dict:
        """Return the user's current subscription plan."""
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return {"plan": "free", "status": "active"}

        plan = "free"
        if user.organization_id:
            org = await self.org_repo.get_by_id(user.organization_id)
            plan = org.plan if org else "free"

        usage = await self.usage.get_usage(user_id)
        return {
            "plan": plan,
            "status": "active",
            "usage": usage,
        }

    async def create_checkout(self, user_id: str, plan: str) -> str:
        """Create a Stripe checkout session and return the checkout URL."""
        # Validate plan against known values before touching any external service
        _valid_plans = frozenset({"starter", "pro", "enterprise"})
        if plan not in _valid_plans:
            raise ValueError(f"Invalid plan. Choose from: {', '.join(sorted(_valid_plans))}")

        user = await self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        stripe = self._get_stripe()

        # Ensure Stripe customer exists
        customer_id = user.stripe_customer_id
        if not customer_id:
            customer_id = await stripe.create_customer(
                email=user.email, name=user.name
            )
            await self.user_repo.update_stripe_customer(user_id, customer_id)

        # Map plan name to a Stripe price ID (configure in settings/env)
        price_ids = {
            "starter": getattr(settings, "stripe_price_starter", ""),
            "pro": getattr(settings, "stripe_price_pro", ""),
            "enterprise": getattr(settings, "stripe_price_enterprise", ""),
        }
        price_id = price_ids.get(plan, "")
        if not price_id:
            raise ValueError(f"Stripe price ID not configured for plan: {plan}")

        checkout_url = await stripe.create_checkout_session(
            customer_id=customer_id, price_id=price_id
        )
        logger.info("billing.checkout_created", user_id=user_id, plan=plan)
        return checkout_url

    async def handle_webhook(self, event_type: str, data: dict) -> None:
        """Process a Stripe webhook event."""
        logger.info("billing.webhook", event_type=event_type)

        if event_type == "customer.subscription.created":
            await self._handle_subscription_updated(data, status="active")
        elif event_type == "customer.subscription.updated":
            await self._handle_subscription_updated(data, status="active")
        elif event_type == "customer.subscription.deleted":
            await self._handle_subscription_updated(data, status="canceled")

    async def _handle_subscription_updated(self, data: dict, status: str) -> None:
        subscription_id = data.get("id", "")
        plan = data.get("plan", {}).get("nickname", "starter")
        customer_id = data.get("customer", "")

        # Find the user by Stripe customer ID
        # In a full implementation this would query by stripe_customer_id
        logger.info(
            "billing.subscription_updated",
            subscription_id=subscription_id,
            customer_id=customer_id,
            plan=plan,
            status=status,
        )
