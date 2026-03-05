from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/subscription")
async def get_subscription():
    """Get current subscription details."""
    raise NotImplementedError


@router.post("/checkout")
async def create_checkout_session(plan: str):
    """Create a Stripe checkout session for subscription."""
    raise NotImplementedError


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    raise NotImplementedError


@router.get("/usage")
async def get_usage():
    """Get current billing period usage."""
    raise NotImplementedError
