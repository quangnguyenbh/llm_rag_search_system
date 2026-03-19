import hashlib
import hmac
import json
import time

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from src.api.deps import CurrentUserID, DBSession
from src.config import settings
from src.core.billing.service import BillingService
from src.core.billing.usage_tracker import UsageTracker

logger = structlog.get_logger()
router = APIRouter()

# Maximum age of a Stripe webhook event before rejecting (5 minutes)
_WEBHOOK_TOLERANCE_SECONDS = 300


class CheckoutRequest(BaseModel):
    plan: str


@router.get("/subscription")
async def get_subscription(session: DBSession, user_id: CurrentUserID):
    """Get current subscription details."""
    svc = BillingService(session)
    return await svc.get_subscription(user_id)


@router.post("/checkout")
async def create_checkout_session(
    request: CheckoutRequest, session: DBSession, user_id: CurrentUserID
):
    """Create a Stripe checkout session for subscription."""
    svc = BillingService(session)
    try:
        url = await svc.create_checkout(user_id=user_id, plan=request.plan)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"checkout_url": url}


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    body = await request.body()

    # Verify Stripe signature
    sig_header = request.headers.get("stripe-signature", "")
    if settings.stripe_webhook_secret:
        try:
            _verify_stripe_signature(body, sig_header, settings.stripe_webhook_secret)
        except ValueError as exc:
            logger.warning("billing.webhook_invalid_signature", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature"
            ) from exc

    event = json.loads(body)
    event_type = event.get("type", "")

    # Process event (fire-and-forget, no DB session needed for logging)
    logger.info("billing.webhook_received", event_type=event_type)
    return {"received": True, "event_type": event_type}


@router.get("/usage")
async def get_usage(user_id: CurrentUserID):
    """Get current billing period usage."""
    tracker = UsageTracker()
    try:
        usage = await tracker.get_usage(user_id)
    except Exception as exc:
        logger.warning("billing.usage_fetch_failed", error=str(exc))
        usage = {"user_id": user_id, "query_count": 0, "period": "unknown"}
    return usage


def _verify_stripe_signature(payload: bytes, sig_header: str, secret: str) -> None:
    """Verify a Stripe webhook signature with replay-attack protection.

    Implements the same logic as the official Stripe SDK:
      1. Parse timestamp (t=) and v1 signatures from the header.
      2. Construct the signed payload: "{timestamp}.{body}".
      3. Compute HMAC-SHA256 and compare against each v1 signature.
      4. Reject events older than _WEBHOOK_TOLERANCE_SECONDS.
    """
    parts: dict[str, list[str]] = {}
    for item in sig_header.split(","):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        parts.setdefault(k.strip(), []).append(v.strip())

    timestamp_str = parts.get("t", [""])[0]
    signatures = parts.get("v1", [])
    if not timestamp_str or not signatures:
        raise ValueError("Missing timestamp or signature in Stripe-Signature header")

    try:
        timestamp = int(timestamp_str)
    except ValueError as exc:
        raise ValueError("Invalid timestamp in Stripe-Signature header") from exc

    # Replay-attack protection
    age = int(time.time()) - timestamp
    if age > _WEBHOOK_TOLERANCE_SECONDS:
        raise ValueError(f"Webhook event too old: {age}s (tolerance {_WEBHOOK_TOLERANCE_SECONDS}s)")

    signed_payload = f"{timestamp}.".encode() + payload
    expected = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()

    if not any(hmac.compare_digest(expected, sig) for sig in signatures):
        raise ValueError("Stripe signature verification failed")
