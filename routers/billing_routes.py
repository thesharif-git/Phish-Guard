from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select
import stripe
import logging

from data.database import get_session
from data.models import User
from services.security import get_current_user
from core.config import settings

logger = logging.getLogger("phishguard")
templates = Jinja2Templates(directory="templates")
router = APIRouter(prefix="/billing", tags=["billing"])

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


def _get_user(request: Request, session: Session) -> User:
    try:
        return get_current_user(request, session)
    except HTTPException:
        raise HTTPException(status_code=401, detail="Not authenticated")


@router.post("/create-checkout")
async def create_checkout(request: Request, session: Session = Depends(get_session)):
    """Redirect the user to Stripe Checkout for the selected plan."""
    try:
        user = _get_user(request, session)
    except HTTPException:
        return RedirectResponse(url="/auth/login", status_code=303)

    form = await request.form()
    plan = form.get("plan")
    cycle = form.get("billing_cycle", "monthly")

    if plan not in ["free_plus", "pro"] or cycle not in ["monthly", "yearly"]:
        return RedirectResponse(url="/pricing", status_code=303)

    price_ids = {
        ("free_plus", "monthly"): settings.STRIPE_PRICE_FREE_PLUS,
        ("pro", "monthly"): settings.STRIPE_PRICE_PRO,
        ("free_plus", "yearly"): settings.STRIPE_PRICE_FREE_PLUS_YEARLY,
        ("pro", "yearly"): settings.STRIPE_PRICE_PRO_YEARLY,
    }
    price_id = price_ids[(plan, cycle)]

    if not settings.STRIPE_SECRET_KEY or not price_id:
        logger.error("Stripe credentials or %s price missing. Checkout aborted.", cycle)
        return templates.TemplateResponse("billing/error.html", {
            "request": request,
            "error": f"{cycle.capitalize()} billing is currently unavailable. Please contact support."
        })

    try:
        checkout_session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=str(request.base_url) + "billing/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=str(request.base_url) + "pricing",
            customer_email=user.email,
            metadata={
                "user_id": str(user.id),
                "plan": plan,
                "billing_cycle": cycle,
            },
            subscription_data={
                "metadata": {
                    "user_id": str(user.id),
                    "plan": plan,
                    "billing_cycle": cycle,
                }
            }
        )
        return RedirectResponse(url=checkout_session.url, status_code=303)
    except stripe.StripeError as e:
        logger.error(f"Stripe checkout error: {e}")
        return templates.TemplateResponse("billing/error.html", {
            "request": request,
            "error": "Failed to create checkout session. Please try again."
        })


@router.get("/success", response_class=HTMLResponse)
async def billing_success(request: Request, session_id: str = "", session: Session = Depends(get_session)):
    """Landing page after a successful payment."""
    try:
        user = _get_user(request, session)
    except HTTPException:
        return RedirectResponse(url="/auth/login", status_code=303)

    # Try to retrieve checkout session and update plan immediately
    if session_id and settings.STRIPE_SECRET_KEY:
        try:
            checkout_session = stripe.checkout.Session.retrieve(session_id)
            plan = checkout_session.metadata.get("plan", "free")
            stripe_customer_id = checkout_session.customer
            stripe_subscription_id = checkout_session.subscription

            user.plan = plan
            user.ls_customer_id = stripe_customer_id   # reuse field for stripe customer id
            user.ls_subscription_id = stripe_subscription_id  # reuse field for stripe sub id
            session.add(user)
            session.commit()
            logger.info(f"User {user.id} upgraded to plan: {plan}")
        except Exception as e:
            logger.error(f"Error upgrading plan after checkout: {e}")

    return templates.TemplateResponse("billing/success.html", {
        "request": request,
        "user": user,
    })


@router.post("/webhook")
async def stripe_webhook(request: Request, session: Session = Depends(get_session)):
    """Handle Stripe webhook events to keep subscription state in sync."""
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature", "")

    if settings.STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, settings.STRIPE_WEBHOOK_SECRET)
        except stripe.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
    else:
        # No webhook secret configured (dev mode) — parse event directly
        import json
        try:
            event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")

    event_type = event["type"]

    if event_type in ["customer.subscription.created", "customer.subscription.updated"]:
        sub = event["data"]["object"]
        user_id = sub.get("metadata", {}).get("user_id")
        plan = sub.get("metadata", {}).get("plan", "free")
        stripe_customer = sub.get("customer")
        stripe_sub_id = sub.get("id")
        status = sub.get("status")

        if user_id and status in ["active", "trialing"]:
            user = session.get(User, int(user_id))
            if user:
                user.plan = plan
                user.ls_customer_id = stripe_customer
                user.ls_subscription_id = stripe_sub_id
                session.add(user)
                session.commit()
                logger.info(f"Webhook: User {user_id} plan set to {plan}")

    elif event_type == "customer.subscription.deleted":
        sub = event["data"]["object"]
        user_id = sub.get("metadata", {}).get("user_id")
        if user_id:
            user = session.get(User, int(user_id))
            if user:
                user.plan = "free"
                user.ls_subscription_id = None
                session.add(user)
                session.commit()
                logger.info(f"Webhook: User {user_id} subscription cancelled, reverted to free")

    return {"status": "success"}


@router.get("/portal")
async def customer_portal(request: Request, session: Session = Depends(get_session)):
    """Redirect user to Stripe Customer Portal to manage their subscription."""
    try:
        user = _get_user(request, session)
    except HTTPException:
        return RedirectResponse(url="/auth/login", status_code=303)

    if not user.ls_customer_id or not settings.STRIPE_SECRET_KEY:
        return RedirectResponse(url="/pricing", status_code=303)

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=user.ls_customer_id,
            return_url=str(request.base_url) + "settings",
        )
        return RedirectResponse(url=portal_session.url, status_code=303)
    except stripe.StripeError as e:
        logger.error(f"Stripe portal error: {e}")
        return templates.TemplateResponse("billing/error.html", {
            "request": request,
            "error": "Failed to open customer portal. Please try again."
        })
