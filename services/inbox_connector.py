"""
PhishGuard Inbox Connector Service
Handles Gmail (and future Outlook) OAuth token management + email fetching.
Tokens are encrypted at rest using Fernet symmetric encryption.
"""
import base64
from email.message import Message
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import httpx
from cryptography.fernet import Fernet, InvalidToken

from core.config import settings
from core.logger import logger
from utils.email_parser import parse_eml_content, decode_email_text, visible_html_text


# ── Token Encryption ──────────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    """Return Fernet instance from config key. Raises if key not set."""
    key = settings.INBOX_ENCRYPTION_KEY
    if not key:
        raise ValueError("INBOX_ENCRYPTION_KEY not configured in .env")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_token(token: str) -> str:
    """Encrypt a token string for database storage."""
    f = _get_fernet()
    return f.encrypt(token.encode()).decode()


def decrypt_token(encrypted: str) -> str:
    """Decrypt a stored token. Returns empty string on failure."""
    try:
        f = _get_fernet()
        return f.decrypt(encrypted.encode()).decode()
    except (InvalidToken, Exception) as e:
        logger.error(f"Token decryption failed: {e}")
        return ""


# ── Gmail API Constants ───────────────────────────────────────────────────────

GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_MESSAGES_URL = "https://www.googleapis.com/gmail/v1/users/me/messages"
GMAIL_MESSAGE_URL = "https://www.googleapis.com/gmail/v1/users/me/messages/{msg_id}"
GMAIL_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Scopes needed for inbox reading (separate from login)
GMAIL_INBOX_SCOPES = " ".join([
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
])


# ── Gmail OAuth URL Builder ───────────────────────────────────────────────────

def build_gmail_connect_url(state: str = "") -> str:
    """Build the Google OAuth authorization URL for inbox access."""
    params = (
        f"?client_id={settings.GOOGLE_CLIENT_ID}"
        f"&redirect_uri={settings.GOOGLE_INBOX_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={GMAIL_INBOX_SCOPES.replace(' ', '%20')}"
        f"&access_type=offline"           # needed to get refresh_token
        f"&prompt=consent"                # force consent screen to always get refresh_token
        f"&include_granted_scopes=false"
    )
    if state:
        params += f"&state={state}"
    return "https://accounts.google.com/o/oauth2/v2/auth" + params


# ── Gmail Token Exchange ──────────────────────────────────────────────────────

async def exchange_gmail_code(code: str) -> dict:
    """
    Exchange an authorization code for access + refresh tokens.
    Returns dict with: access_token, refresh_token, expires_in, email
    """
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GMAIL_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_INBOX_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )

    if token_resp.status_code != 200:
        raise ValueError(f"Token exchange failed: {token_resp.text}")

    token_data = token_resp.json()

    # Fetch user email so we know which mailbox this is
    async with httpx.AsyncClient() as client:
        userinfo_resp = await client.get(
            GMAIL_USERINFO_URL,
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )

    email = ""
    if userinfo_resp.status_code == 200:
        email = userinfo_resp.json().get("email", "")

    return {
        "access_token": token_data.get("access_token", ""),
        "refresh_token": token_data.get("refresh_token", ""),
        "expires_in": token_data.get("expires_in", 3600),
        "email": email,
    }


# ── Gmail Token Refresh ───────────────────────────────────────────────────────

async def refresh_gmail_access_token(refresh_token: str) -> dict:
    """
    Use a refresh token to get a new short-lived access token.
    Returns dict with: access_token, expires_in
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GMAIL_TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "grant_type": "refresh_token",
            },
        )

    if resp.status_code != 200:
        raise ValueError(f"Token refresh failed: {resp.text}")

    data = resp.json()
    return {
        "access_token": data.get("access_token", ""),
        "expires_in": data.get("expires_in", 3600),
    }


# ── Gmail Email Fetcher ───────────────────────────────────────────────────────

def _decode_base64_url(data: str) -> str:
    """Decode base64url-encoded string (Gmail uses this for message bodies)."""
    try:
        return decode_email_text(_decode_gmail_bytes(data))[0]
    except Exception:
        return ""


def _decode_gmail_bytes(data: str) -> bytes:
    return base64.b64decode(data + "=" * (-len(data) % 4), altchars=b"-_", validate=True)


def _gmail_inline_text_parts(payload):
    disposition = _extract_header(payload.get("headers", []), "Content-Disposition")
    if "attachment" in disposition.lower() or payload.get("filename") or payload.get("mimeType") == "message/rfc822":
        return
    if payload.get("mimeType") in ("text/plain", "text/html"):
        yield payload
    else:
        for child in payload.get("parts", []):
            yield from _gmail_inline_text_parts(child)


def _decode_gmail_part(part):
    header = Message()
    header['Content-Type'] = _extract_header(part.get('headers', []), 'Content-Type') or part.get('mimeType', 'text/plain')
    return decode_email_text(_decode_gmail_bytes(part.get('body', {}).get('data', '')), header.get_content_charset())[0]


def _extract_body_from_payload(payload: dict) -> str:
    """Legacy full-payload fallback; raw Gmail ingestion uses the RFC parser."""
    parts = list(_gmail_inline_text_parts(payload))
    plain = [_decode_gmail_part(p).strip() for p in parts if p.get('mimeType') == 'text/plain']
    body = next((p for p in plain if p), '')
    if body:
        return body
    markup = [_decode_gmail_part(p) for p in parts if p.get('mimeType') == 'text/html']
    return next((text for p in markup if (text := visible_html_text(p))), '')


def _extract_header(headers: list, name: str) -> str:
    """Extract a specific header value from the Gmail headers list."""
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _extract_html_from_payload(payload: dict) -> str:
    """Keep HTML alternatives for link evidence, independent of text selection."""
    return '\n'.join(_decode_gmail_part(p) for p in _gmail_inline_text_parts(payload) if p.get('mimeType') == 'text/html')


async def fetch_gmail_emails(access_token: str, max_results: int = 20) -> List[dict]:
    """
    Fetch the latest emails from Gmail inbox.
    Returns a list of dicts with: subject, sender, body, headers_raw, date
    """
    emails = []
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=20) as client:
        # Step 1: List message IDs from INBOX
        list_resp = await client.get(
            GMAIL_MESSAGES_URL,
            headers=headers,
            params={"maxResults": max_results, "labelIds": "INBOX", "q": "is:unread OR newer_than:3d"},
        )

        if list_resp.status_code != 200:
            logger.error(f"Gmail list failed: {list_resp.status_code} {list_resp.text[:200]}")
            return []

        messages = list_resp.json().get("messages", [])
        logger.info(f"Gmail: found {len(messages)} messages to fetch")

        # Step 2: Fetch each message in full
        for msg in messages[:max_results]:
            msg_id = msg.get("id")
            try:
                msg_resp = await client.get(
                    GMAIL_MESSAGE_URL.format(msg_id=msg_id),
                    headers=headers,
                    params={"format": "raw"},
                )
                if msg_resp.status_code != 200:
                    continue

                data = msg_resp.json()
                payload = data.get("payload", {})
                msg_headers = payload.get("headers", [])

                date_str = _extract_header(msg_headers, "Date") or ""
                if data.get('raw'):
                    parsed = parse_eml_content(_decode_gmail_bytes(data['raw']))
                    parsed['parsing']['source_kind'] = 'gmail_raw_rfc_email'
                else:
                    # Explicitly partial fallback for older payload providers.
                    # It cannot hide detached HTML/image parts or claim completeness.
                    parsed = {'body': _extract_body_from_payload(payload),
                              'html_body': _extract_html_from_payload(payload), 'attachments': [],
                              'subject': _extract_header(msg_headers, 'Subject'),
                              'sender': _extract_header(msg_headers, 'From'),
                              'headers': '\n'.join(f"{h['name']}: {h['value']}" for h in msg_headers),
                              'parsing': {'source_kind': 'gmail_full_payload_fallback', 'text_complete': False,
                                          'errors': ['Original RFC message unavailable'], 'unscanned_parts': []}}

                emails.append({
                    "gmail_id": msg_id,
                    "subject": parsed['subject'],
                    "sender": parsed['sender'],
                    "body": parsed['body'],
                    "html_body": parsed['html_body'],
                    "attachments": parsed['attachments'],
                    "parsing": parsed['parsing'],
                    "headers_raw": parsed['headers'],
                    "date": date_str,
                })
            except Exception as e:
                logger.warning(f"Failed to fetch Gmail message {msg_id}: {e}")
                continue

    return emails


# ── Sync Inbox (fetch + scan + store) ────────────────────────────────────────

async def sync_gmail_inbox(connection, session) -> dict:
    """
    Full sync cycle for a Gmail inbox connection:
    1. Refresh access token if needed
    2. Fetch latest emails
    3. Run PhishGuard scan on each
    4. Store results in ScanRecord
    Returns summary dict with counts.
    """
    from data.models import InboxConnection, User
    from services.scan_engine import scan_email as run_scan
    from sqlmodel import select
    from utils.scan_outcomes import count_scan_outcomes, scan_outcome

    user = session.get(User, connection.user_id)

    # 1. Refresh token if expired or expiring soon
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    needs_refresh = (
        not connection.token_expiry or
        connection.token_expiry <= now + timedelta(minutes=5)
    )

    access_token = decrypt_token(connection.access_token_enc)
    refresh_token = decrypt_token(connection.refresh_token_enc)

    if needs_refresh and refresh_token:
        try:
            new_tokens = await refresh_gmail_access_token(refresh_token)
            access_token = new_tokens["access_token"]
            connection.access_token_enc = encrypt_token(access_token)
            connection.token_expiry = now + timedelta(seconds=new_tokens["expires_in"])
            session.add(connection)
            session.commit()
            logger.info(f"Gmail token refreshed for user_id={connection.user_id}")
        except Exception as e:
            logger.error(f"Token refresh failed for user_id={connection.user_id}: {e}")
            return {"success": False, "error": "Token refresh failed. Please reconnect your inbox."}

    # 2. Fetch emails
    try:
        emails = await fetch_gmail_emails(access_token, max_results=15)
    except Exception as e:
        logger.error(f"Gmail fetch failed: {e}")
        return {"success": False, "error": "Failed to fetch emails from Gmail."}

    if not emails:
        connection.last_synced = now
        session.add(connection)
        session.commit()
        return {"success": True, "fetched": 0, "new_scans": 0, "phishing_found": 0,
                "review_required": 0, "results_unavailable": 0, "policy_matches": 0,
                "quarantine_recommended": 0}

    # 3. Scan each email (skip duplicates by checking gmail_id in existing scan notes)
    new_scans = 0
    outcomes = count_scan_outcomes([])

    for email in emails:
        # Build a unique scan_id using the Gmail message ID
        scan_id_check = f"gm-{email['gmail_id']}"
        from data.models import ScanRecord
        existing = session.exec(
            select(ScanRecord).where(ScanRecord.scan_id == scan_id_check)
        ).first()
        if existing:
            continue  # Already scanned

        try:
            result = run_scan(
                content=email["body"],
                sender=email["sender"],
                subject=email["subject"],
                headers=email["headers_raw"],
                html_body=email.get("html_body", ""),
                attachments=email.get("attachments", []),
                parsing_info=email.get("parsing"),
                session=session,
                gmail_id=email["gmail_id"],  # passed to store for custom scan_id prefix
                user_id=connection.user_id,
            )
            session.commit()  # Commit each email to avoid bulk rollback on IntegrityError
            
            new_scans += 1
            outcome = scan_outcome(result)
            outcomes[outcome] += 1
            if outcome == 'phishing':
                if result.get('security_action') == 'AUTO_QUARANTINE':
                    outcomes['quarantine_recommended'] += 1
                if user and user.browser_alerts:
                    from utils.email_alerts import send_phishing_alert
                    await send_phishing_alert(user.email, user.name, result)
        except Exception as e:
            session.rollback() # Clear the failed transaction so the loop can continue
            logger.warning(f"Scan failed for Gmail msg {email['gmail_id']}: {e}")
            continue

    # 4. Update last_synced
    connection.last_synced = now
    session.add(connection)
    session.commit()

    if user and user.sync_alerts and new_scans:
        from utils.email_alerts import send_sync_summary
        await send_sync_summary(user.email, user.name, new_scans, outcomes['phishing'],
            review_required=outcomes['review_required'], unavailable=outcomes['unavailable'],
            policy_matches=outcomes['policy_matches'], quarantine_recommended=outcomes['quarantine_recommended'])

    logger.info("Gmail sync complete: %s new scans; %s model phishing classifications, %s review, %s unavailable, %s policy matches",
                new_scans, outcomes['phishing'], outcomes['review_required'], outcomes['unavailable'], outcomes['policy_matches'])
    return {
        "success": True,
        "fetched": len(emails),
        "new_scans": new_scans,
        "phishing_found": outcomes['phishing'],
        "review_required": outcomes['review_required'],
        "results_unavailable": outcomes['unavailable'],
        "policy_matches": outcomes['policy_matches'],
        "quarantine_recommended": outcomes['quarantine_recommended'],
    }
