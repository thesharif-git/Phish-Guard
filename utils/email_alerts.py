import httpx
from core.config import settings
from core.logger import logger
from datetime import datetime, timezone
from html import escape
import math


async def _send_summary_email(to_email: str, name: str, subject: str, heading: str, lines: list[str]) -> bool:
    """Send a small, escaped summary through the configured mail provider."""
    if not (settings.BREVO_API_KEY and settings.BREVO_FROM_EMAIL and to_email):
        logger.warning("[Alerts] Summary email skipped: mail delivery is not configured.")
        return False
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    html = (
        '<div style="font-family:Arial,sans-serif;max-width:560px;margin:auto;color:#14213d">'
        f"<h2>{escape(heading)}</h2><p>Hello {escape(name or 'there')},</p>"
        f"<ul>{items}</ul><p>Open PhishGuard to review your scan history.</p></div>"
    )
    payload = {
        "sender": {"name": settings.BREVO_FROM_NAME, "email": settings.BREVO_FROM_EMAIL},
        "to": [{"email": to_email, "name": name or to_email}],
        "subject": subject,
        "htmlContent": html,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "https://api.brevo.com/v3/smtp/email", json=payload,
                headers={"api-key": settings.BREVO_API_KEY, "Content-Type": "application/json"},
            )
        if response.status_code in (200, 201, 202):
            return True
        logger.error("[Alerts] Summary delivery failed with status %s", response.status_code)
    except Exception as exc:
        logger.error("[Alerts] Summary delivery failed: %s", exc)
    return False


def _outcome_summary_lines(phishing, review_required, unavailable, policy_matches, quarantine_recommended):
    return [f"{phishing} recorded phishing classifications; these are not independently confirmed attacks.",
            f"{review_required} scans require review, separately from phishing classifications.",
            f"{unavailable} scans could not be classified reliably.",
            f"{policy_matches} manual policy matches, separately from model classification.",
            f"{quarantine_recommended} quarantine recommendations; recommendations are not executed message moves."]


async def send_sync_summary(to_email: str, name: str, new_scans: int, phishing: int, *,
                            review_required=0, unavailable=0, policy_matches=0, quarantine_recommended=0) -> bool:
    return await _send_summary_email(
        to_email, name, "PhishGuard Gmail sync summary", "New Gmail scans",
        [f"{new_scans} new message{'s' if new_scans != 1 else ''} scanned."] +
        _outcome_summary_lines(phishing, review_required, unavailable, policy_matches, quarantine_recommended),
    )


async def send_weekly_digest(to_email: str, name: str, scans: int, phishing: int, *,
                             review_required=0, unavailable=0, policy_matches=0, quarantine_recommended=0) -> bool:
    return await _send_summary_email(
        to_email, name, "Your PhishGuard weekly scan summary", "Your past seven days",
        [f"{scans} scan{'s' if scans != 1 else ''} saved."] +
        _outcome_summary_lines(phishing, review_required, unavailable, policy_matches, quarantine_recommended),
    )

async def send_phishing_alert(to_email: str, recipient_name: str, scan_record: dict) -> bool:
    """
    Sends a beautifully formatted phishing alert email to the user via Brevo API.
    Brevo free plan: 300 emails/day (3x more than Resend's 100/day).
    Returns True if successfully sent, False otherwise.
    """
    api_key = settings.BREVO_API_KEY
    from_email = settings.BREVO_FROM_EMAIL
    from_name = settings.BREVO_FROM_NAME

    if not api_key:
        logger.warning(
            f"[Alerts] BREVO_API_KEY is not configured in .env. "
            f"Skipping dispatch of phishing alert email to '{to_email}'."
        )
        return False

    if not from_email:
        logger.warning(
            f"[Alerts] BREVO_FROM_EMAIL is not configured in .env. "
            f"Skipping dispatch of phishing alert email to '{to_email}'."
        )
        return False

    if not to_email:
        logger.error("[Alerts] Recipient email is empty. Skipping email alert dispatch.")
        return False

    scan_id = scan_record.get("scan_id") or scan_record.get("id") or "recent"
    sender = str(scan_record.get("sender") or "Unknown Sender")
    subject = str(scan_record.get("subject") or "No Subject")
    risk_score = scan_record.get("final_score") if 'final_score' in scan_record else scan_record.get("risk_score")
    action = scan_record.get('security_action')
    if action == 'BLOCK_BY_POLICY' or scan_record.get('decision_basis') == 'manual_sender_block':
        outcome_heading = 'Manual policy match'
        outcome_description = 'The sender matched a configured administrative rule. This is not a model phishing classification or proof of an attack.'
        score_label, score_text = 'Policy result', 'Manual rule match'
    else:
        outcome_heading = 'Quarantine recommended' if action == 'AUTO_QUARANTINE' else 'Scanner phishing classification'
        outcome_description = 'Review the scan evidence and verify the sender through a trusted channel. A classification is not independent confirmation of an attack or an executed message block.'
        score_label = 'Recorded model output'
        score_text = f'{risk_score}/100' if (isinstance(risk_score, (int, float)) and not isinstance(risk_score, bool)
            and math.isfinite(risk_score) and 0 <= risk_score <= 100) else 'Unavailable'
    explanation_data = scan_record.get("explanation") or {}
    if isinstance(explanation_data, dict):
        explanation = explanation_data.get("ai_paragraph") or explanation_data.get("summary") or "Review the available evidence; no detailed explanation was supplied."
    else:
        explanation = str(explanation_data) or "Review the available evidence; no detailed explanation was supplied."
    safe_sender = escape(sender)
    safe_subject = escape(subject)
    safe_explanation = escape(str(explanation))
    detected_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PhishGuard Threat Alert</title>
        <style>
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: #f8fafc;
                color: #0f172a;
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }}
            .email-wrapper {{
                width: 100%;
                background-color: #f8fafc;
                padding: 40px 0;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #0f172a, #020617);
                padding: 30px 40px;
                text-align: center;
            }}
            .header h1 {{
                color: #ffffff;
                font-size: 20px;
                font-weight: 800;
                margin: 0;
                letter-spacing: -0.02em;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }}
            .content {{
                padding: 40px;
            }}
            .alert-banner {{
                background-color: #fef2f2;
                border: 1px solid #fee2e2;
                border-left: 4px solid #ef4444;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
            }}
            .alert-title {{
                color: #991b1b;
                font-size: 15px;
                font-weight: 700;
                margin: 0 0 6px 0;
            }}
            .alert-desc {{
                color: #7f1d1d;
                font-size: 13px;
                margin: 0;
            }}
            .section-title {{
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #94a3b8;
                margin: 0 0 15px 0;
                border-bottom: 1px solid #f1f5f9;
                padding-bottom: 6px;
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }}
            .metrics-table td {{
                padding: 12px 0;
                border-bottom: 1px solid #f1f5f9;
                font-size: 13.5px;
            }}
            .metrics-table td.label {{
                color: #64748b;
                font-weight: 600;
                width: 120px;
            }}
            .metrics-table td.value {{
                color: #0f172a;
                font-weight: 500;
            }}
            .score-badge {{
                display: inline-block;
                background-color: #fef2f2;
                color: #ef4444;
                border: 1px solid #fee2e2;
                padding: 2px 8px;
                border-radius: 999px;
                font-size: 11.5px;
                font-weight: 700;
            }}
            .explanation-block {{
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 18px;
                font-size: 13px;
                color: #475569;
                font-family: inherit;
                line-height: 1.6;
                margin-bottom: 30px;
                white-space: pre-line;
            }}
            .btn-container {{
                text-align: center;
                margin-bottom: 10px;
            }}
            .btn {{
                display: inline-block;
                background: linear-gradient(135deg, #2563eb, #1e40af);
                color: #ffffff !important;
                text-decoration: none;
                font-size: 13.5px;
                font-weight: 600;
                padding: 12px 28px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
            }}
            .footer {{
                text-align: center;
                padding: 30px 40px;
                font-size: 11px;
                color: #94a3b8;
                border-top: 1px solid #f1f5f9;
            }}
            .footer a {{
                color: #2563eb;
                text-decoration: none;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <div class="email-wrapper">
            <div class="container">
                <!-- Header -->
                <div class="header">
                    <h1>🛡️ PhishGuard Security Alert</h1>
                </div>
                
                <!-- Content -->
                <div class="content">
                    <div class="alert-banner">
                        <p class="alert-title">{escape(outcome_heading)}</p>
                        <p class="alert-desc">{escape(outcome_description)}</p>
                    </div>

                    <p class="section-title">Incident Details</p>
                    <table class="metrics-table">
                        <tr>
                            <td class="label">Sender (From)</td>
                            <td class="value" style="word-break: break-all;"><code>{safe_sender}</code></td>
                        </tr>
                        <tr>
                            <td class="label">Subject</td>
                            <td class="value"><strong>{safe_subject}</strong></td>
                        </tr>
                        <tr>
                            <td class="label">{escape(score_label)}</td>
                            <td class="value">
                                <span class="score-badge">{escape(score_text)}</span>
                            </td>
                        </tr>
                        <tr>
                            <td class="label">Detected At</td>
                            <td class="value">{detected_at}</td>
                        </tr>
                    </table>

                    <p class="section-title">Scan Analysis &amp; Verdict</p>
                    <div class="explanation-block">
                        {safe_explanation}
                    </div>

                </div>

                <!-- Footer -->
                <div class="footer">
                    <p>This is an automated scan-result notification from PhishGuard.</p>
                    <p>You can change alert preferences in PhishGuard Settings.</p>
                    <p style="margin-top: 15px;">&copy; 2026 PhishGuard AI Security. All rights reserved.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # Brevo API v3 payload format
    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": to_email, "name": recipient_name or to_email}],
        "subject": f"PhishGuard alert: {outcome_heading.lower()} from {sender.split('@')[-1].replace('>', '')}",
        "htmlContent": html_content
    }

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        logger.info(f"[Alerts] Dispatching Brevo phishing email alert to '{to_email}' for scan '{scan_id}'...")
        
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post("https://api.brevo.com/v3/smtp/email", json=payload, headers=headers)
            
            if resp.status_code in (200, 201, 202):
                logger.info(f"[Alerts] Phishing email alert successfully dispatched to '{to_email}' via Brevo (Status: {resp.status_code}).")
                return True
            else:
                logger.error(
                    f"[Alerts] Brevo API returned non-success code: {resp.status_code}. "
                    f"Response: {resp.text}"
                )
                return False
    except Exception as e:
        logger.error(f"[Alerts] Exception occurred while dispatching email alert to '{to_email}' via Brevo: {e}")
        return False
