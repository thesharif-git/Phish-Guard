"""Deliver weekly scan summaries for users who enabled them."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, update
from sqlmodel import Session, select

from data.database import engine
from data.models import ScanRecord, User
from utils.email_alerts import send_weekly_digest
from utils.scan_outcomes import count_scan_outcomes


async def send_due_weekly_digests(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    if now.weekday() == 0 and now.hour < 9:
        return 0
    iso_year, iso_week, _ = now.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"
    since = (now - timedelta(days=7)).replace(tzinfo=None)
    delivered = 0

    with Session(engine) as session:
        recipients = session.exec(select(User.id).where(User.weekly_reports == True)).all()

    for user_id in recipients:
        with Session(engine) as session:
            # Claim the week before sending, so two scheduler workers cannot
            # deliver the same user's digest at the same time.
            claim = session.execute(
                update(User).where(
                    User.id == user_id,
                    User.weekly_reports == True,
                    or_(User.weekly_digest_last_sent_week.is_(None),
                        User.weekly_digest_last_sent_week != week_key),
                ).values(weekly_digest_last_sent_week=week_key)
            )
            session.commit()
            if claim.rowcount != 1:
                continue

            user = session.get(User, user_id)
            scans = session.exec(select(ScanRecord).where(
                ScanRecord.user_id == user_id,
                ScanRecord.timestamp >= since,
            )).all()
            outcomes = count_scan_outcomes([
                {**(scan.full_analysis if isinstance(scan.full_analysis, dict) else {}), 'status': scan.status}
                for scan in scans])
            success = await send_weekly_digest(user.email, user.name, len(scans), outcomes['phishing'],
                review_required=outcomes['review_required'], unavailable=outcomes['unavailable'],
                policy_matches=outcomes['policy_matches'], quarantine_recommended=outcomes['quarantine_recommended'])
            if success:
                delivered += 1
            else:
                session.execute(update(User).where(
                    User.id == user_id,
                    User.weekly_digest_last_sent_week == week_key,
                ).values(weekly_digest_last_sent_week=None))
                session.commit()

    return delivered
