"""Scan allowances for signed-in accounts."""

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlmodel import Session, select, func

from data.models import ScanRecord, User

FREE_MONTHLY_SCANS = 25


def scans_this_month(session: Session, user_id: int) -> int:
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).replace(tzinfo=None)
    return session.exec(
        select(func.count(ScanRecord.id)).where(
            ScanRecord.user_id == user_id,
            ScanRecord.timestamp >= month_start,
        )
    ).one()


def require_scan_allowance(session: Session, user: User | None) -> None:
    if user and user.plan == "free" and scans_this_month(session, user.id) >= FREE_MONTHLY_SCANS:
        raise HTTPException(
            status_code=429,
            detail="You've used your 25 free scans this month. Upgrade or try again next month.",
        )
