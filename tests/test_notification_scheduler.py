import asyncio
from datetime import datetime, timezone

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from data.models import ScanRecord, User
from services import notification_scheduler


def test_weekly_digest_sends_once_with_account_scoped_counts(monkeypatch):
    test_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)
    now = datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc)
    with Session(test_engine) as session:
        user = User(uuid="digest-user", name="Digest User", email="digest@example.com",
                    hashed_password="unused", weekly_reports=True)
        other = User(uuid="other-user", name="Other", email="other@example.com",
                     hashed_password="unused", weekly_reports=False)
        session.add(user)
        session.add(other)
        session.commit()
        session.refresh(user)
        session.refresh(other)
        session.add(ScanRecord(scan_id="digest-1", user_id=user.id, sender="a@example.com",
                               subject="A", risk_score=80, status="PHISHING",
                               risk_level="critical", timestamp=datetime(2026, 9, 27)))
        session.add(ScanRecord(scan_id="other-1", user_id=other.id, sender="b@example.com",
                               subject="B", risk_score=80, status="PHISHING",
                               risk_level="critical", timestamp=datetime(2026, 9, 27)))
        session.commit()

    sent = []

    async def fake_send(email, name, scans, flagged, **outcomes):
        sent.append((email, scans, flagged, outcomes))
        return True

    monkeypatch.setattr(notification_scheduler, "engine", test_engine)
    monkeypatch.setattr(notification_scheduler, "send_weekly_digest", fake_send)

    assert asyncio.run(notification_scheduler.send_due_weekly_digests(now)) == 1
    assert asyncio.run(notification_scheduler.send_due_weekly_digests(now)) == 0
    assert sent == [("digest@example.com", 1, 1, {'review_required': 0, 'unavailable': 0,
                                                 'policy_matches': 0, 'quarantine_recommended': 0})]


def test_gmail_sync_passes_account_id_to_scan(monkeypatch):
    from datetime import timedelta
    from data.models import InboxConnection
    from services import inbox_connector, scan_engine

    test_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)
    captured = {}

    async def fake_fetch(*args, **kwargs):
        return [{"gmail_id": "gmail-one", "body": "Hello", "sender": "a@example.com",
                 "subject": "Test", "headers_raw": ""}]

    def fake_scan(**kwargs):
        captured.update(kwargs)
        return {"status": "CLEAN"}

    monkeypatch.setattr(inbox_connector, "fetch_gmail_emails", fake_fetch)
    monkeypatch.setattr(inbox_connector, "decrypt_token", lambda value: "token")
    monkeypatch.setattr(scan_engine, "scan_email", fake_scan)

    with Session(test_engine) as session:
        user = User(uuid="sync-user", name="Sync User", email="sync@example.com",
                    hashed_password="unused")
        session.add(user)
        session.commit()
        session.refresh(user)
        connection = InboxConnection(
            user_id=user.id, provider="gmail", email_address=user.email,
            access_token_enc="encrypted", refresh_token_enc="encrypted",
            token_expiry=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1),
        )
        session.add(connection)
        session.commit()
        result = asyncio.run(inbox_connector.sync_gmail_inbox(connection, session))

    assert result["new_scans"] == 1
    assert captured["user_id"] == user.id
