"""Decision reporting must retain review, missing evidence and account scope."""
import asyncio
from datetime import datetime, timezone

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from data.models import ScanRecord, User
from routers import api, pages


def test_settings_activity_preserves_review_unavailable_and_policy(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    account = User(id=7, uuid="activity-owner", name="Owner", email="owner@example.test",
                   hashed_password="test-only")
    monkeypatch.setattr(pages, "get_current_user", lambda *_: account)
    monkeypatch.setattr(pages.store, "get_kpi", lambda *_: {})
    monkeypatch.setattr(pages.store, "get_dashboard_metrics", lambda *_: {})
    monkeypatch.setattr(pages.store, "get_block_rules", lambda *_: [])
    captured = {}
    def render(_, context):
        captured.update(context)
        return context
    monkeypatch.setattr(pages.templates, "TemplateResponse", render)
    now = datetime.now(timezone.utc)
    decisions = [("PHISHING", "AUTO_QUARANTINE"), ("SUSPICIOUS", "WARNING_BANNER"),
                 ("UNKNOWN", "MANUAL_REVIEW"), ("PHISHING", "BLOCK_BY_POLICY"),
                 ("CLEAN", "SAFE_DELIVERY"), ("PHISHING", "WARNING_BANNER")]
    with Session(engine) as session:
        for index, (status, action) in enumerate(decisions):
            scan = ScanRecord(scan_id=f"mine-{index}", user_id=7, scan_type="email",
                              sender="observed@example.test", subject="Observed message",
                              status=status, risk_level="unknown", timestamp=now)
            scan.full_analysis = {"status": status, "security_action": action}
            session.add(scan)
        session.add(ScanRecord(scan_id="another-account", user_id=8, scan_type="email",
                               sender="private@example.test", subject="Private",
                               status="PHISHING", risk_level="critical", timestamp=now))
        session.commit()
        asyncio.run(pages.settings(object(), session))
    activity = captured["activity"]
    assert activity["week_total"] == 6
    assert activity["week_flagged"] == 5
    assert activity["week_phishing"] == 1
    assert activity["week_review"] == 2
    assert activity["week_unavailable"] == 1
    assert activity["week_policy"] == 1
    assert sum(day["count"] for day in activity["days"]) == 6


def test_dashboard_classification_cues_are_requested_for_current_account(monkeypatch):
    account = User(id=41, uuid="api-owner", name="Owner", email="api@example.test",
                   hashed_password="test-only")
    monkeypatch.setattr(api.store, "get_kpi", lambda *_: {"phishing_detected": 0, "emails_scanned_today": 0})
    monkeypatch.setattr(api.store, "get_risk_trend", lambda *_: {})
    monkeypatch.setattr(api.store, "get_top_targets", lambda *_: {})
    monkeypatch.setattr(api.store, "get_recent_scans", lambda *_, **__: [])
    captured = []
    def types(session, user_id):
        captured.append(user_id)
        return {"labels": [], "values": []}
    monkeypatch.setattr(api.store, "get_phishing_types", types)
    result = asyncio.run(api.dashboard_data(object(), account))
    assert captured == [41]
    assert result["phishing_types"] == {"labels": [], "values": []}
