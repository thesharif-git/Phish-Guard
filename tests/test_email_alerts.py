import asyncio
from types import SimpleNamespace
import pytest

from utils import email_alerts


def test_phishing_alert_uses_scan_explanation_without_claiming_email_was_blocked(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json, headers):
            captured.update(url=url, payload=json, headers=headers)
            return SimpleNamespace(status_code=201)

    monkeypatch.setattr(email_alerts.settings, "BREVO_API_KEY", "test-key")
    monkeypatch.setattr(email_alerts.settings, "BREVO_FROM_EMAIL", "alerts@example.com")
    monkeypatch.setattr(email_alerts.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(email_alerts.send_phishing_alert(
        "user@example.com", "User", {
            "sender": "fraud@example.net",
            "subject": "<script>alert(1)</script>",
            "final_score": 85,
            "risk_level": "critical",
            "explanation": {"ai_paragraph": "The message asks for a password."},
        }
    ))

    assert result is True
    html = captured["payload"]["htmlContent"]
    assert "The message asks for a password." in html
    assert "&lt;script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Phishing Attack Blocked" not in html
    assert "Action has been taken to isolate" not in html


def test_sync_summary_sends_only_counts(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json, headers):
            captured.update(json)
            return SimpleNamespace(status_code=201)

    monkeypatch.setattr(email_alerts.settings, "BREVO_API_KEY", "test-key")
    monkeypatch.setattr(email_alerts.settings, "BREVO_FROM_EMAIL", "alerts@example.com")
    monkeypatch.setattr(email_alerts.httpx, "AsyncClient", FakeClient)

    assert asyncio.run(email_alerts.send_sync_summary("user@example.com", "User", 3, 1))
    assert "3 new messages scanned" in captured["htmlContent"]
    assert "1 recorded phishing classifications" in captured["htmlContent"]
    assert "not independently confirmed attacks" in captured["htmlContent"]
    assert "flagged as phishing or suspicious" not in captured["htmlContent"]


@pytest.mark.parametrize('record,expected,forbidden', [
    ({'final_score': None, 'risk_score': 99, 'security_action': 'AUTO_QUARANTINE'}, 'Unavailable', '99/100'),
    ({'final_score': 0, 'risk_score': 99, 'security_action': 'AUTO_QUARANTINE'}, '0/100', '99/100'),
    ({'final_score': 100, 'security_action': 'BLOCK_BY_POLICY'}, 'Manual rule match', '100/100'),
])
def test_alert_does_not_fabricate_score_or_treat_administrative_priority_as_model_probability(monkeypatch, record, expected, forbidden):
    captured = {}
    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json, headers):
            captured.update(json)
            return SimpleNamespace(status_code=201)
    monkeypatch.setattr(email_alerts.settings, 'BREVO_API_KEY', 'test-key')
    monkeypatch.setattr(email_alerts.settings, 'BREVO_FROM_EMAIL', 'alerts@example.com')
    monkeypatch.setattr(email_alerts.httpx, 'AsyncClient', FakeClient)
    assert asyncio.run(email_alerts.send_phishing_alert('user@example.com', 'User', record))
    html = captured['htmlContent']
    assert expected in html and forbidden not in html
    assert 'High-risk email detected' not in html
    if record['security_action'] == 'BLOCK_BY_POLICY':
        assert 'not a model phishing classification' in html
        assert 'Manual policy match' in html
