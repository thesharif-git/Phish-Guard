"""Warnings, missing inference and administrative rules are distinct outcomes."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from data import store
from data.models import InboxConnection, ScanRecord, User
from services import inbox_connector, notification_scheduler, scan_engine
from utils import email_alerts
from utils.scan_outcomes import count_scan_outcomes, scan_outcome


CASES = [
    {'status': 'PHISHING', 'security_action': 'AUTO_QUARANTINE'},
    {'status': 'SUSPICIOUS', 'security_action': 'WARNING_BANNER'},
    {'status': 'UNKNOWN', 'security_action': 'WARNING_BANNER'},
    {'status': 'PHISHING', 'security_action': 'BLOCK_BY_POLICY'},
    {'status': 'CLEAN', 'security_action': 'SAFE_DELIVERY'},
]


@pytest.mark.parametrize('result,outcome', [
    (CASES[0], 'phishing'), (CASES[1], 'review_required'),
    (CASES[2], 'unavailable'), (CASES[3], 'policy_matches'), (CASES[4], 'low_risk'),
    ({'status': 'PHISHING', 'security_action': 'WARNING_BANNER'}, 'review_required'),
    ({'status': 'PHISHING', 'decision_basis': 'manual_sender_block'}, 'policy_matches'),
    ({'status': 'UNKNOWN', 'final_score': 99}, 'unavailable'),
    ({'status': 'CLEAN', 'final_score': 99}, 'low_risk'),
    ({}, 'unavailable'), (None, 'unavailable'),
])
def test_outcomes_follow_status_and_action_not_score_or_missing_metadata(result, outcome):
    assert scan_outcome(result) == outcome


def test_outcomes_are_partitioned_without_counting_reviews_as_attacks():
    counts = count_scan_outcomes(CASES)
    assert counts == {'phishing': 1, 'review_required': 1, 'unavailable': 1,
                      'policy_matches': 1, 'low_risk': 1, 'quarantine_recommended': 1}
    assert sum(counts[key] for key in ('phishing', 'review_required', 'unavailable', 'policy_matches', 'low_risk')) == len(CASES)


def in_memory_engine():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    return engine


def test_gmail_sync_counts_and_summary_preserve_review_unknown_and_policy_without_real_sends(monkeypatch):
    test_engine = in_memory_engine()
    sends, summaries = [], []
    async def fake_fetch(*args, **kwargs):
        return [{'gmail_id': str(index), 'body': 'Original body', 'sender': 'sender@example.com',
                 'subject': 'Original subject', 'headers_raw': ''} for index in range(len(CASES))]
    def fake_scan(**kwargs):
        result = {**CASES[int(kwargs['gmail_id'])], 'final_score': 99, 'sender': kwargs['sender'],
                  'subject': kwargs['subject']}
        record = ScanRecord(scan_id='gm-' + kwargs['gmail_id'], user_id=kwargs['user_id'],
            sender=kwargs['sender'], subject=kwargs['subject'], status=result['status'], risk_level='unknown')
        record.full_analysis = result
        kwargs['session'].add(record)
        return result
    async def fake_alert(email, name, result):
        sends.append(result)
        return True
    async def fake_summary(email, name, new_scans, phishing, **outcomes):
        summaries.append((new_scans, phishing, outcomes))
        return True
    monkeypatch.setattr(inbox_connector, 'fetch_gmail_emails', fake_fetch)
    monkeypatch.setattr(inbox_connector, 'decrypt_token', lambda value: 'mock-token')
    monkeypatch.setattr(scan_engine, 'scan_email', fake_scan)
    monkeypatch.setattr(email_alerts, 'send_phishing_alert', fake_alert)
    monkeypatch.setattr(email_alerts, 'send_sync_summary', fake_summary)
    with Session(test_engine) as session:
        user = User(uuid='outcome-sync', name='User', email='user@example.com', hashed_password='unused',
                    sync_alerts=True, browser_alerts=True)
        session.add(user)
        session.commit()
        session.refresh(user)
        connection = InboxConnection(user_id=user.id, provider='gmail', email_address=user.email,
            access_token_enc='mock', refresh_token_enc='mock',
            token_expiry=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1))
        session.add(connection)
        session.commit()
        result = asyncio.run(inbox_connector.sync_gmail_inbox(connection, session))
        assert result['new_scans'] == 5
        assert result['phishing_found'] == 1
        assert result['review_required'] == result['results_unavailable'] == result['policy_matches'] == 1
        assert result['quarantine_recommended'] == 1
        again = asyncio.run(inbox_connector.sync_gmail_inbox(connection, session))
        assert again['new_scans'] == again['phishing_found'] == 0
    assert len(sends) == 1 and sends[0]['security_action'] == 'AUTO_QUARANTINE'
    assert summaries == [(5, 1, {'review_required': 1, 'unavailable': 1,
                                 'policy_matches': 1, 'quarantine_recommended': 1})]


def test_weekly_digest_uses_saved_outcomes_keeps_account_scope_and_sends_once(monkeypatch):
    test_engine = in_memory_engine()
    now = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)
    with Session(test_engine) as session:
        user = User(uuid='outcome-digest', name='User', email='user@example.com', hashed_password='unused', weekly_reports=True)
        other = User(uuid='outcome-other', name='Other', email='other@example.com', hashed_password='unused')
        session.add(user)
        session.add(other)
        session.commit()
        session.refresh(user)
        session.refresh(other)
        for index, result in enumerate(CASES):
            record = ScanRecord(scan_id=f'outcome-{index}', user_id=user.id, sender='sender@example.com',
                subject='Message', risk_score=99, status=result['status'], risk_level='critical',
                timestamp=datetime(2026, 10, 4))
            record.full_analysis = result
            session.add(record)
        session.add(ScanRecord(scan_id='unrelated-account', user_id=other.id, sender='other@example.com',
            subject='Other', status='PHISHING', risk_level='critical', timestamp=datetime(2026, 10, 4)))
        session.add(ScanRecord(scan_id='too-old', user_id=user.id, sender='sender@example.com',
            subject='Old', status='PHISHING', risk_level='critical', timestamp=datetime(2026, 9, 20)))
        session.commit()
    sends = []
    async def fake_digest(email, name, scans, phishing, **outcomes):
        sends.append((email, scans, phishing, outcomes))
        return True
    monkeypatch.setattr(notification_scheduler, 'engine', test_engine)
    monkeypatch.setattr(notification_scheduler, 'send_weekly_digest', fake_digest)
    assert asyncio.run(notification_scheduler.send_due_weekly_digests(now)) == 1
    assert asyncio.run(notification_scheduler.send_due_weekly_digests(now)) == 0
    assert sends == [('user@example.com', 5, 1, {'review_required': 1, 'unavailable': 1,
                                               'policy_matches': 1, 'quarantine_recommended': 1})]


def test_phishing_cue_chart_excludes_reviews_unknown_policy_and_false_boolean_keys():
    test_engine = in_memory_engine()
    with Session(test_engine) as session:
        for index, (status, analysis) in enumerate([
            ('SUSPICIOUS', {'content_analysis': {'has_urgency_language': True, 'has_impersonation': True}}),
            ('UNKNOWN', {'content_analysis': {'has_urgency_language': True}}),
            ('PHISHING', {'security_action': 'BLOCK_BY_POLICY', 'content_analysis': {'has_urgency_language': True}}),
            ('PHISHING', {'content_analysis': {'has_urgency_language': False, 'has_impersonation': False}}),
        ]):
            record = ScanRecord(scan_id=f'cue-{index}', user_id=1, sender='sender@example.com',
                subject='Message', status=status, risk_level='critical')
            record.full_analysis = analysis
            session.add(record)
        other = ScanRecord(scan_id='other-cue', user_id=2, sender='other@example.com', subject='Other',
                            status='PHISHING', risk_level='critical')
        other.full_analysis = {'content_analysis': {'has_urgency_language': True}}
        session.add(other)
        session.commit()
        result = store.get_phishing_types(session, user_id=1)
        assert result['labels'] == ['Classification detail unavailable']
        assert result['values'] == [1]
        second = store.get_phishing_types(session, user_id=2)
        assert second['labels'] == ['Account-action wording']
        assert second['values'] == [1]
        empty = store.get_phishing_types(session, user_id=3)
        assert empty['labels'] == empty['values'] == []


def test_sync_and_weekly_summary_words_cannot_conflate_review_with_phishing(monkeypatch):
    captured = []
    async def fake_send(email, name, subject, heading, lines):
        captured.append(lines)
        return True
    monkeypatch.setattr(email_alerts, '_send_summary_email', fake_send)
    for send in (email_alerts.send_sync_summary, email_alerts.send_weekly_digest):
        assert asyncio.run(send('user@example.com', 'User', 5, 1, review_required=2, unavailable=1,
                                 policy_matches=1, quarantine_recommended=1))
    for lines in captured:
        text = ' '.join(lines)
        assert '1 recorded phishing classifications' in text
        assert '2 scans require review' in text
        assert '1 scans could not be classified reliably' in text
        assert '1 manual policy matches' in text
        assert 'not independently confirmed attacks' in text
        assert 'not executed message moves' in text
        assert 'phishing or suspicious' not in text
