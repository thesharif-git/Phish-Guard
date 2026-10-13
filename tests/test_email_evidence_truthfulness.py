"""Concrete false-evidence regressions, unrelated to predictive accuracy."""
import pytest
from services import scan_engine as engine


@pytest.mark.parametrize('host', ['paypa1-alert.com.evil.example', 'notpaypa1-alert.com', 'safe.example'])
def test_unrelated_hostname_does_not_match_local_list(host):
    assert not engine._matches_local_domain_list(host)


@pytest.mark.parametrize('host', ['paypa1-alert.com', 'sub.paypa1-alert.com', 'PAYPA1-ALERT.COM.'])
def test_hostname_and_real_subdomain_match_local_list(host):
    assert engine._matches_local_domain_list(host)


def test_blocklisted_text_in_url_path_is_not_a_threat_feed_verdict():
    result = engine.build_threat_intel({}, {'flagged_urls': [
        {'url': 'https://safe.example/document/paypa1-alert.com?next=corp-helpdesk.xyz'}]}, 'sender@notpaypa1-alert.com')
    assert result['matches'] == [] and result['score'] == 0


def test_text_cooccurrence_is_not_duplicated_as_threat_intelligence():
    result = engine.build_threat_intel({'has_impersonation': True, 'has_urgency_language': True}, {}, '')
    assert result['matches'] == []
    assert result['heuristic_observations']


def test_brand_mentions_do_not_establish_impersonation_or_probability_confidence():
    result = engine.build_explanation(85, {'has_impersonation': True, 'ml_powered': True, 'ml_probability': .85}, {}, {})
    text = ' '.join(result['risk_signals'])
    assert 'does not establish impersonation' in text
    assert 'probability' in text and 'confidence' not in text


def test_low_display_score_does_not_erase_real_observations():
    result = engine.build_explanation(10, {'has_impersonation': True}, {}, {})
    assert any('Brand and account-action' in r for r in result['risk_signals'])


def test_email_pillar_does_not_turn_qr_or_credential_topic_into_proven_fraud():
    from services.origintrace_pillars import SemanticDeception
    result = SemanticDeception.analyze({'bec_score': 49, 'intent_category': 'Credential review'},
        {'has_urgency_language': True}, {}, {'has_quishing': True})
    text = ' '.join(result['signals'])
    assert 'unvalidated review score' in text
    assert 'QR presence alone' in text
    assert 'legitimate notifications' in text
    assert not result['threat_probability_validated']


def test_email_sender_relay_mismatch_is_an_observation_not_verified_spoofing():
    from services.origintrace_pillars import ProvenanceMatrix
    result = ProvenanceMatrix.analyze('Spotify <alias@privaterelay.appleid.com>', '')
    assert result['display_name_domain_difference']
    assert not result['display_name_spoofed']
    assert any('relay use is possible' in s for s in result['signals'])


def test_account_security_topic_does_not_establish_manipulation():
    result = engine.build_explanation(5, {'suspicious_phrases': ['two-factor']}, {}, {})
    assert any('manipulation is not established' in s for s in result['risk_signals'])


def test_embedded_link_score_is_not_redirect_evidence(monkeypatch):
    import numpy as np
    import utils.threat_intel as intel
    class Model:
        def predict_proba(self, frame):
            return np.array([[.01, .99]])
    monkeypatch.setattr(engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(engine, 'URL_MODEL', Model())
    monkeypatch.setattr(engine, 'DOMAIN_MODEL', None)
    monkeypatch.setattr(engine, 'numeric_url_frame', lambda model, records: records)
    monkeypatch.setattr(engine, 'resolve_short_url', lambda url: url)
    monkeypatch.setattr(intel, 'check_virustotal', lambda url: {'available': False})
    result = engine.score_urls('Visit https://is.gd/abc')
    assert result['score'] == 99
    assert not result['has_redirect']
    assert result['has_shortener']
    assert result['analyzed_urls'][0]['model_probability'] == .99
    assert not result['analyzed_urls'][0]['threat_probability_validated']
    result = engine.score_urls('Visit https://safe.example/path/is.gd')
    assert not result['has_shortener']


def test_tld_text_in_path_is_not_infrastructure_reputation_evidence():
    from services.origintrace_pillars import InfrastructureLineage
    result = InfrastructureLineage.analyze('', ['https://safe.example/file.xyz'], 0)
    assert result['score'] == 0
    assert not result['threat_probability_validated']
    result = InfrastructureLineage.analyze('', ['https://example.xyz'], 0)
    assert result['score'] > 0
    assert any('not established' in s for s in result['signals'])
