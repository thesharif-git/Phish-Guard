"""Review evidence must not fabricate a higher aggregate Email score."""
import sys
import types

import pytest

from services import bec_detector, conformal_evidential_engine as fusion, scan_engine as engine


QR_DESTINATION = 'https://example.invalid/review'


def isolate_email_sensors(monkeypatch, *, probability=.02, qr=False, bec=False):
    predictor_calls, url_inputs = [], []

    class Predictor:
        is_calibrated = True
        pipeline_id = 'display-integrity-fixture'

        def predict_set(self, value, **kwargs):
            predictor_calls.append(value)
            return {'CLEAN'}, 'CLEAN', 'SAFE_DELIVERY'

    predictor = Predictor()
    content_result = {
        'ml_powered': probability is not None, 'ml_probability': probability,
        'ml_error': None if probability is not None else 'Fake content model unavailable',
        'score': round(probability * 100) if probability is not None else 0,
        'keyword_hits': [], 'suspicious_phrases': [], 'has_urgency_language': False,
        'has_impersonation': False, 'is_english': True,
    }
    monkeypatch.setattr(engine, 'score_content', lambda text: dict(content_result))
    monkeypatch.setattr(engine, 'security_pipeline_id', lambda: predictor.pipeline_id)
    monkeypatch.setattr(fusion, 'CONFORMAL_PREDICTOR', predictor)

    def urls(text, **kwargs):
        url_inputs.append((text, kwargs))
        return {'score': 0, 'urls_found': int(QR_DESTINATION in text), 'inference_errors': [],
                'flagged_urls': [], 'analyzed_urls': []}

    monkeypatch.setattr(engine, 'score_urls', urls)
    monkeypatch.setattr(engine, 'build_threat_intel',
                        lambda *args: {'score': 0, 'matches': []})
    vector = types.ModuleType('services.vector_store')
    vector.search_similar_campaigns = lambda text: {'has_match': False, 'matches': []}
    monkeypatch.setitem(sys.modules, 'services.vector_store', vector)
    bec_result = {
        'is_bec': bec, 'bec_score': 85 if bec else 0,
        'intent_category': 'CEO Wire Fraud' if bec else 'Clean',
        'score_kind': 'unvalidated_review_score',
        'signals': ['Mocked financial-request review signal'] if bec else [],
    }
    monkeypatch.setattr(bec_detector, 'predict_bec_intent', lambda *a, **k: dict(bec_result))
    vision = types.ModuleType('services.vision_scanner')
    vision.analyze_image_attachment = lambda *a, **k: {
        'is_valid_image': True, 'has_qr': qr, 'qr_scan_complete': True,
        'qr_urls': [QR_DESTINATION] if qr else [],
    }
    monkeypatch.setitem(sys.modules, 'services.vision_scanner', vision)
    # Fail loudly if a supposedly isolated regression reaches network or weights.
    monkeypatch.setattr(engine.requests.sessions.Session, 'request',
                        lambda *a, **k: pytest.fail('No network permitted in this regression'))
    monkeypatch.setattr(engine, 'load_ml_model',
                        lambda: pytest.fail('No model weights permitted in this regression'))
    attachments = [{'data': b'fake image fixture', 'filename': 'fixture.png'}] if qr else None
    return predictor, attachments, predictor_calls, url_inputs


@pytest.mark.parametrize('qr,bec', [(False, True), (True, False), (True, True)])
def test_low_model_score_is_not_inflated_by_separate_review_evidence(monkeypatch, qr, bec):
    predictor, attachments, calls, url_inputs = isolate_email_sensors(monkeypatch, qr=qr, bec=bec)
    result = engine.evaluate_email_security('Please review the attached meeting document.',
                                            attachments=attachments, predictor=predictor)
    assert result['content_result']['ml_probability'] == .02
    assert result['conformal_res']['threat_probability'] == .02
    assert result['final_score'] == result['conformal_res']['final_score'] == 2
    assert 'floor_applied' not in result['conformal_res']
    assert result['security_action'] == 'WARNING_BANNER'
    assert result['conformal_res']['decision_basis'] == 'separate_evidence_review'
    assert calls == [.02]
    evidence = result['conformal_res']['heuristic_evidence']
    assert evidence['bec_detected'] is bec and evidence['qr_detected'] is qr
    assert result['bec_result']['is_bec'] is bec
    assert result['vision_result']['has_quishing'] is qr
    if bec:
        assert result['bec_result']['bec_score'] == 85
        assert result['bec_result']['signals'] == ['Mocked financial-request review signal']
    if qr:
        assert result['qr_urls_extracted'] == [QR_DESTINATION]
        assert QR_DESTINATION in url_inputs[0][0]


def test_missing_ml_remains_unavailable_and_requires_review(monkeypatch):
    predictor, attachments, calls, _ = isolate_email_sensors(monkeypatch, probability=None,
                                                            qr=True, bec=True)
    result = engine.evaluate_email_security('Please review the attached meeting document.',
                                            attachments=attachments, predictor=predictor)
    assert result['content_result']['ml_probability'] is None
    assert not result['content_result']['ml_powered']
    assert result['conformal_res']['threat_probability'] is None
    assert not result['conformal_res']['calibration_valid']
    assert result['security_action'] == 'WARNING_BANNER'
    assert result['final_score'] is None and result['conformal_res']['final_score'] is None
    assert 'floor_applied' not in result['conformal_res']
    assert any('Fake content model unavailable' in error for error in result['sensor_errors'])
    assert not calls  # Unavailable content cannot enter statistical prediction.
    assert result['conformal_res']['heuristic_evidence']['bec_detected']
    assert result['conformal_res']['heuristic_evidence']['qr_detected']


def test_complete_unavailable_scan_returns_review_with_no_numeric_estimate(monkeypatch):
    isolate_email_sensors(monkeypatch, probability=None)
    saved = []
    monkeypatch.setattr('data.store.is_blocked', lambda *a: False)
    monkeypatch.setattr('data.store.add_scan', lambda result, *a, **k: saved.append(result))
    result = engine.scan_email('Please review the meeting document.', session=object())
    assert result['final_score'] is None
    assert result['pillars']['pillar4_fusion']['final_score'] is None
    assert result['pillars']['pillar4_fusion']['threat_probability'] is None
    assert result['security_action'] == 'WARNING_BANNER'
    assert result['status'] == 'SUSPICIOUS'
    assert 'Review required' in result['explanation']['summary']
    assert saved == [result]


def test_unavailable_summary_does_not_infer_safety_from_placeholder(monkeypatch):
    _, _, _, _ = isolate_email_sensors(monkeypatch, probability=None)
    explanation = engine.build_explanation(None, {}, {'flagged_urls': [], 'urls_found': 0}, {})
    assert 'unavailable' in explanation['summary'].lower()
    assert 'No risk score is available' in explanation['ai_paragraph']
    assert 'No significant phishing indicators' not in explanation['summary']


def test_complete_scan_keeps_qr_and_bec_warnings_without_inventing_aggregate_score(monkeypatch):
    _, attachments, _, url_inputs = isolate_email_sensors(monkeypatch, qr=True, bec=True)
    saved = []
    monkeypatch.setattr('data.store.is_blocked', lambda *a: False)
    monkeypatch.setattr('data.store.add_scan', lambda result, *a, **k: saved.append(result))
    result = engine.scan_email('Please review the attached meeting document.', session=object(),
                               attachments=attachments)
    assert result['final_score'] == result['pillars']['pillar4_fusion']['final_score'] == 2
    assert result['pillars']['pillar4_fusion']['threat_probability'] == .02
    assert result['security_action'] == 'WARNING_BANNER'
    assert result['bec_intelligence']['is_bec'] and result['vision_intelligence']['has_quishing']
    assert QR_DESTINATION in url_inputs[0][0]
    warnings = result['explanation']['risk_signals']
    assert any('Possible Business Email Compromise' in warning for warning in warnings)
    assert any('QR destination extracted' in warning and QR_DESTINATION in warning for warning in warnings)
    assert 'Review required' in result['explanation']['summary']
    assert saved == [result]
