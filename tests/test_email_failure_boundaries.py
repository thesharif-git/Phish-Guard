"""Actual inference/decision boundaries, using controlled failing dependencies."""
import asyncio
import base64
import os
import sys
import types
from email.message import EmailMessage
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from services import scan_engine as engine, inbox_connector as inbox, bec_detector as bec


def install_model(monkeypatch, classes=(0, 1), probabilities=((.8, .2),), recognized=True):
    model = types.SimpleNamespace(classes_=np.array(classes), predict_proba=lambda x: np.array(probabilities))
    vectorizer = types.SimpleNamespace(transform=lambda texts: csr_matrix(np.ones((len(texts), 2)) if recognized else np.zeros((len(texts), 2))))
    monkeypatch.setattr(engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(engine, 'EMAIL_INPUT_KIND', 'legacy_tfidf_structural')
    monkeypatch.setattr(engine, 'ML_MODEL', model)
    monkeypatch.setattr(engine, 'TFIDF_VECTORIZER', vectorizer)
    return model, vectorizer


def test_optional_model_load_failure_cannot_disable_email_inference(monkeypatch):
    model, vectorizer = install_model(monkeypatch)
    monkeypatch.setattr(engine, 'MODEL_LOADED', False)
    monkeypatch.setattr(engine, 'MODEL_LOAD_ERRORS', {})
    monkeypatch.setattr(engine, 'URL_MODEL', None)
    monkeypatch.setattr(engine, 'DOMAIN_MODEL', None)
    def load(path):
        name = os.path.basename(path)
        if name == 'phish_model.pkl':
            return model
        if name == 'vectorizer.pkl':
            return vectorizer
        raise ValueError('Deliberately corrupt auxiliary artifact')
    monkeypatch.setattr(engine.joblib, 'load', load)
    engine.load_ml_model()
    assert engine.MODEL_LOADED
    assert engine.MODEL_LOAD_ERRORS.keys() == {'url', 'domain'}
    assert engine.predict_email_probabilities(['Meeting agenda for tomorrow.']).tolist() == [.2]


def test_probability_column_uses_class_semantics_not_position(monkeypatch):
    install_model(monkeypatch, classes=(1, 0))
    assert engine.predict_email_probabilities(['Meeting agenda for tomorrow.']).tolist() == [.8]


@pytest.mark.parametrize('classes,probabilities', [((0, 2), ((.8, .2),)), ((0, 1), ((.8,),)), ((0, 1), ((float('nan'), float('nan')),)), ((0, 1), ((float('nan'), .2),)), ((0, 1), ((.8, .8),))])
def test_malformed_classifier_outputs_fail_explicitly(monkeypatch, classes, probabilities):
    install_model(monkeypatch, classes, probabilities)
    with pytest.raises(ValueError):
        engine.predict_email_probabilities(['Meeting agenda for tomorrow.'])


def test_zero_vocabulary_text_cannot_produce_a_fake_clean_probability(monkeypatch):
    install_model(monkeypatch, recognized=False)
    result = engine.score_content('Entirely unknown vocabulary for this model.')
    assert result['ml_probability'] is None and not result['ml_powered']
    assert 'No recognized email vocabulary' in result['ml_error']


def isolate_sensors(monkeypatch):
    monkeypatch.setattr(engine, 'score_content', lambda t: {'ml_powered': True, 'ml_probability': .99, 'score': 99})
    monkeypatch.setattr(engine, 'score_urls', lambda *a, **k: {'score': 0, 'urls_found': 0, 'inference_errors': []})
    vector = types.ModuleType('services.vector_store')
    vector.search_similar_campaigns = lambda t: {'has_match': False}
    monkeypatch.setitem(sys.modules, 'services.vector_store', vector)
    monkeypatch.setattr(bec, 'predict_bec_intent', lambda *a, **k: {'is_bec': False, 'bec_score': 0})
    class Predictor:
        is_calibrated = True
        pipeline_id = engine.security_pipeline_id()
        def predict_set(self, *a, **k):
            return {'PHISHING'}, 'PHISHING', 'AUTO_QUARANTINE'
    return Predictor()


def test_corrupt_or_partial_text_prevents_automatic_quarantine(monkeypatch):
    predictor = isolate_sensors(monkeypatch)
    result = engine.evaluate_email_security('Decoded but corrupt email content.', predictor=predictor,
                                           parsing_info={'text_complete': False, 'errors': ['bad encoding']})
    assert result['security_action'] == 'WARNING_BANNER'
    assert not result['conformal_res']['calibration_valid']
    assert result['conformal_res']['threat_probability'] == .99
    assert any('bad encoding' in s for s in result['sensor_errors'])


def test_supplementary_links_reach_url_evidence_without_changing_model_text(monkeypatch):
    predictor = isolate_sensors(monkeypatch)
    texts, destination_lists = [], []
    def content(text):
        texts.append(text)
        return {'ml_powered': True, 'ml_probability': .99, 'score': 99}
    def urls(text, **kwargs):
        destination_lists.append(kwargs.get('additional_urls'))
        return {'score': 0, 'urls_found': 1, 'inference_errors': []}
    monkeypatch.setattr(engine, 'score_content', content)
    monkeypatch.setattr(engine, 'score_urls', urls)
    result = engine.evaluate_email_security('Please review the document.', subject='Meeting', predictor=predictor,
                                            supplementary_urls=['https://example.com/hidden-link'])
    assert texts == ['Please review the document. Meeting']
    assert destination_lists == [['https://example.com/hidden-link']]
    assert 'not verified' in result['contra_result']['alignment_summary']


def test_invalid_supplementary_url_shape_is_not_silently_dropped(monkeypatch):
    predictor = isolate_sensors(monkeypatch)
    with pytest.raises(ValueError, match='list of HTTP'):
        engine.evaluate_email_security('Please review the document.', predictor=predictor,
                                       supplementary_urls='https://example.com/link')


def test_gmail_raw_uses_the_same_parser_and_keeps_image_payloads(monkeypatch):
    message = EmailMessage()
    message['Subject'] = 'Document review'
    message.set_content('Please review the meeting document.')
    message.add_alternative('<a href="https://example.com/help">Help</a>', subtype='html')
    message.add_attachment(b'image fixture bytes', maintype='image', subtype='png', filename='image.png')
    calls = []
    class Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url, **kwargs):
            calls.append(kwargs.get('params', {}))
            data = {'messages': [{'id': 'fixture'}]} if url == inbox.GMAIL_MESSAGES_URL else {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
            return types.SimpleNamespace(status_code=200, json=lambda: data)
    monkeypatch.setattr(inbox.httpx, 'AsyncClient', Client)
    result = asyncio.run(inbox.fetch_gmail_emails('fixture-token'))[0]
    assert calls[-1] == {'format': 'raw'}
    assert result['body'] == 'Please review the meeting document.'
    assert result['attachments'][0]['data'] == b'image fixture bytes'
    assert result['parsing']['text_complete']
    assert result['parsing']['source_kind'] == 'gmail_raw_rfc_email'
    assert 'https://example.com/help' in result['html_body']


@pytest.mark.parametrize('response', [{}, {'labels': [], 'scores': []}, {'labels': ['payroll bank account update'], 'scores': [.98]}])
def test_malformed_or_nonfraud_topic_label_cannot_be_semantic_bec_evidence(monkeypatch, response):
    monkeypatch.setattr(bec, '_get_bec_classifier', lambda: lambda *a, **k: response)
    result = bec.predict_bec_intent('Please review the meeting agenda for Thursday.')
    assert result['transformer_error']
    assert not result['transformer_available'] and not result['is_bec']


def test_short_bec_words_cannot_match_inside_unrelated_words(monkeypatch):
    monkeypatch.setattr(bec, '_get_bec_classifier', lambda: None)
    result = bec.predict_bec_intent('The hotpot cookbook and business newsletter are ready.')
    assert result['rule_score'] == 0 and not result['is_bec']
    actual = bec.predict_bec_intent('Please reply with your OTP and current password.')
    assert actual['rule_score'] > 0


def test_missing_embedding_weights_never_attempt_network_or_create_collection(monkeypatch):
    from services import vector_store
    calls = []
    module = types.ModuleType('sentence_transformers')
    def unavailable(name, **kwargs):
        calls.append(kwargs)
        raise OSError('Not cached')
    module.SentenceTransformer = unavailable
    monkeypatch.setitem(sys.modules, 'sentence_transformers', module)
    monkeypatch.setattr(vector_store, '_EMBEDDING_MODEL', None)
    monkeypatch.setattr(vector_store, '_get_chroma_collection', lambda: pytest.fail('Must not create collection'))
    result = vector_store.search_similar_campaigns('A meeting agenda is available for review.')
    assert not result['available'] and not result['verified_campaign_evidence']
    assert len(calls) == 2 and all(c == {'local_files_only': True} for c in calls)
    vector_store.search_similar_campaigns('A second meeting agenda is ready.')
    assert len(calls) == 2  # Failed cache load is not retried per email.


@pytest.mark.parametrize('raises', [False, True])
def test_qr_detector_failure_cannot_look_like_successful_no_qr_scan(monkeypatch, raises):
    from services import vision_scanner as vision
    monkeypatch.setattr(vision, 'decode_image_bytes', lambda b: np.zeros((16, 16, 3), dtype=np.uint8))
    class Broken:
        def detectAndDecodeMulti(self, image):
            raise RuntimeError('Controlled failure')
    monkeypatch.setattr(vision, 'get_qr_detector', lambda: Broken() if raises else None)
    result = vision.analyze_image_attachment(b'fixture')
    assert result['is_valid_image'] and not result['qr_scan_complete'] and result['qr_scan_error']


def test_obsolete_calibration_is_named_in_decision_basis():
    from services.conformal_evidential_engine import synthesize_email_security_opinion
    predictor = types.SimpleNamespace(is_calibrated=True, pipeline_id='old',
        predict_set=lambda *a, **k: ({'CLEAN', 'PHISHING'}, 'SUSPICIOUS', 'WARNING_BANNER'))
    result = synthesize_email_security_opinion(content_prob=.99, predictor=predictor, pipeline_id='new')
    assert not result['calibration_valid']
    assert result['decision_basis'] == 'calibration_unavailable_or_pipeline_mismatch'
