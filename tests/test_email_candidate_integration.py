"""Candidate/live input boundaries and calibration source/runtime binding."""
import types
import sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from services import scan_engine as engine, security_pipeline as binding
from utils.feature_extractor import extract_features, extract_features_single
from utils.email_input_contract import build_email_input, contract_sha256


def test_raw_bundle_unknown_vocabulary_cannot_fake_a_clean_probability(monkeypatch):
    empty = types.SimpleNamespace(transform=lambda texts: csr_matrix((len(texts), 2)))
    features = types.SimpleNamespace(vectorizer_=empty)
    fold = types.SimpleNamespace(estimator=types.SimpleNamespace(named_steps={'features': features}))
    model = types.SimpleNamespace(calibrated_classifiers_=[fold], classes_=np.array([0, 1]),
            predict_proba=lambda texts: pytest.fail('OOV input must abstain before prediction'))
    monkeypatch.setattr(engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(engine, 'ML_MODEL', model)
    monkeypatch.setattr(engine, 'EMAIL_INPUT_KIND', 'raw_email_model_text')
    with pytest.raises(ValueError, match='No recognized email vocabulary'):
        engine.predict_email_probabilities(['Entirely unknown model vocabulary'])


def test_one_supported_fold_is_sufficient_and_raw_input_is_not_double_transformed(monkeypatch):
    vectors = [types.SimpleNamespace(transform=lambda texts: csr_matrix((len(texts), 2))),
               types.SimpleNamespace(transform=lambda texts: csr_matrix(np.ones((len(texts), 2))))]
    folds = [types.SimpleNamespace(estimator=types.SimpleNamespace(named_steps={'features': types.SimpleNamespace(vectorizer_=v)})) for v in vectors]
    received = []
    def predict(texts):
        received.extend(texts)
        return np.array([[.1, .9]])
    model = types.SimpleNamespace(calibrated_classifiers_=folds, classes_=np.array([0, 1]), predict_proba=predict)
    monkeypatch.setattr(engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(engine, 'ML_MODEL', model)
    monkeypatch.setattr(engine, 'EMAIL_INPUT_KIND', 'raw_email_model_text')
    text = 'Please review https://example.com/document today.'
    assert engine.predict_email_probabilities([text]).tolist() == [.9]
    assert received == [text]


def test_training_transform_and_scientific_runtime_are_bound_to_calibration(monkeypatch, tmp_path):
    transform = tmp_path / 'training/email_pipeline.py'
    transform.parent.mkdir()
    transform.write_text('first transform')
    monkeypatch.setattr(binding, 'ROOT', tmp_path)
    monkeypatch.setattr(binding, '_scoring_runtime_versions', lambda: (('scikit-learn', 'first'),))
    binding._fingerprint.cache_clear()
    first = binding.security_pipeline_id()
    transform.write_text('second transform: different features')
    second = binding.security_pipeline_id()
    assert first != second
    monkeypatch.setattr(binding, '_scoring_runtime_versions', lambda: (('scikit-learn', 'second'),))
    binding._fingerprint.cache_clear()
    assert binding.security_pipeline_id() != second
    binding._fingerprint.cache_clear()


def test_single_message_fast_features_preserve_batch_contract():
    texts = ['Verify your account! https://example.com/login, today?',
             '<html><a href="https://example.com">Document</a></html>',
             'URGENT transfer $25,000 to PAYPAL immediately!',
             'A normal business meeting agenda.', 'Unicode Réunion جمعرات', '']
    assert np.array_equal(extract_features(texts), np.vstack([extract_features_single(t) for t in texts]))


def structured_runtime(monkeypatch, *, probability=.02, action='SAFE_DELIVERY'):
    """Controlled dependencies exercise the production branch, not real accuracy."""
    received, scanned = [], []
    feature = types.SimpleNamespace(supported=lambda rows: np.ones(len(rows), dtype=bool))
    def predict(rows):
        received.extend(rows)
        return np.array([[1 - probability, probability] for _ in rows])
    model = types.SimpleNamespace(classes_=np.array([0, 1]),
        named_steps={'features': feature}, predict_proba=predict)
    monkeypatch.setattr(engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(engine, 'ML_MODEL', model)
    monkeypatch.setattr(engine, 'TFIDF_VECTORIZER', None)
    monkeypatch.setattr(engine, 'EMAIL_INPUT_KIND', engine.STRUCTURED_EMAIL_INPUT_KIND)
    def urls(text, *, additional_urls=None):
        destinations = list(dict.fromkeys(engine.extract_urls(text) + list(additional_urls or [])))
        scanned.extend(destinations)
        return {'score': 0, 'urls_found': len(destinations), 'inference_errors': []}
    monkeypatch.setattr(engine, 'score_urls', urls)
    vector = types.ModuleType('services.vector_store')
    vector.search_similar_campaigns = lambda *args, **kwargs: {'has_match': False}
    monkeypatch.setitem(sys.modules, 'services.vector_store', vector)
    bec = types.ModuleType('services.bec_detector')
    bec.predict_bec_intent = lambda *args, **kwargs: {'is_bec': False, 'bec_score': 0}
    monkeypatch.setitem(sys.modules, 'services.bec_detector', bec)
    class Predictor:
        is_calibrated = True
        pipeline_id = engine.security_pipeline_id()
        def predict_set(self, *args, **kwargs):
            return ({'PHISHING'}, 'PHISHING', action) if action == 'AUTO_QUARANTINE' else ({'CLEAN'}, 'CLEAN', action)
    return received, scanned, Predictor(), model


def test_structured_training_live_contract_parity_preserves_original_evidence(monkeypatch):
    received, scanned, predictor, _ = structured_runtime(monkeypatch)
    kwargs = dict(subject='Original subject', sender='Editor <a@example.org>',
        headers='Reply-To: editor@example.org\nAuthentication-Results: relay; spf=pass\n',
        html_body='<a href="https://example.org/report">Report</a>',
        parsing_info={'text_complete': True, 'source_kind': 'rfc_email'},
        supplementary_urls=['https://example.org/help'])
    body = 'The complete original meeting agenda.'
    expected = build_email_input(body, **kwargs)
    result = engine.evaluate_email_security(body, predictor=predictor, **kwargs)
    assert received == [expected]
    assert scanned == expected['link_destinations']
    assert result['email_input_diagnostics']['contract_sha256'] == contract_sha256(expected)
    assert result['email_input_diagnostics']['model_input_kind'] == engine.STRUCTURED_EMAIL_INPUT_KIND
    assert result['header_result']['verified'] is False
    assert result['conformal_res']['threat_probability'] == .02


def test_lazy_loading_structured_bundle_never_predicts_without_original_contract(monkeypatch):
    received, _, predictor, _ = structured_runtime(monkeypatch)
    monkeypatch.setattr(engine, 'MODEL_LOADED', False)
    monkeypatch.setattr(engine, 'EMAIL_INPUT_KIND', 'legacy_tfidf_structural')
    loads = []
    def load():
        loads.append(True)
        engine.MODEL_LOADED = True
        engine.EMAIL_INPUT_KIND = engine.STRUCTURED_EMAIL_INPUT_KIND
    monkeypatch.setattr(engine, 'load_ml_model', load)
    body = 'Meeting agenda for tomorrow.'
    result = engine.evaluate_email_security(body, subject='Agenda', predictor=predictor)
    assert loads == [True]
    assert received == [build_email_input(body, subject='Agenda')]
    assert result['content_result']['ml_powered']
    assert result['conformal_res']['threat_probability'] == .02


def test_structured_inference_requires_contract_and_rejects_tampered_fields(monkeypatch):
    received, _, _, _ = structured_runtime(monkeypatch)
    bare = engine.score_content('Meeting agenda for tomorrow.')
    assert bare['ml_probability'] is None and not bare['ml_powered']
    assert received == []
    contract = build_email_input('Meeting agenda for tomorrow.')
    contract['link_destinations'] = ['https://invented.example/']
    result = engine.score_content('Meeting agenda for tomorrow.', email_input=contract)
    assert result['ml_probability'] is None and not result['ml_powered']
    assert received == []


def test_structured_unknown_vocabulary_abstains_before_prediction(monkeypatch):
    received, _, _, model = structured_runtime(monkeypatch)
    model.named_steps['features'].supported = lambda rows: np.zeros(len(rows), dtype=bool)
    result = engine.score_content('Unrecognized message text.',
        email_input=build_email_input('Unrecognized message text.'))
    assert result['ml_probability'] is None
    assert 'No recognized Email' in result['ml_error']
    assert received == []


def test_unsupported_destinations_trigger_coverage_review_without_changing_probability(monkeypatch):
    received, scanned, predictor, _ = structured_runtime(monkeypatch)
    result = engine.evaluate_email_security('Review the meeting agenda.', predictor=predictor,
        supplementary_urls=['mailto:editor@example.org', 'https://[', 'https://example.org/help'])
    assert scanned == ['https://example.org/help']
    assert len(received[0]['unsupported_urls']) == 2
    assert result['email_input_diagnostics']['unsupported_link_count'] == 2
    assert result['security_action'] == 'WARNING_BANNER'
    assert result['conformal_res']['threat_probability'] == .02
    assert result['final_score'] == 2
    assert any('not a phishing finding' in value for value in result['sensor_errors'])


def test_qr_destinations_remain_separate_from_trained_structured_input(monkeypatch):
    received, scanned, predictor, _ = structured_runtime(monkeypatch)
    vision = types.ModuleType('services.vision_scanner')
    vision.analyze_image_attachment = lambda *args, **kwargs: {
        'is_valid_image': True, 'qr_scan_complete': True, 'has_qr': True,
        'qr_urls': ['https://example.org/image-target', 'javascript:alert(1)']}
    monkeypatch.setitem(sys.modules, 'services.vision_scanner', vision)
    result = engine.evaluate_email_security('Review the meeting agenda.', predictor=predictor,
        attachments=[{'data': b'engineering image fixture', 'filename': 'image.png'}])
    assert received == [build_email_input('Review the meeting agenda.')]
    assert scanned == ['https://example.org/image-target']
    assert result['email_input_diagnostics']['attachment_qr'] == {
        'included_in_model_input': False, 'link_destination_count': 1, 'unsupported_link_count': 1}
    assert result['security_action'] == 'WARNING_BANNER'
    assert result['conformal_res']['threat_probability'] == .02


def test_structured_incomplete_parsing_cannot_retain_quarantine_calibration(monkeypatch):
    received, _, predictor, _ = structured_runtime(monkeypatch, probability=.99, action='AUTO_QUARANTINE')
    result = engine.evaluate_email_security('Partially decoded meeting agenda.', predictor=predictor,
        parsing_info={'text_complete': False, 'errors': ['Invalid transfer encoding']})
    assert received[0]['parsing']['completeness'] == 'INCOMPLETE'
    assert result['security_action'] == 'WARNING_BANNER'
    assert not result['conformal_res']['calibration_valid']
    assert result['conformal_res']['threat_probability'] == .99


def test_new_contract_source_changes_invalidate_saved_pipeline_identity(monkeypatch, tmp_path):
    file = tmp_path / 'utils/email_input_contract.py'
    file.parent.mkdir()
    file.write_text('original contract')
    monkeypatch.setattr(binding, 'ROOT', tmp_path)
    binding._fingerprint.cache_clear()
    first = binding.security_pipeline_id()
    file.write_text('changed original contract field handling')
    assert binding.security_pipeline_id() != first
    binding._fingerprint.cache_clear()


def approved_artifact_fixture():
    """Artificial admission metadata tests validation, never deployment quality."""
    from training.email_evidence_model import contract_source_hashes, library_versions
    model = types.SimpleNamespace(classes_=np.array([0, 1]),
        named_steps={'features': types.SimpleNamespace(supported=lambda rows: np.ones(len(rows), dtype=bool))},
        predict_proba=lambda rows: np.array([[.8, .2] for _ in rows]))
    return dict(schema_version=3, input_kind=engine.STRUCTURED_EMAIL_INPUT_KIND,
        input_contract_version='email-evidence-v1', approved_for_deployment=True,
        deployment_generalization_established=True, contract_source_hashes=contract_source_hashes(),
        library_versions=library_versions(), protocol_sha256='a' * 64, corpus_sha256='b' * 64,
        model=model)


def test_structured_research_artifact_cannot_be_loaded_as_approved_model():
    artifact = approved_artifact_fixture()
    artifact['approved_for_deployment'] = False
    with pytest.raises(ValueError, match='deployment approval'):
        binding.validate_structured_email_artifact(artifact)


@pytest.mark.parametrize('field,mutation', [
    ('input_contract_version', 'different-contract'),
    ('deployment_generalization_established', False),
    ('contract_source_hashes', {}), ('library_versions', {}),
    ('corpus_sha256', 'missing')])
def test_structured_artifact_cannot_bypass_source_runtime_and_evaluation_bindings(field, mutation):
    artifact = approved_artifact_fixture()
    artifact[field] = mutation
    with pytest.raises(ValueError):
        binding.validate_structured_email_artifact(artifact)
