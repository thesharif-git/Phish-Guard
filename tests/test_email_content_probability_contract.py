"""Backend contract checks; these do not establish model detection accuracy."""
import numpy as np
import pytest
from services import scan_engine as engine


@pytest.mark.parametrize('text', [
    'URGENT: Verify your password immediately or your account will be suspended.',
    'Aviso urgente: Su cuenta bancaria ha sido suspendida. Verificar su contraseña hoy.',
    'تنبيه أمني عاجل: تم تعليق حسابك مؤقتاً. اضغط هنا لتأكيد حسابك.',
    'আপনার অ্যাকাউন্ট যাচাই করুন এবং পাসওয়ার্ড নিশ্চিত করুন।',
])
def test_phrase_observations_do_not_inflate_model_probability(monkeypatch, text):
    monkeypatch.setattr(engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(engine, 'ML_MODEL', object())
    monkeypatch.setattr(engine, 'TFIDF_VECTORIZER', object())
    monkeypatch.setattr(engine, 'EMAIL_INPUT_KIND', 'raw_email_model_text')
    monkeypatch.setattr(engine, 'predict_email_probabilities', lambda values: np.array([0.12]))
    result = engine.score_content(text)
    assert result['score'] == 12
    assert result['ml_probability'] == 0.12
    assert result['score_source'] == 'email_model_probability'
    assert result['phrase_matches_are_phishing_proof'] is False
    assert result['language_detection']['validated'] is False


def test_failed_model_does_not_fall_back_to_phrase_score(monkeypatch):
    monkeypatch.setattr(engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(engine, 'ML_MODEL', object())
    monkeypatch.setattr(engine, 'TFIDF_VECTORIZER', object())
    monkeypatch.setattr(engine, 'EMAIL_INPUT_KIND', 'raw_email_model_text')
    def fail(values):
        raise ValueError('Test inference failure')
    monkeypatch.setattr(engine, 'predict_email_probabilities', fail)
    result = engine.score_content('URGENT verify your password immediately account suspended')
    assert result['score'] is None
    assert result['ml_probability'] is None
    assert result['score_source'] == 'unavailable'
    assert result['ml_error'] == 'Test inference failure'
