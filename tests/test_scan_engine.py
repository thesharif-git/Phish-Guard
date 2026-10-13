import os
import sys
import pytest
import numpy as np

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extractor import clean_text, extract_features
from services.scan_engine import score_content, load_ml_model, ML_MODEL, TFIDF_VECTORIZER


def test_clean_text():
    sample_html = "<html><body><h1>URGENT ALERT</h1><a href='http://phish.xyz'>Click here</a></body></html>"
    cleaned = clean_text(sample_html)
    assert "urgent alert" in cleaned
    assert "http" not in cleaned
    assert "<" not in cleaned and ">" not in cleaned


def test_extract_features_single():
    text = "URGENT: Click here http://paypa1-verify.xyz to confirm your account details immediately!"
    features = extract_features(text)
    assert isinstance(features, np.ndarray)
    assert features.shape == (1, 16)
    # Feature 0 is url_count
    assert features[0, 0] == 1.0
    # Feature 4 is urgent_words count (urgent, click here, confirm)
    assert features[0, 4] >= 2.0


def test_extract_features_batch():
    texts = [
        "Normal email text hello world",
        "URGENT phishing email with link http://bad.xyz"
    ]
    features = extract_features(texts)
    assert isinstance(features, np.ndarray)
    assert features.shape == (2, 16)


def test_scan_engine_inference():
    load_ml_model()
    phishing_text = "URGENT: Your account has been suspended! Please verify your password at http://paypal-verify.xyz immediately."
    res = score_content(phishing_text)
    
    assert "score" in res
    assert res["score"] > 0
    assert res["has_urgency_language"] is True
    assert res["is_english"] is True
    if ML_MODEL and TFIDF_VECTORIZER:
        assert res["ml_powered"] is True
        assert res["ml_probability"] is not None
        assert 0.0 <= res["ml_probability"] <= 1.0


def test_multilingual_scoring():
    # Spanish phishing lure test (Latin script with vocabulary and structural signals)
    es_lure = "Aviso urgente: Su cuenta bancaria ha sido suspendida. Por favor haga clic aquí para verificar su contraseña hoy."
    res_es = score_content(es_lure)
    assert res_es["language"] == "Spanish"
    assert res_es["is_english"] is False
    assert len(res_es["keyword_hits"]) > 0

    # Arabic phishing lure test (Non-Latin script: 0 English vocabulary tokens)
    ar_lure = "تنبيه أمني عاجل: تم تعليق حسابك مؤقتاً بسبب نشاط غير عادي. اضغط هنا لتأكيد حسابك."
    res_ar = score_content(ar_lure)
    assert res_ar["language"] == "Arabic"
    assert res_ar["is_english"] is False
    assert len(res_ar["keyword_hits"]) > 0
    # In pure statistical architecture, unrecognized script does not synthesize a fake score
    assert res_ar["score"] is None
    assert res_ar["score_source"] == "unavailable"
    assert res_ar["ml_powered"] is False


def test_quishing_email_scan():
    """Verify scan_email processes image attachment, extracts QR code URL, and flags Quishing."""
    import cv2
    from services.scan_engine import scan_email
    from data.database import engine
    from sqlmodel import Session

    # Generate a QR code containing a phishing URL
    phish_url = "https://secure-login-m365-verify.xyz/auth"
    encoder = cv2.QRCodeEncoder.create()
    qr_matrix = encoder.encode(phish_url)
    qr_img = cv2.resize(qr_matrix, (200, 200), interpolation=cv2.INTER_NEAREST)
    qr_bgr = cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)
    _, encoded = cv2.imencode(".png", qr_bgr)
    qr_bytes = encoded.tobytes()

    attachments = [{
        "filename": "m365_qr_code.png",
        "content_type": "image/png",
        "data": qr_bytes
    }]

    with Session(engine) as session:
        res = scan_email(
            content="Please scan the attached QR code to verify your corporate account.",
            session=session,
            sender="admin@m365-support.xyz",
            subject="Action Required: MFA QR Verification",
            attachments=attachments
        )

        assert "vision_intelligence" in res
        assert res["vision_intelligence"]["has_quishing"] is True
        assert phish_url in res["vision_intelligence"]["qr_urls"]
        model_probability = res["content_analysis"].get("ml_probability")
        if model_probability is None:
            assert res["final_score"] is None
        else:
            assert res["final_score"] == round(model_probability * 100)
        assert res["security_action"] in ("WARNING_BANNER", "AUTO_QUARANTINE")
        assert any("Quishing" in sig for sig in res["explanation"]["risk_signals"])


def test_score_urls_risk_aware_triage_prioritizes_suspicious_links(monkeypatch):
    """Verify that when links exceed 10, risk triage prioritizes threats over benign padding."""
    from services.scan_engine import score_urls

    # 10 benign links followed by 2 high-risk links
    benign_links = [f"https://www.google.com/search?q=query{i}" for i in range(10)]
    phish_shortener = "https://bit.ly/security-update-now"
    phish_ip = "http://192.168.1.100/banking-login"
    all_test_urls = benign_links + [phish_shortener, phish_ip]

    monkeypatch.setattr("services.scan_engine.resolve_short_url", lambda u: u)

    result = score_urls("Here are our company links: " + " ".join(all_test_urls))

    analyzed_urls = [u["url"] for u in result.get("analyzed_urls", [])]
    assert len(analyzed_urls) == 10
    assert result.get("urls_analyzed") == 10
    # Both suspicious links MUST be prioritized into the analyzed set
    assert phish_shortener in analyzed_urls, "High-risk shortener was unexpectedly dropped by link cap"
    assert phish_ip in analyzed_urls, "High-risk IP address destination was unexpectedly dropped by link cap"
    # The incomplete-evidence sensor warning MUST still be present
    assert any("exceeds the 10-destination limit" in err for err in result.get("inference_errors", []))


def test_resolve_short_url_caching(monkeypatch):
    """Verify resolve_short_url caches resolutions thread-safely."""
    from services.scan_engine import resolve_short_url, _SHORT_URL_CACHE, _SHORT_URL_LOCK

    call_count = 0
    test_short = "https://bit.ly/test-cache-destination"

    def mock_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = type("MockResponse", (), {"url": "https://example-target.com/landing"})()
        return mock_resp

    monkeypatch.setattr("requests.get", mock_get)

    with _SHORT_URL_LOCK:
        _SHORT_URL_CACHE.pop(test_short, None)

    res1 = resolve_short_url(test_short)
    assert res1 == "https://example-target.com/landing"
    assert call_count == 1

    # Second call must hit the cache without network requests
    res2 = resolve_short_url(test_short)
    assert res2 == "https://example-target.com/landing"
    assert call_count == 1

