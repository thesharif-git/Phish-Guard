"""
PhishGuard AI — Phase 2 URL Scanner Integration & Non-Regression Tests
Verifies that CalibratedUrlPredictor and extract_url_features satisfy:
- Exact probabilistic calibration in [0.0, 1.0]
- High discrimination on known malicious vs official URLs
- Pure-CPU single-item latency under 3.0 ms
- Seamless integration with score_urls and scan_email
"""
import time
import numpy as np
import pytest
import services.scan_engine as scan_engine
from utils.url_features import (
    extract_url_features,
    extract_url_features_single,
    extract_url_features_batch,
    URL_FEATURE_NAMES
)


def test_url_feature_extraction_contract():
    """Verify 33 features are extracted correctly with proper types."""
    url = "https://login.apple-account-update.xyz/verify?token=12345"
    d = extract_url_features(url)
    assert isinstance(d, dict)
    assert len(d) == 33
    for name in URL_FEATURE_NAMES:
        assert name in d, f"Missing feature: {name}"
        assert isinstance(d[name], (int, float, np.floating, np.integer))

    arr = extract_url_features_single(url)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 33)
    assert arr.dtype == np.float32

    batch = extract_url_features_batch([url, "https://www.google.com"])
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (2, 33)
    assert batch.dtype == np.float32


def test_calibrated_url_predictor_inference(monkeypatch):
    """Verify calibrated URL model inference behavior and probabilities."""
    monkeypatch.setattr('utils.threat_intel.check_virustotal', lambda *a: {'available': False})
    scan_engine.load_ml_model()
    assert scan_engine.URL_MODEL is not None, "URL model should be loaded"

    # Known phishing link
    phish_url = "https://login.apple-id-verify-account.top/auth/session"
    prob_phish = float(scan_engine.URL_MODEL.predict_proba([phish_url])[0, 1])
    assert prob_phish >= 0.80, f"Expected high calibrated probability for phishing, got {prob_phish}"

    res_phish = scan_engine.score_urls(phish_url)
    assert "score" in res_phish
    assert res_phish["score"] >= 80, f"Expected high risk for phishing URL, got {res_phish['score']}"
    assert len(res_phish.get("inference_errors", [])) == 0

    # Clean official URL
    clean_url = "https://www.google.com/intl/en/about/"
    prob_clean = float(scan_engine.URL_MODEL.predict_proba([clean_url])[0, 1])
    assert prob_clean <= 0.20, f"Expected low calibrated probability for clean URL, got {prob_clean}"

    res_clean = scan_engine.score_urls(clean_url)
    assert res_clean["score"] <= 20, f"Expected low risk for clean URL, got {res_clean['score']}"
    assert len(res_clean.get("inference_errors", [])) == 0


def test_url_single_item_latency_budget():
    """Verify single-URL CPU inference executes within 3.0 ms budget."""
    scan_engine.load_ml_model()
    assert scan_engine.URL_MODEL is not None
    url = "https://auth.microsoft-security-notice.com/login"

    # Warmup
    _ = scan_engine.URL_MODEL.predict_proba([url])

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = scan_engine.URL_MODEL.predict_proba([url])
        latencies.append((time.perf_counter() - t0) * 1000)

    p95 = float(np.percentile(latencies, 95))
    assert p95 <= 3.0, f"P95 latency {p95:.2f}ms exceeds the 3.0ms gate budget"


def test_url_scanner_bounded_deadline_on_hanging_workers(monkeypatch):
    """Verify that scan_url enforces a strictly bounded 6s total join deadline even when workers hang."""
    from utils import url_scanner

    # Mock all enrichment workers to sleep indefinitely (10s each)
    monkeypatch.setattr(url_scanner, "full_threat_check", lambda *a: time.sleep(10) or {})
    monkeypatch.setattr(url_scanner, "analyze_live_html", lambda *a: time.sleep(10) or {})
    monkeypatch.setattr(url_scanner, "geolocate_ip", lambda *a: time.sleep(10) or {})
    monkeypatch.setattr(url_scanner, "analyze_tls_cert_lineage", lambda *a: time.sleep(10) or {})
    monkeypatch.setattr(url_scanner, "calculate_dns_ephemerality", lambda *a: time.sleep(10) or {})
    monkeypatch.setattr(url_scanner, "check_asn_brand_incongruity", lambda *a: time.sleep(10) or {})

    t0 = time.time()
    res = url_scanner.scan_url("https://example.org/test")
    elapsed = time.time() - t0

    # Total elapsed must be bounded around 6.0s (definitely under 7.5s), NOT 6x6=36 seconds
    assert elapsed < 7.5, f"Expected bounded join timeout under 7.5s, but took {elapsed:.2f}s"
    assert "final_score" in res
    assert "ml_probability" in res

