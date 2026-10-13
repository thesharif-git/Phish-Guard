"""
PhishGuard AI — Phase 3 Domain Scanner Integration & Non-Regression Tests
Verifies that CalibratedDomainPredictor and extract_domain_features satisfy:
- Exact probabilistic calibration in [0.0, 1.0]
- Correct 16-feature schema contract compatible with numeric_url_frame
- Pure-CPU single-item latency under 2.0 ms
- Seamless integration with services/scan_engine.py
"""
import time
import numpy as np
import pytest
import services.scan_engine as scan_engine
from services.security_pipeline import numeric_url_frame, domain_model_input
from utils.domain_features import (
    extract_domain_features,
    extract_domain_features_single,
    extract_domain_features_batch,
    DOMAIN_FEATURE_NAMES
)


def test_domain_feature_extraction_contract():
    """Verify 16 domain features are extracted correctly with proper types."""
    domain = "login.apple-account-update.xyz"
    d = extract_domain_features(domain)
    assert isinstance(d, dict)
    assert len(d) == 16
    for name in DOMAIN_FEATURE_NAMES:
        assert name in d, f"Missing domain feature: {name}"
        assert isinstance(d[name], (int, float, np.floating, np.integer))

    arr = extract_domain_features_single(domain)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 16)
    assert arr.dtype == np.float32

    batch = extract_domain_features_batch([domain, "google.com"])
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (2, 16)
    assert batch.dtype == np.float32


def test_calibrated_domain_predictor_inference():
    """Verify calibrated domain model inference behavior, schema, and probabilities."""
    scan_engine.load_ml_model()
    assert scan_engine.DOMAIN_MODEL is not None, "DOMAIN_MODEL should be loaded"
    assert hasattr(scan_engine.DOMAIN_MODEL, "feature_names_in_")
    assert len(scan_engine.DOMAIN_MODEL.feature_names_in_) == 16

    # Test DataFrame input via numeric_url_frame
    feat_clean = scan_engine.extract_domain_features("google.com")
    df_clean = numeric_url_frame(scan_engine.DOMAIN_MODEL, [feat_clean])
    probs_clean = scan_engine.DOMAIN_MODEL.predict_proba(df_clean)
    assert probs_clean.shape == (1, 2)
    assert np.all((probs_clean >= 0.0) & (probs_clean <= 1.0))
    assert np.isclose(probs_clean.sum(axis=1)[0], 1.0)
    p_clean = float(probs_clean[0, 1])
    assert p_clean < 0.50, f"Expected legitimate domain to have < 50% risk, got {p_clean}"

    # Test high-risk DGA / abuse TLD domain
    feat_phish = scan_engine.extract_domain_features("hgnkljbn.cf")
    df_phish = numeric_url_frame(scan_engine.DOMAIN_MODEL, [feat_phish])
    probs_phish = scan_engine.DOMAIN_MODEL.predict_proba(df_phish)
    p_phish = float(probs_phish[0, 1])
    assert p_phish > 0.70, f"Expected high-risk DGA/abuse domain to have > 70% risk, got {p_phish}"

    # Test direct domain string prediction
    probs_direct = scan_engine.DOMAIN_MODEL.predict_proba(["microsoft.com", "hgnkljbn.cf"])
    assert probs_direct.shape == (2, 2)
    assert probs_direct[0, 1] < 0.50
    assert probs_direct[1, 1] > 0.70


def test_domain_single_item_latency_budget():
    """Verify single-domain CPU inference executes well within 2.0 ms budget."""
    scan_engine.load_ml_model()
    assert scan_engine.DOMAIN_MODEL is not None
    domain = "microsoft.com"

    # Warmup
    _ = scan_engine.DOMAIN_MODEL.predict_proba([domain])

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = scan_engine.DOMAIN_MODEL.predict_proba([domain])
        latencies.append((time.perf_counter() - t0) * 1000)

    p95 = float(np.percentile(latencies, 95))
    assert p95 <= 2.0, f"P95 latency {p95:.2f}ms exceeds the 2.0ms gate budget"


def test_domain_operational_threshold_and_synergy_contract():
    """Verify calibrated operational threshold (tau=0.70) protects clean domains and flags DGA abuse."""
    import json
    import os
    scan_engine.load_ml_model()
    assert scan_engine.DOMAIN_MODEL is not None
    assert hasattr(scan_engine.DOMAIN_MODEL, "OPERATIONAL_THRESHOLD")
    assert scan_engine.DOMAIN_MODEL.OPERATIONAL_THRESHOLD == 0.70

    # Clean enterprise domains predict 0 (Clean) under calibrated operational threshold
    clean_domains = ["microsoft.com", "google.com", "apple.com"]
    clean_preds = scan_engine.DOMAIN_MODEL.predict(clean_domains)
    assert np.all(clean_preds == 0), f"Expected clean domains to be labeled 0, got {clean_preds}"

    # High-risk DGA registrar abuse predicts 1 (Phishing)
    dga_domains = ["hgnkljbn.cf"]
    dga_preds = scan_engine.DOMAIN_MODEL.predict(dga_domains)
    assert np.all(dga_preds == 1), f"Expected DGA abuse domain to be labeled 1, got {dga_preds}"

    # Metadata reflects calibrated operational contract
    from pathlib import Path
    meta_path = Path(__file__).resolve().parents[1] / "models" / "domain_phase3_metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["thresholds"]["tau_operational"] == 0.70
    assert meta["test_metrics"]["fpr"] <= 0.05, "Operational FPR must be <= 5%"
