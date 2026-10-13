"""Regression checks for evidence shown in email scan results."""

import sys
import types
from email.message import EmailMessage

from services import scan_engine
from services.origintrace_pillars import InfrastructureLineage, ProvenanceMatrix
from utils.email_parser import parse_eml_content, parse_pasted_email


def test_missing_authentication_is_not_reported_as_a_pass_or_failure():
    auth = scan_engine.score_headers("")
    assert auth["available"] is False
    assert auth["all_passed"] is False
    assert auth["fail_count"] == 0

    paragraph = scan_engine.build_ai_paragraph(
        85,
        {"has_urgency_language": True},
        {"flagged_urls": [], "urls_found": 0},
        auth,
    )
    assert "authentication" not in paragraph.lower()

    explanation = scan_engine.build_explanation(
        85, {}, {"flagged_urls": [], "urls_found": 0}, auth
    )
    assert not any("authentication checks" in signal.lower() for signal in explanation["positive_signals"])


def test_reported_passes_are_retained_but_not_verified():
    partial = scan_engine.score_headers("Authentication-Results: mx; spf=pass")
    assert partial["available"] is False
    assert partial["reported_available"] is True
    assert partial["all_passed"] is False
    complete = scan_engine.score_headers("Authentication-Results: mx; spf=pass; dkim=pass; dmarc=pass")
    assert complete["all_passed"] is False
    assert len(complete["reported_results"]) == 3
    assert complete["verified"] is False


def test_clean_url_is_not_counted_as_suspicious(monkeypatch):
    monkeypatch.setattr(scan_engine, "MODEL_LOADED", True)
    monkeypatch.setattr(scan_engine, "URL_MODEL", None)
    monkeypatch.setattr(scan_engine, "DOMAIN_MODEL", None)
    monkeypatch.setattr("utils.threat_intel.check_virustotal", lambda url: {"available": False})
    result = scan_engine.score_urls("See https://example.com/help for details")
    assert result["urls_analyzed"] == 1
    assert len(result["analyzed_urls"]) == 1
    assert result["flagged_urls"] == []

    suspicious = scan_engine.score_urls("Open http://paypal-login-security.xyz/verify")
    assert suspicious["urls_analyzed"] == 1
    # A configured hostname/TLD cue without model weights is not a fabricated
    # phishing prediction. Missing inference remains explicit for engine review.
    assert suspicious["analyzed_urls"][0]["model_probability"] is None
    assert "URL ML unavailable" in suspicious["inference_errors"]
    assert not suspicious["analyzed_urls"][0]["threat_probability_validated"]


def test_url_evidence_is_not_discarded_without_mail_headers():
    result = InfrastructureLineage.analyze("", ["https://example.com/login"], url_risk_score=80)
    assert result["available"] is True
    assert result["score"] >= 80
    missing = InfrastructureLineage.analyze("", [])
    assert missing["available"] is False
    assert missing["confidence"] == 0


def test_missing_sender_evidence_does_not_claim_authentication():
    result = ProvenanceMatrix.analyze("", "", urls=[])
    assert result["available"] is False
    assert result["confidence"] == 0
    assert "aligned" not in " ".join(result["signals"]).lower()


def test_reported_auth_pass_does_not_cancel_lookalike_risk():
    result = ProvenanceMatrix.analyze(
        "PayPal Support <alerts@paypal-security.example>",
        "Authentication-Results: mx; spf=pass; dkim=pass; dmarc=pass",
        spf="PASS",
        dkim="PASS",
        dmarc="PASS",
    )
    assert result["score"] >= 45
    assert result["crypto_auth"]["verified"] is False
    assert result["crypto_auth"]["spf"] == "UNKNOWN"


def test_multipart_email_prefers_plain_body_and_keeps_auth_headers():
    message = EmailMessage()
    message["From"] = "Example <sender@example.com>"
    message["Subject"] = "Account notice"
    message["Authentication-Results"] = "mx.example.com; spf=pass; dkim=pass; dmarc=pass"
    message.add_alternative("<p>HTML body</p>", subtype="html")
    message.add_alternative("Plain body", subtype="plain")

    parsed = parse_eml_content(message.as_bytes())
    assert parsed["body"] == "Plain body"
    assert "Authentication-Results:" in parsed["headers"]


def test_pasted_full_message_uses_header_block_but_plain_body_does_not():
    raw = "From: sender@example.com\nSubject: Notice\nAuthentication-Results: mx; spf=fail\n\nPlease review this message."
    parsed = parse_pasted_email(raw)
    assert parsed is not None
    assert parsed["body"] == "Please review this message."
    assert "spf=fail" in parsed["headers"]
    assert parse_pasted_email("Dear customer, please review this message.") is None


def test_complete_scan_does_not_invent_authentication_failure(monkeypatch):
    monkeypatch.setattr("data.store.is_blocked", lambda session, sender: False)
    monkeypatch.setattr("data.store.add_scan", lambda *args, **kwargs: None)
    monkeypatch.setattr(scan_engine, "score_content", lambda content: {
        "score": 75,
        "has_urgency_language": True,
        "has_impersonation": True,
        "suspicious_phrases": ["verify account"],
        "keyword_hits": [],
        "is_english": True,
        "language": "English",
        "lang_code": "en",
    })
    monkeypatch.setattr(scan_engine, "score_urls", lambda content, *, additional_urls=None: {
        "score": 80,
        "urls_found": 1,
        "urls_analyzed": 1,
        "flagged_urls": [{"url": "https://paypal-security.example/login", "score": 80, "flags": ["Lookalike domain"]}],
        "has_shortener": False,
    })

    vector = types.ModuleType("services.vector_store")
    vector.search_similar_campaigns = lambda content: {"has_match": False, "match_percentage": 0}
    vector.index_phishing_email = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "services.vector_store", vector)
    bec = types.ModuleType("services.bec_detector")
    bec.predict_bec_intent = lambda *args, **kwargs: {"is_bec": False, "bec_score": 0, "intent_category": "Clean"}
    monkeypatch.setitem(sys.modules, "services.bec_detector", bec)

    result = scan_engine.scan_email(
        "Verify your account at https://paypal-security.example/login",
        session=object(),
        sender="PayPal Support <alerts@paypal-security.example>",
        headers="",
    )
    assert result["header_authentication"]["available"] is False
    assert result["pillars"]["pillar1_provenance"]["score"] >= 45
    assert result["pillars"]["pillar2_infrastructure"]["score"] >= 80
    # This fixture supplies heuristic scores but no model probability. Those
    # scores request review; they must not fabricate a calibrated quarantine.
    assert result["security_action"] == "WARNING_BANNER"
    assert result["status"] == "SUSPICIOUS"
    assert result["pillars"]["pillar4_fusion"]["threat_probability"] is None
    assert "failed authentication" not in result["ai_explanation"]["summary"].lower()
    assert result["epistemic_uncertainty"] is None
