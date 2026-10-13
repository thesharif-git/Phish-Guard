import pytest
from services.origintrace_pillars import (
    ProvenanceMatrix,
    InfrastructureLineage,
    SemanticDeception,
    EvidentialFusion,
    CampaignAttributor
)
from utils.threat_intel import (
    analyze_tls_cert_lineage,
    calculate_dns_ephemerality,
    check_asn_brand_incongruity
)

def test_provenance_matrix_web_clean():
    res = ProvenanceMatrix.analyze_web("google.com", "https://google.com")
    assert res["score"] == 0.0
    assert not res["lookalike_domain"]
    assert not res["tls_lineage_mismatch"]

def test_provenance_matrix_web_lookalike_and_lineage_mismatch():
    mock_tls = {
        "available": True,
        "lineage_mismatch": True,
        "mismatch_reason": "TLS Certificate Lineage Mismatch: Domain targets brand 'paypal' with free DV cert",
        "cert_fingerprint": "a" * 64,
        "issuer": "Let's Encrypt",
        "cert_age_days": 2
    }
    res = ProvenanceMatrix.analyze_web("paypal-security-update.com", "https://paypal-security-update.com", tls_intel=mock_tls)
    assert res["score"] > 60
    assert res["lookalike_domain"]
    assert res["tls_lineage_mismatch"]

def test_infrastructure_lineage_web_abci():
    mock_asn = {
        "available": True,
        "asn": "AS51167",
        "isp": "Contabo GmbH",
        "is_hosting": True,
        "is_incongruent": True,
        "signals": ["Autonomous System Brand Incongruity: Domain targets 'paypal' on Contabo VPS"]
    }
    mock_dns = {
        "available": True,
        "ephemerality_index": 0.45,
        "min_ttl": 60,
        "signals": ["Fast-Flux DNS Detected: Host TTL is ultra-low (60s)"]
    }
    res = InfrastructureLineage.analyze_web("paypal-login.xyz", dns_ephemerality=mock_dns, asn_intel=mock_asn)
    assert res["score"] >= 75
    assert any("Autonomous System" in s for s in res["signals"])
    assert any("Fast-Flux" in s for s in res["signals"])

def test_semantic_deception_web():
    mock_html = {
        "success": True,
        "brand_mismatch": True,
        "has_password_field": True,
        "obfuscated_js": True
    }
    res = SemanticDeception.analyze_web(html_analysis=mock_html, url="https://verify-account.com/login", domain="verify-account.com")
    assert res["score"] >= 80
    assert res["has_credential_form"]
    assert res["brand_deception"]

def test_evidential_fusion_web():
    p1 = {"score": 85, "threat_prob": 0.85, "confidence": 0.90, "tls_lineage_mismatch": True}
    p2 = {"score": 80, "threat_prob": 0.80, "confidence": 0.85}
    p3 = {"score": 75, "threat_prob": 0.75, "confidence": 0.80, "has_credential_form": True}
    fusion = EvidentialFusion.synthesize(p1, p2, p3, context="web")
    assert fusion["final_score"] >= 80
    assert fusion["threat_probability"] > 0.80
    assert fusion["belief_phish"] > fusion["belief_clean"]

def test_campaign_attribution():
    attr = CampaignAttributor.attribute(
        domain="paypal-verify-auth.xyz",
        brand="paypal",
        asn="AS51167 Contabo",
        isp="Contabo",
        tls_fingerprint="abcdef123456",
        tls_issuer="Let's Encrypt",
        is_threat=True
    )
    assert attr["is_attributed"]
    assert "PayPal" in attr["campaign_name"]
    assert attr["campaign_id"].startswith("CAMP-")

if __name__ == "__main__":
    print("Running OriginTrace Web-Mode Unit Tests...")
    test_provenance_matrix_web_clean()
    print("PASS: test_provenance_matrix_web_clean")
    test_provenance_matrix_web_lookalike_and_lineage_mismatch()
    print("PASS: test_provenance_matrix_web_lookalike_and_lineage_mismatch")
    test_infrastructure_lineage_web_abci()
    print("PASS: test_infrastructure_lineage_web_abci")
    test_semantic_deception_web()
    print("PASS: test_semantic_deception_web")
    test_evidential_fusion_web()
    print("PASS: test_evidential_fusion_web")
    test_campaign_attribution()
    print("PASS: test_campaign_attribution")
    print("\nALL ORIGINTRACE 4-PILLAR & CAMPAIGN TESTS PASSED SUCCESSFULLY!")
