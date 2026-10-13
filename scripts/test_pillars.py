"""
Test Suite for OriginTrace 4-Pillars Security Architecture
Pillar 1: Provenance Matrix
Pillar 2: Infrastructure Lineage
Pillar 3: Semantic Deception
Pillar 4: Evidential Fusion
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.origintrace_pillars import (
    ProvenanceMatrix,
    InfrastructureLineage,
    SemanticDeception,
    EvidentialFusion
)

def run_tests():
    print("=" * 60)
    print("ORIGINTRACE 4-PILLARS INTEGRATION TEST")
    print("=" * 60)

    # ── Test 1: Provenance Matrix (Display Name Spoofing + Failed DKIM/SPF) ──
    print("\n--- Pillar 1: Provenance Matrix ---")
    p1 = ProvenanceMatrix.analyze(
        sender_header='PayPal Support <attacker@gmail.com>',
        raw_headers='spf=fail; dkim=fail; dmarc=fail',
        spf='FAIL',
        dkim='FAIL'
    )
    print(f"P1 Threat Score: {p1['score']}/100")
    print(f"Display Name Spoofed: {p1['display_name_spoofed']}")
    print(f"Signals: {p1['signals']}")
    assert p1['score'] >= 45, "Pillar 1 score should reflect display name difference + unverified authentication"

    # ── Test 1b: Homoglyph Lookalike Domain (micros0ft.com) ───────────────────
    is_lookalike, target, reason = ProvenanceMatrix.check_lookalike_domain('micros0ft.com')
    print(f"Lookalike check 'micros0ft.com': {is_lookalike} -> target: {target} ({reason})")
    assert is_lookalike is True

    # ── Test 2: Infrastructure Lineage (Disposable VPS Origin) ───────────────
    print("\n--- Pillar 2: Infrastructure Lineage ---")
    raw_hops = """
Received: from mail.contabo.vps.net ([185.220.101.5]) by mx.google.com with ESMTPS; Fri, 25 Sep 2026 10:00:00 -0700
Received: from dynamic-dsl-relay.pool.telecom.net ([192.168.1.1]) by mail.contabo.vps.net with ESMTP; Fri, 25 Sep 2026 09:59:00 -0700
    """
    p2 = InfrastructureLineage.analyze(
        headers_text=raw_hops,
        urls=["http://malicious-login.xyz/verify"]
    )
    print(f"P2 Threat Score: {p2['score']}/100")
    print(f"Hops Count: {p2['hops_count']}")
    print(f"Signals: {p2['signals']}")
    assert p2['score'] >= 50

    # ── Test 3: Semantic Deception (CEO Wire Fraud + Quishing) ───────────────
    print("\n--- Pillar 3: Semantic Deception ---")
    p3 = SemanticDeception.analyze(
        bec_result={'is_bec': True, 'bec_score': 85, 'intent_category': 'CEO Wire Fraud'},
        content_result={'score': 75, 'has_urgency_language': True, 'is_english': True},
        vector_result={'has_match': True, 'match_percentage': 82, 'matched_title': 'Executive Wire Fraud'},
        vision_result={'has_quishing': False}
    )
    print(f"P3 Threat Score: {p3['score']}/100")
    print(f"Intent Category: {p3['intent_category']}")
    print(f"Signals: {p3['signals']}")
    assert p3['score'] >= 80

    # ── Test 4: Evidential Fusion (Mathematical Synthesis) ───────────────────
    print("\n--- Pillar 4: Evidential Fusion ---")
    p4 = EvidentialFusion.synthesize(p1, p2, p3)
    print(f"Synthesized Final Threat Score: {p4['final_score']}/100")
    print(f"Belief Phish Mass: {p4['belief_phish']}")
    print(f"Belief Clean Mass: {p4['belief_clean']}")
    print(f"Uncertainty Theta: {p4['uncertainty']}")
    print(f"Modality Contributions: {p4['modality_contributions']}")
    assert p4['final_score'] >= 80

    # ── Test 5: Missing Modalities Handling (e.g. No Headers, No URLs) ───────
    print("\n--- Test 5: Missing Modality Resilience (Vacuous Distribution) ---")
    missing_p1 = {"score": 0.0, "threat_prob": 0.0, "confidence": 0.10}  # No headers
    missing_p2 = {"score": 0.0, "threat_prob": 0.0, "confidence": 0.10}  # No hops
    active_p3 = {"score": 90.0, "threat_prob": 0.90, "confidence": 0.90}  # Strong BEC intent
    p4_missing = EvidentialFusion.synthesize(missing_p1, missing_p2, active_p3)
    print(f"Score when headers missing but strong BEC: {p4_missing['final_score']}/100 (Uncertainty: {p4_missing['uncertainty']})")
    assert p4_missing['final_score'] >= 70, "Missing headers should NOT dilute high BEC score to 0!"

    print("\n" + "=" * 60)
    print("ALL 4 PILLARS PASSED VERIFICATION WITH 100% ACCURACY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
