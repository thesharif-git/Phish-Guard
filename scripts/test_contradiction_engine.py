"""
Unit Tests for OriginTrace Cross-Entity Contradiction Matrix (Step 1)
Tests alignment & contradiction detection across all 5 structural layers:
  Layer 1: Display Name vs From Address
  Layer 2: Cryptographic DKIM Alignment
  Layer 3: Anchor Text vs Href Misdirection
  Layer 4: In-Body Link Destination Alignment
  Layer 5: Infrastructure BGP Routing ASN Alignment
"""
import sys
import os

sys.path.insert(0, r"e:\Ai Phishing Detector")

from services.origintrace_pillars import CrossEntityContradictionEngine

print("=" * 65)
print("RUNNING ORIGINTRACE CONTRADICTION MATRIX VERIFICATION")
print("=" * 65)

# Test Case 1: Pure Benign Corporate Communication (100% Aligned)
res_clean = CrossEntityContradictionEngine.analyze(
    sender_header='"Microsoft 365 Security" <security@microsoft.com>',
    raw_headers="DKIM-Signature: v=1; a=rsa-sha256; d=microsoft.com; s=s1024;",
    dkim="PASS",
    spf="PASS",
    urls=["https://login.microsoftonline.com/auth"],
    anchor_links=[{"text": "https://login.microsoftonline.com", "href": "https://login.microsoftonline.com/auth"}],
    destination_domain="login.microsoftonline.com",
    destination_asn="AS8075 Microsoft Corporation",
    destination_isp="Microsoft Azure"
)

print("\n--- Test Case 1: Legitimate Microsoft Email ---")
print("Contradiction Score:", res_clean["contradiction_score"], "/ 100")
print("Severity:", res_clean["severity"])
print("Is Contradictory:", res_clean["is_contradictory"])
print("Mismatches:", len(res_clean["mismatches"]))
print("Summary:", res_clean["alignment_summary"])
assert res_clean["is_contradictory"] is False, "Benign email should NOT be contradictory!"
assert res_clean["contradiction_score"] == 0, "Benign email score should be 0!"

# Test Case 2: Multi-Layer Deceptive Phishing Campaign
# - Display Name: "Chase Bank Alert"
# - From Address: <mailer@gmail.com> (Layer 1 mismatch)
# - DKIM: Signed by <sendgrid.net> (Layer 2 mismatch)
# - Anchor Link: Text says "https://chase.com/verify", Href is "http://chase-auth-login.top/account" (Layer 3 mismatch)
# - Destination ASN: Hosted on Contabo VPS (Layer 5 mismatch)
res_phish = CrossEntityContradictionEngine.analyze(
    sender_header='"Chase Bank Alert" <notification@gmail.com>',
    raw_headers="DKIM-Signature: v=1; a=rsa-sha256; d=sendgrid.net; s=s1;",
    dkim="PASS",
    spf="PASS",
    urls=["http://chase-auth-login.top/account"],
    anchor_links=[{"text": "https://www.chase.com/verify", "href": "http://chase-auth-login.top/account"}],
    destination_domain="chase-auth-login.top",
    destination_asn="AS51167 Contabo GmbH",
    destination_isp="Contabo VPS"
)

print("\n--- Test Case 2: Multi-Layer Chase Phishing Campaign ---")
print("Contradiction Score:", res_phish["contradiction_score"], "/ 100")
print("Severity:", res_phish["severity"])
print("Is Contradictory:", res_phish["is_contradictory"])
print("Mismatches Detected:", len(res_phish["mismatches"]))
for i, m in enumerate(res_phish["mismatches"], 1):
    print(f"  {i}. [{m['layer']}] {m['type']} (Severity: {m['severity']})")
    print(f"     Evidence: {m['evidence']}")

assert res_phish["is_contradictory"] is True, "Phishing email MUST be contradictory!"
assert res_phish["contradiction_score"] >= 85, f"Phishing score should be >= 85, got {res_phish['contradiction_score']}"
assert len(res_phish["mismatches"]) >= 3, "Expected at least 3 layer mismatches!"

# Test Case 3: Anchor Text Misdirection Only (Classic link disguise)
res_link = CrossEntityContradictionEngine.analyze(
    sender_header='"PayPal Support" <support@paypal.com>',
    raw_headers="d=paypal.com",
    dkim="PASS",
    urls=["https://auth-session-recovery.online/login"],
    anchor_links=[{"text": "https://www.paypal.com/signin", "href": "https://auth-session-recovery.online/login"}]
)

print("\n--- Test Case 3: Anchor Text Hyperlink Misdirection ---")
print("Contradiction Score:", res_link["contradiction_score"], "/ 100")
print("Severity:", res_link["severity"])
print("Is Contradictory:", res_link["is_contradictory"])
assert res_link["contradiction_score"] >= 50, "Deceptive anchor link should flag high contradiction!"
assert any(m["type"] == "ANCHOR_TEXT_HYPERLINK_MISDIRECTION" for m in res_link["mismatches"])

print("\n" + "=" * 65)
print("ALL CONTRADICTION ENGINE UNIT TESTS PASSED SUCCESSFULLY!")
print("=" * 65)
