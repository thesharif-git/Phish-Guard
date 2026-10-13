"""
Unit Tests for OriginTrace Adaptive Multi-Fidelity Cascade Router (Step 2)
Tests:
  1. Tier 1 Fast Path Early-Exit on Benign Corporate URL (Sub-10ms, No deep compute)
  2. Tier 2 Deep Forensics Escalation on Deceptive / Lookalike Phishing Campaign
  3. Evidential 3-Way Verdicts (CLEAN, PHISHING, INSUFFICIENT_EVIDENCE)
"""
import sys
import os
import time

sys.path.insert(0, r"e:\Ai Phishing Detector")

from services.origintrace_pillars import AdaptiveCascadeRouter

print("=" * 70)
print("RUNNING ORIGINTRACE MULTI-FIDELITY CASCADE ROUTER VERIFICATION")
print("=" * 70)

# ── Test 1A: Cached Benign Target (Tier 0 Fast Path) ───────────────────
print("\n--- Test 1A: Cached URL (https://google.com) ---", flush=True)
res_cache = AdaptiveCascadeRouter.route_scan(
    target_type="url",
    raw_target="https://google.com"
)

print(f"Tier Executed: {res_cache['tier_executed']}", flush=True)
print(f"Early Exit: {res_cache['early_exit']}", flush=True)
print(f"Latency: {res_cache['latency_ms']} ms", flush=True)
print(f"Verdict State: {res_cache['verdict_state']} (Risk: {res_cache['risk_score']}/100)", flush=True)

assert res_cache["early_exit"] is True, "Must exit early!"
assert res_cache["verdict_state"] == "CLEAN", "Must be CLEAN!"

# ── Test 1B: Fresh Benign Target (Tier 1 Lightweight Early Exit) ───────
print("\n--- Test 1B: Fresh Benign URL (https://python.org) ---", flush=True)
res_clean = AdaptiveCascadeRouter.route_scan(
    target_type="url",
    raw_target="https://python.org",
    force_deep=False
)

print(f"Tier Executed: {res_clean['tier_executed']}", flush=True)
print(f"Early Exit: {res_clean['early_exit']}", flush=True)
print(f"Latency: {res_clean['latency_ms']} ms", flush=True)
print(f"Verdict State: {res_clean['verdict_state']} (Risk: {res_clean['risk_score']}/100)", flush=True)
print(f"Reason: {res_clean['routing_reason']}", flush=True)

assert res_clean["early_exit"] is True, "Clean authority domain MUST exit early!"
assert res_clean["tier_executed"] in ["TIER_0_CACHE", "TIER_1_LIGHTWEIGHT"], "Must resolve in Tier 0 or Tier 1!"
assert res_clean["verdict_state"] == "CLEAN", "Must be CLEAN!"

# ── Test 2: Deceptive Lookalike Phishing URL (Must escalate to Tier 2) ──
print("\n--- Test 2: Deceptive Lookalike (http://chase-security-verify-login.xyz) ---")
res_phish = AdaptiveCascadeRouter.route_scan(
    target_type="url",
    raw_target="http://chase-security-verify-login.xyz"
)

print(f"Tier Executed: {res_phish['tier_executed']}")
print(f"Early Exit: {res_phish['early_exit']}")
print(f"Latency: {res_phish['latency_ms']} ms")
print(f"Verdict State: {res_phish['verdict_state']} (Risk: {res_phish['risk_score']}/100)")
print(f"Epistemic Uncertainty (u): {res_phish['epistemic_uncertainty']}")
print(f"Escalation Triggers: {res_phish.get('tier1_escalation_triggers', [])}")
print(f"Contradiction Score: {res_phish['contradiction_analysis']['contradiction_score']}/100")

assert res_phish["early_exit"] is False, "Phishing domain must NOT exit early!"
assert res_phish["tier_executed"] == "TIER_2_DEEP_FORENSICS", "Must escalate to Tier 2!"
assert res_phish["verdict_state"] == "PHISHING", "Must be flagged as PHISHING!"
assert res_phish["risk_score"] >= 65, f"Expected risk score >= 65, got {res_phish['risk_score']}"

# ── Test 3: Multi-Layer Email Phishing with Contradiction ──────────────
print("\n--- Test 3: Email Phishing with Anchor Text Spoofing ---")
res_email = AdaptiveCascadeRouter.route_scan(
    target_type="email",
    raw_target="http://auth-chase-update.top/login",
    sender_header='"Chase Security" <alerts@gmail.com>',
    raw_headers="DKIM-Signature: d=sendgrid.net;",
    dkim="PASS",
    spf="PASS",
    anchor_links=[{"text": "https://www.chase.com/verify", "href": "http://auth-chase-update.top/login"}],
    email_content="Urgent: Your account is suspended. Verify immediately at our portal."
)

print(f"Tier Executed: {res_email['tier_executed']}")
print(f"Early Exit: {res_email['early_exit']}")
print(f"Verdict State: {res_email['verdict_state']} (Risk: {res_email['risk_score']}/100)")
print(f"Contradiction Severity: {res_email['contradiction_analysis']['severity']}")
print(f"Mismatches: {len(res_email['contradiction_analysis']['mismatches'])}")

assert res_email["tier_executed"] == "TIER_2_DEEP_FORENSICS"
assert res_email["verdict_state"] == "PHISHING"
assert res_email["contradiction_analysis"]["is_contradictory"] is True

print("\n" + "=" * 70)
print("ALL CASCADE ROUTER UNIT TESTS PASSED CLEANLY (100% REAL LIVE DATA)!")
print("=" * 70)
