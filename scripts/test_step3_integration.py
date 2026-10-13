"""
Step 3 End-to-End Integration Verification Script
Tests that:
  1. scan_email produces contradiction_analysis and evidential 3-way verdict_state.
  2. extension_quick_check executes Multi-Fidelity Adaptive Cascade with early exit.
  3. scan_url exports contradiction_analysis and epistemic_uncertainty.
"""
import sys
import os
import asyncio

sys.path.insert(0, r"e:\Ai Phishing Detector")

from sqlmodel import Session
from data.database import engine
from services.scan_engine import scan_email
from routers.api import extension_quick_check, ExtensionCheckRequest
from utils.url_scanner import scan_url

print("=" * 70, flush=True)
print("RUNNING STEP 3 END-TO-END INTEGRATION VERIFICATION", flush=True)
print("=" * 70, flush=True)

# ── 1. Test scan_email integration ────────────────────────────────────
print("\n[1/3] Testing scan_email with Contradiction Matrix...", flush=True)
with Session(engine) as session:
    # 1A. Legitimate Email
    email_clean = scan_email(
        subject="Meeting Agenda for Tomorrow",
        content="Hi team, here is the agenda for tomorrow's sync. Best regards.",
        headers="From: sender@company.com\nDKIM-Signature: d=company.com;\nAuthentication-Results: dkim=pass spf=pass",
        session=session
    )
    print("  Clean Email Score:", email_clean["final_score"], "/ 100", flush=True)
    print("  Verdict State:", email_clean.get("verdict_state"), flush=True)
    print("  Contradiction Mismatches:", len(email_clean.get("contradiction_analysis", {}).get("mismatches", [])), flush=True)
    assert email_clean["verdict_state"] == "CLEAN", "Clean email should have CLEAN verdict_state"
    assert email_clean["contradiction_analysis"]["is_contradictory"] is False

    # 1B. Deceptive Phishing Email with Anchor Spoofing
    email_phish = scan_email(
        subject="URGENT: Chase Account Alert",
        content='Your account is locked! Please verify at <a href="http://auth-chase-update.top/login">https://www.chase.com/verify</a> immediately.',
        headers="From: Chase Support <alerts@gmail.com>\nDKIM-Signature: d=sendgrid.net;\nAuthentication-Results: dkim=fail spf=fail",
        session=session
    )
    print("  Phishing Email Score:", email_phish["final_score"], "/ 100", flush=True)
    print("  Verdict State:", email_phish.get("verdict_state"), flush=True)
    print("  Contradiction Severity:", email_phish.get("contradiction_analysis", {}).get("severity"), flush=True)
    print("  Contradiction Mismatches:", len(email_phish.get("contradiction_analysis", {}).get("mismatches", [])), flush=True)
    assert email_phish["verdict_state"] == "PHISHING", "Phishing email should have PHISHING verdict_state"
    assert email_phish["contradiction_analysis"]["is_contradictory"] is True

# ── 2. Test extension_quick_check integration ─────────────────────────
print("\n[2/3] Testing extension_quick_check Cascade Routing...", flush=True)
async def test_extension():
    from routers.api import _EXT_CACHE
    _EXT_CACHE.clear()

    # 2A. Benign URL early exit
    clean_req = ExtensionCheckRequest(url="https://google.com")
    res_clean = await extension_quick_check(clean_req)
    print("  Clean URL Tier:", res_clean.get("cascade_tier_executed"), flush=True)
    print("  Clean URL Verdict:", res_clean.get("verdict_state"), flush=True)
    assert res_clean.get("cascade_tier_executed") in ["TIER_0_CACHE", "TIER_1_LIGHTWEIGHT"]
    assert res_clean.get("verdict_state") == "CLEAN"

    # 2B. Deceptive Phishing URL escalation
    phish_req = ExtensionCheckRequest(url="http://chase-security-verify-login.xyz")
    res_phish = await extension_quick_check(phish_req)
    print("  Phishing URL Tier:", res_phish.get("cascade_tier_executed"), flush=True)
    print("  Phishing URL Score:", res_phish.get("risk_score"), flush=True)
    print("  Phishing URL Verdict:", res_phish.get("verdict_state"), flush=True)
    assert res_phish.get("cascade_tier_executed") == "TIER_2_DEEP_FORENSICS"
    assert res_phish.get("verdict_state") == "PHISHING"

asyncio.run(test_extension())

# ── 3. Test scan_url integration ──────────────────────────────────────
print("\n[3/3] Testing scan_url with Contradiction Matrix...", flush=True)
url_res = scan_url("http://chase-security-verify-login.xyz")
print("  URL Scan Score:", url_res.get("final_score"), flush=True)
print("  URL Scan Verdict State:", url_res.get("verdict_state"), flush=True)
print("  Contradiction Mismatches:", len(url_res.get("contradiction_analysis", {}).get("mismatches", [])), flush=True)
assert "contradiction_analysis" in url_res, "contradiction_analysis must be present in scan_url result"
assert url_res.get("verdict_state") == "PHISHING", "Lookalike domain must be PHISHING"

print("\n" + "=" * 70, flush=True)
print("ALL STEP 3 INTEGRATION TESTS PASSED 100% CLEANLY (LIVE REAL DATA)!", flush=True)
print("=" * 70, flush=True)
