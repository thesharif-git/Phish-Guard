"""
Verification script for PhishGuard Production Engine
Tests both Email and URL detection against real-world benign, authority, and phishing samples.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import services.scan_engine as scan_engine
from utils.url_scanner import scan_url, is_safe_authority_domain, run_url_heuristics
import pandas as pd

def run_tests():
    print("=" * 80, flush=True)
    print("PHISHGUARD PRODUCTION ENGINE END-TO-END VALIDATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Test Authority Domain Recognition
    print("\n[TEST 1] Authority Domain Anchor Lookup:", flush=True)
    test_domains = [
        ("uni-mainz.de", True),
        ("www.uni-mainz.de", True),
        ("google.com", True),
        ("github.com", True),
        ("harvard.edu", True),
        ("irs.gov", True),
        ("paypal-update-account.xyz", False),
        ("login-secure-verification.cc", False),
    ]
    for dom, expected in test_domains:
        res = is_safe_authority_domain(dom)
        status = "PASS" if res == expected else "FAIL"
        print(f"  {dom:<35} -> Authority={res!s:<5} (Expected {expected!s:<5}) [{status}]", flush=True)
        assert res == expected, f"Failed authority check on {dom}"

    # 2. Test URL ML Features & Heuristics Directly
    print("\n[TEST 2] URL ML Models & Heuristics Scoring:", flush=True)
    scan_engine.load_ml_model()
    test_cases = [
        # (URL, Expected Label)
        ("https://www.uni-mainz.de/", "CLEAN"),
        ("https://github.com/thesharif-git", "CLEAN"),
        ("https://paypal.com/signin", "CLEAN"),
        ("http://paypa1-account-security-update.xyz/login?id=99", "PHISHING"),
        ("http://185.220.101.5/bank-verification/login.php", "PHISHING"),
        ("http://secure-login-portal.cc/verify", "PHISHING")
    ]
    for url, exp_type in test_cases:
        h = run_url_heuristics(url)
        u_feat = pd.DataFrame([scan_engine.extract_url_features(url)])
        d_feat = pd.DataFrame([scan_engine.extract_domain_features(url)])
        u_prob = float(scan_engine.URL_MODEL.predict_proba(u_feat)[0][1]) if scan_engine.URL_MODEL else 0.0
        d_prob = float(scan_engine.DOMAIN_MODEL.predict_proba(d_feat)[0][1]) if scan_engine.DOMAIN_MODEL else 0.0
        auth = is_safe_authority_domain(h.get("domain", ""))
        print(f"  URL: {url}", flush=True)
        print(f"       Authority={auth} | HeuristicScore={h['heuristic_score']} | URL_ML={u_prob*100:.1f}% | Domain_ML={d_prob*100:.1f}%", flush=True)
        if exp_type == "CLEAN":
            assert auth or (u_prob < 0.5 and d_prob < 0.5), f"False positive on clean URL: {url}"
        else:
            assert h['heuristic_score'] >= 25 or u_prob >= 0.5 or d_prob >= 0.5, f"False negative on phishing URL: {url}"

    # 3. Test Email Scoring
    print("\n[TEST 3] Email Content Scoring:", flush=True)
    benign_email = "Hi team, please find attached the quarterly business review draft. Let me know if you have any feedback before tomorrow's meeting. Thanks, Alice."
    phish_email = "URGENT ACTION REQUIRED: Your PayPal account has been temporarily restricted due to unauthorized login attempts. Please click the link below immediately to verify your identity: http://paypa1-verify-account.xyz/login. If you do not respond within 24 hours, your account will be permanently terminated."

    res_benign = scan_engine.score_content(benign_email)
    res_phish = scan_engine.score_content(phish_email)

    print(f"  Benign Email Score: {res_benign['score']} (ML Prob: {res_benign.get('ml_probability')})", flush=True)
    print(f"  Phish  Email Score: {res_phish['score']} (ML Prob: {res_phish.get('ml_probability')})", flush=True)

    assert res_benign['score'] <= 30, f"Benign email scored too high: {res_benign['score']}"
    assert res_phish['score'] >= 50, f"Phishing email scored too low: {res_phish['score']}"

    # 4. Test End-to-End scan_email with SQLite session
    print("\n[TEST 4] End-to-End Composite Email Scan (Content + URL + Evidential Fusion):", flush=True)
    from data.database import engine
    from sqlmodel import Session
    with Session(engine) as session:
        scan_benign = scan_engine.scan_email(benign_email, session=session, sender="alice@mycompany.com", subject="QBR Review")
        scan_phish = scan_engine.scan_email(phish_email, session=session, sender="service@paypa1-security.xyz", subject="URGENT: Account Restricted")

    print(f"  Composite Benign Email Verdict: {scan_benign['status']} (Final Score: {scan_benign['final_score']}, Risk: {scan_benign['risk_level']})", flush=True)
    print(f"  Composite Phish  Email Verdict: {scan_phish['status']} (Final Score: {scan_phish['final_score']}, Risk: {scan_phish['risk_level']})", flush=True)

    assert scan_benign['status'] == 'CLEAN', f"Expected CLEAN for benign email, got {scan_benign['status']}"
    assert scan_phish['status'] == 'PHISHING', f"Expected PHISHING for phishing email, got {scan_phish['status']}"

    # 5. Test Sneaky URL on Authority Domain (No Blind Spot in Scan Engine and Standalone Scanner)
    print("\n[TEST 5] Sneaky Credential Path on Authority Domain (Scan Engine & Standalone Scanner):", flush=True)
    test_ugc_url = "https://docs.google.com/forms/login?email=victim@target.com"
    res_sneaky = scan_engine.score_urls(f"Please submit credentials: {test_ugc_url}")
    sneaky_item = res_sneaky["analyzed_urls"][0]
    print(f"  Scan Engine Sneaky Google Form Score: {sneaky_item['score']} | Flags: {sneaky_item['flags']}", flush=True)
    assert sneaky_item['score'] >= 45, f"Authority blind spot failure: sneaky link scored {sneaky_item['score']}"

    standalone_res = scan_url(test_ugc_url)
    print(f"  Standalone scan_url Combined Score: {standalone_res['combined_score']} | Risk: {standalone_res['risk_level']}", flush=True)
    assert standalone_res['combined_score'] >= 40, f"Standalone URL scanner cleared UGC phishing: {standalone_res['combined_score']}"

    # 6. Test Text-Only Social Engineering & BEC (Linkless)
    print("\n[TEST 6] Text-Only Financial & Identity Social Engineering (Linkless):", flush=True)
    bec_email = "URGENT: Please process an immediate wire transfer of $75,000 to our vendor bank account today. I am travelling and cannot take calls."
    with Session(engine) as session:
        scan_bec = scan_engine.scan_email(bec_email, session=session, sender="ceo@company-exec.com", subject="Urgent wire request")
    print(f"  BEC Wire Fraud Verdict: {scan_bec['status']} (Final Score: {scan_bec['final_score']}, Risk: {scan_bec['risk_level']})", flush=True)
    assert scan_bec['final_score'] >= 30, f"Financial BEC falsely suppressed: {scan_bec['final_score']}"

    # Non-financial credential / 2FA / identity solicitation (linkless)
    identity_email = "URGENT: Your employee workstation access has been locked. Reply immediately with your current 2-factor SMS verification code and full Social Security Number to avoid immediate termination."
    with Session(engine) as session:
        scan_identity = scan_engine.scan_email(identity_email, session=session, sender="security-admin@it-helpdesk.cc", subject="CRITICAL: Account Termination Notice")
    print(f"  Non-Financial Identity Lure Verdict: {scan_identity['status']} (Final Score: {scan_identity['final_score']}, Risk: {scan_identity['risk_level']})", flush=True)
    assert scan_identity['final_score'] >= 40, f"Identity/2FA solicitation falsely capped or missed: {scan_identity['final_score']}"

    print("\n" + "=" * 80, flush=True)
    print("ALL 6 TESTS PASSED! PRODUCTION ENGINE VERIFIED HEALTHY, ROBUST & CALIBRATED.", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_tests()

