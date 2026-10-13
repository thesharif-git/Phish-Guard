"""Final web result scope: missing evidence is distinct from a negative finding."""
import math

def web_evidence_decision(score, *, url_probability, html, tls, dns, asn, intel, contradiction, ignorance):
    model_complete = url_probability is not None and math.isfinite(float(url_probability)) and 0 <= float(url_probability) <= 1
    checks = {
        "url_model": model_complete, "page": bool(html.get("success")),
        "tls": bool(tls.get("available")), "dns": bool(dns.get("available")),
        "asn": bool(asn.get("available")),
        **{name: bool((intel.get(name) or {}).get("available")) for name in ("virustotal", "phishtank", "urlhaus")},
    }
    gaps = [name for name in ("url_model", "page") if not checks[name]]
    strong_contradiction = contradiction.get("is_contradictory") and contradiction.get("severity") == "CRITICAL"
    if score >= 75 or strong_contradiction:
        status, verdict, risk = "PHISHING", "PHISHING", "critical"
        summary = "High-risk indicators detected. Avoid visiting until reviewed; unavailable checks remain unverified."
    elif score >= 45:
        status, verdict, risk = "SUSPICIOUS", "SUSPICIOUS", "medium"
        summary = "Suspicious indicators detected. Review the evidence before interacting with this URL."
    elif gaps or ignorance >= .35:
        status, verdict, risk = "UNKNOWN", "INSUFFICIENT_EVIDENCE", "unknown"
        summary = "Insufficient evidence to assess this URL. " + ("Required checks unavailable: " + ", ".join(gaps) + "." if gaps else "The completed evidence does not support a clear decision.")
    else:
        status, verdict, risk = "CLEAN", "CLEAN", "safe"
        summary = "No threat indicators found in the completed checks. This is not proof of safety; unavailable supporting checks remain unverified."
    return {"status": status, "verdict_state": verdict, "risk_level": risk, "summary": summary,
            "evidence_gaps": gaps, "check_availability": checks,
            "evidence_incomplete": not all(checks.values()),
            "reputation_checked": any(checks[n] for n in ("virustotal", "phishtank", "urlhaus"))}
