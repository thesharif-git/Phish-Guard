"""
OriginTrace 4-Pillar Security Engine
Enterprise Cyber Defense Architecture:
- Pillar 1: Provenance Matrix (Cryptographic identity, display name spoofing, lookalike domains)
- Pillar 2: Infrastructure Lineage (Server hops, disposable VPS, residential relays, newly registered domains)
- Pillar 3: Semantic Deception (Zero-link BEC, CEO wire fraud, gift card scams, QR codes, multilingual lures)
- Email Pillar 4 is integrated separately in scan_engine.py; legacy web fusion below retains configured priors.
"""

import re
import socket
import math
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from email.utils import parseaddr
from email.parser import HeaderParser
from email import policy as email_policy
from urllib.parse import urlparse
from core.logger import logger

# ── Top Protected Enterprise & Financial Brands for Lookalike Analysis ────────
TOP_BRANDS = [
    "paypal", "microsoft", "google", "apple", "amazon", "netflix",
    "chase", "bankofamerica", "wellsfargo", "citibank", "docusign",
    "dropbox", "adobe", "facebook", "instagram", "linkedin", "irs",
    "github", "spotify", "binance", "coinbase"
]

# Common Homoglyph / Levenshtein character replacements
HOMOGLYPH_MAP = {
    '0': 'o', '1': 'l', 'l': 'i', 'vv': 'w', 'rn': 'm', '@': 'a', '$': 's'
}

# Two-part ccTLDs to correctly extract registered domains
CCTLD_DOUBLE = {
    'co.uk', 'com.pk', 'gov.pk', 'edu.pk', 'org.pk', 'net.pk',
    'com.au', 'net.au', 'org.au', 'co.nz', 'co.jp', 'com.br',
    'co.in', 'com.in', 'net.in', 'org.in', 'co.za', 'com.sg'
}

DISPOSABLE_HOSTING_PATTERNS = [
    r'contabo', r'hetzner', r'digitalocean', r'ovh', r'linode',
    r'vultr', r'choopa', r'hostinger', r'scaleway', r'kamatera'
]

RESIDENTIAL_RELAY_PATTERNS = [
    r'\bdsl\b', r'\bdynamic\b', r'\bresidential\b', r'\bpool\b',
    r'\bcable\b', r'\bdhcp\b', r'\bbroadband\b', r'\bhome\b', r'\buser\b'
]


def _identity_hostname(value: str, *, bare: bool = False) -> str:
    """Parse an observed hostname; prose, mailbox userinfo and paths are not hosts."""
    text = (value or '').strip()
    if not text or any(character.isspace() for character in text):
        return ''
    try:
        parsed = urlparse(text if '://' in text else ('https://' + text if bare else text))
        if parsed.scheme not in ('http', 'https') or parsed.username is not None or parsed.password is not None:
            return ''
        host = (parsed.hostname or '').rstrip('.').lower()
        # Also validate a malformed port instead of accepting its hostname.
        parsed.port
        if not host or (bare and '.' not in host):
            return ''
        host = host.encode('idna').decode('ascii')
        if not all(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label) for label in host.split('.')):
            return ''
        return host.removeprefix('www.')
    except (ValueError, UnicodeError):
        return ''


def _mailbox_hostname(address: str) -> str:
    if address.count('@') != 1:
        return ''
    domain = address.rsplit('@', 1)[1]
    if any(character in domain for character in '/?#:'):
        return ''
    return _identity_hostname(domain, bare=True)


def _display_brand(display_name: str) -> Optional[str]:
    """A brand token can be a claim; a substring of a person's name cannot."""
    text = (display_name or '').casefold()
    phrases = {'bankofamerica': 'bank of america', 'wellsfargo': 'wells fargo'}
    for brand in TOP_BRANDS:
        phrase = phrases.get(brand, brand)
        if re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text):
            return brand
    return None


def _email_lookalike_candidate(hostname: str) -> Tuple[bool, Optional[str], str]:
    """Email identity rules share the configured domain relationships, not trust."""
    if any(CrossEntityContradictionEngine._is_official_domain(brand, hostname) for brand in TOP_BRANDS):
        return False, None, 'Configured brand-domain relationship; sender not verified'
    candidate, brand, reason = ProvenanceMatrix.check_lookalike_domain(hostname)
    if candidate and reason.startswith('Brand hijacking compound domain'):
        _, root, _, _ = ProvenanceMatrix._parse_domain_parts(hostname)
        if not re.search(r'(?<![a-z0-9])' + re.escape(brand) + r'(?![a-z0-9])', root):
            return False, None, 'No bounded compound brand token'
    return candidate, brand, reason


def _reported_dkim_domains(headers: str) -> List[str]:
    """Read d= only from DKIM-Signature fields; a signature is not validation."""
    message = HeaderParser(policy=email_policy.default).parsestr(headers or '')
    domains = []
    for signature in message.get_all('DKIM-Signature', []):
        matches = re.findall(r'(?:^|;)\s*d\s*=\s*([^;\s]+)', str(signature), re.I)
        if len(matches) != 1:
            continue
        if any(character in matches[0] for character in '/?#:@'):
            continue
        domain = _identity_hostname(matches[0], bare=True)
        if domain and domain not in domains:
            domains.append(domain)
    return domains


# ==============================================================================
# PILLAR 1: PROVENANCE MATRIX (Cryptographic Identity & Domain Authenticity)
# ==============================================================================
class ProvenanceMatrix:
    """
    Evaluates cryptographic identity, display name manipulation, and homoglyph lookalikes.
    """

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if s1 == s2:
            return 0
        if len(s1) == 0:
            return len(s2)
        if len(s2) == 0:
            return len(s1)
        v0 = list(range(len(s2) + 1))
        v1 = [0] * (len(s2) + 1)
        for i in range(len(s1)):
            v1[0] = i + 1
            for j in range(len(s2)):
                cost = 0 if s1[i] == s2[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            v0 = v1[:]
        return v0[len(s2)]

    @staticmethod
    def _parse_domain_parts(domain: str) -> Tuple[str, str, str, str]:
        d = domain.lower().split(':')[0].strip().strip('/')
        if d.startswith('www.'):
            d = d[4:]
        parts = d.split('.')
        if len(parts) >= 3 and '.'.join(parts[-2:]) in CCTLD_DOUBLE:
            root_label = parts[-3]
            subdomain = '.'.join(parts[:-3])
            full_root = '.'.join(parts[-3:])
        elif len(parts) >= 2:
            root_label = parts[-2]
            subdomain = '.'.join(parts[:-2])
            full_root = '.'.join(parts[-2:])
        else:
            root_label = d
            subdomain = ''
            full_root = d
        return d, root_label, subdomain, full_root

    @staticmethod
    def check_lookalike_domain(domain: str) -> Tuple[bool, Optional[str], str]:
        if not domain:
            return False, None, "None"

        d, root_label, subdomain, full_root = ProvenanceMatrix._parse_domain_parts(domain)

        for brand in TOP_BRANDS:
            # 1. Authentic root domain (e.g. google.com, google.co.uk, apple.co)
            if root_label == brand:
                # Check if another brand is hijacked in subdomains of this legitimate domain
                for sub_brand in TOP_BRANDS:
                    if sub_brand != brand and sub_brand in subdomain.split('.'):
                        return True, sub_brand, f"Brand '{sub_brand}' spoofed in subdomain of {full_root}"
                return False, None, "Clean"

            # 2. Homoglyph substitution on root label (e.g. paypa1 -> paypal)
            norm_root = root_label
            for char, repl in HOMOGLYPH_MAP.items():
                norm_root = norm_root.replace(char, repl)
            if norm_root == brand and root_label != brand:
                return True, brand, f"Homoglyph substitution detected ({root_label} -> {brand})"

            # 3. Edit-distance typosquatting on root label (e.g. Distance 1 from brand)
            dist = ProvenanceMatrix._levenshtein(root_label, brand)
            if dist == 1 and len(root_label) >= 4 and root_label != brand:
                return True, brand, f"Typosquatting lookalike (Distance 1 from {brand})"

            # 4. Brand compounding / hijacking in root label (e.g. paypal-security, secure-google, apple-verify)
            if brand in root_label and root_label != brand:
                return True, brand, f"Brand hijacking compound domain ({root_label}) targeting {brand}"

            # 5. Brand impersonation in subdomain of third-party root (e.g. paypal.update-service.tk)
            if brand in subdomain.split('.'):
                return True, brand, f"Brand impersonation in subdomain targeting {brand}"

        return False, None, "Clean"

    @staticmethod
    def analyze(
        sender_header: str,
        raw_headers: str,
        spf: str = "UNKNOWN",
        dkim: str = "UNKNOWN",
        dmarc: str = "UNKNOWN",
        urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        signals = []
        threat_score = 0.0  # 0.0 (Clean) to 1.0 (Critical)
        confidence = 0.85

        display_name, email_addr = parseaddr(sender_header) if sender_header else ("", "")
        sender_domain = _mailbox_hostname(email_addr)

        # 1. Display Name Spoofing Detection
        if display_name:
            brand = _display_brand(display_name)
            if brand and sender_domain and not CrossEntityContradictionEngine._is_official_domain(brand, sender_domain):
                threat_score += 0.45
                signals.append(f"Display Name Domain Difference: '{display_name}' uses sender domain '@{sender_domain}'. Delegation or relay use is possible; spoofing is not verified.")

            # Check if display name contains an embedded email that differs from sender
            embedded_email = re.search(r'[\w\.-]+@[\w\.-]+', display_name)
            if embedded_email and embedded_email.group(0).lower() != email_addr.lower():
                threat_score += 0.35
                signals.append(f"Envelope Discrepancy: Display-name address '{embedded_email.group(0)}' differs from the From address; this alone does not prove a fake identity.")

        # 2. Lookalike & Homoglyph Domain Detection
        is_lookalike, target_brand, reason = _email_lookalike_candidate(sender_domain)
        if is_lookalike:
            threat_score += 0.50
            signals.append(f"Lookalike Domain Candidate: Sender domain '{sender_domain}' resembles '{target_brand}' under configured rules; ownership and deception are not verified.")

        # 3. Check Embedded Link Domains for Combosquatting / Lookalikes
        if urls:
            for u in urls[:5]:
                try:
                    u_domain = _identity_hostname(u)
                    u_is_look, u_target, u_reason = _email_lookalike_candidate(u_domain)
                    if u_is_look:
                        threat_score += 0.45
                        signals.append(f"Embedded Link Lookalike Candidate: Link domain '{u_domain}' resembles '{u_target}' under configured rules; requires verification.")
                        break
                except Exception:
                    pass

        # Message headers are assertions, not trusted receiver validation.
        from utils.email_authentication import untrusted_authentication_report
        authentication = untrusted_authentication_report(raw_headers, spf, dkim, dmarc)
        spf, dkim, dmarc = (authentication[name] for name in ("spf", "dkim", "dmarc"))
        if authentication["reported_available"]:
            signals.append("Supplied authentication results are unverified; trusted receiver validation is unavailable")

        crypto_fails = [spf, dkim, dmarc].count("FAIL")
        if crypto_fails >= 2:
            threat_score += 0.40
            signals.append(f"Cryptographic Identity Failure: Multiple auth checks failed (SPF: {spf}, DKIM: {dkim}, DMARC: {dmarc})")
        elif crypto_fails == 1:
            threat_score += 0.20
            signals.append(f"Authentication Weakness: Failed check (SPF: {spf}, DKIM: {dkim})")
        elif spf == "PASS" and dkim == "PASS":
            signals.append("Supplied headers report SPF and DKIM as passing; these results were not independently verified")

        threat_score = min(max(threat_score, 0.0), 1.0)
        has_headers = bool(raw_headers and len(raw_headers.strip()) > 10)
        has_identity_evidence = bool(sender_domain or urls or has_headers)
        auth_reported = any(result != "UNKNOWN" for result in (spf, dkim, dmarc))
        if not has_identity_evidence:
            confidence = 0.0
        elif is_lookalike or any("Spoofing" in signal or "Lookalike" in signal for signal in signals):
            confidence = 0.75
        elif auth_reported:
            confidence = 0.65
        else:
            confidence = 0.30

        return {
            "pillar": "Pillar 1: Provenance Matrix",
            "score": round(threat_score * 100, 1),
            "threat_prob": round(threat_score, 3),
            "confidence": confidence,
            "available": has_identity_evidence,
            "score_kind": "unvalidated_review_score",
            "confidence_kind": "assigned_reliability_prior_not_measured",
            "threat_probability_validated": False,
            "display_name_spoofed": False,
            "display_name_domain_difference": any("Display Name Domain Difference" in s for s in signals),
            "lookalike_domain": is_lookalike,
            "crypto_auth": authentication,
            "signals": signals if signals else [
                "No impersonation signal found in the available sender and link data."
                if has_identity_evidence else "Sender identity evidence unavailable."
            ]
        }

    @staticmethod
    def analyze_web(
        domain: str,
        url: str = "",
        tls_intel: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Web Mode: Evaluates domain authenticity, combosquatting/lookalike targets,
        and cryptographic TLS Certificate Lineage Mismatch.
        """
        signals = []
        threat_score = 0.0
        confidence = 0.85

        domain_clean = domain.lower().strip() if domain else ""
        if "://" in domain_clean:
            domain_clean = domain_clean.split("://")[-1].split("/")[0].split(":")[0]
        if domain_clean.startswith("www."):
            domain_clean = domain_clean[4:]

        # 1. Lookalike & Homoglyph / Typosquatting / Combosquatting Domain Detection
        is_lookalike, target_brand, reason = ProvenanceMatrix.check_lookalike_domain(domain_clean)
        if is_lookalike:
            threat_score += 0.55
            signals.append(f"Lookalike Domain: {reason} targeting '{target_brand}'")

        # 2. TLS Certificate Lineage Mismatch Analysis
        tls = tls_intel or {}
        if tls.get("available"):
            # Direct lineage mismatch (Brand impersonation with free/disposable DV cert or wrong subject)
            if tls.get("lineage_mismatch"):
                threat_score += 0.50
                mismatch_reason = tls.get("mismatch_reason", "TLS Certificate Lineage Mismatch detected.")
                signals.append(f"[Lineage Mismatch] {mismatch_reason}")

            if tls.get("is_self_signed"):
                threat_score += 0.40
                signals.append("Cryptographic Trust Failure: Target domain employs an untrusted Self-Signed Certificate.")

            if tls.get("is_expired"):
                threat_score += 0.25
                signals.append("Certificate Expiration: SSL certificate has expired, indicating abandoned or compromised host.")

            # Ephemeral certificate issuance (< 7 days old)
            cert_age = tls.get("cert_age_days", 999)
            if 0 <= cert_age <= 7:
                threat_score += 0.20
                signals.append(f"Ephemeral TLS Certificate: Issued only {cert_age} days ago (rapid provisioning).")

            # High SAN count indicating cheap shared multi-tenant proxy
            if tls.get("san_count", 0) > 40:
                threat_score += 0.15
                signals.append(f"Shared Infrastructure: Certificate SAN covers {tls.get('san_count')} distinct domains.")
        else:
            confidence = 0.60  # Reduced confidence if TLS couldn't be evaluated

        threat_score = min(max(threat_score, 0.0), 1.0)

        return {
            "pillar": "Pillar 1: Provenance Matrix",
            "score": round(threat_score * 100, 1),
            "threat_prob": round(threat_score, 3),
            "confidence": confidence,
            "lookalike_domain": is_lookalike,
            "target_brand": target_brand,
            "tls_lineage_mismatch": tls.get("lineage_mismatch", False),
            "evidence_complete": bool(tls.get("available")),
            "cert_fingerprint": tls.get("cert_fingerprint", ""),
            "signals": signals or (["No provenance threat indicators in completed checks."] if tls.get("available") else ["TLS evidence unavailable; only local domain indicators were assessed."])
        }


# ==============================================================================
# PILLAR 2: INFRASTRUCTURE LINEAGE (Server Hops & Network Lineage)
# ==============================================================================
class InfrastructureLineage:
    """
    Parses MTA Received hops, bulletproof/disposable VPS origins, and residential proxy relays.
    """

    @staticmethod
    def parse_received_hops(headers_text: str) -> List[Dict[str, str]]:
        if not headers_text:
            return []
        
        hops = []
        # Match Received: from ... by ... with ...
        matches = re.findall(r'received:\s*from\s+([^\n;]+)', headers_text, re.IGNORECASE)
        for m in matches:
            cleaned = m.strip()
            # Extract IP if present in hop
            ip_match = re.search(r'\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]', cleaned)
            ip = ip_match.group(1) if ip_match else ""
            hops.append({"raw_hop": cleaned, "ip": ip})
        return hops

    @staticmethod
    def analyze(headers_text: str, urls: List[str] = None, url_risk_score: Optional[float] = None) -> Dict[str, Any]:
        signals = []
        threat_score = 0.0
        urls = urls or []
        has_headers = bool(headers_text and len(headers_text.strip()) >= 10)

        if not has_headers and not urls:
            return {
                "pillar": "Pillar 2: Infrastructure Lineage",
                "score": 0.0,
                "threat_prob": 0.0,
                "confidence": 0.0,
                "available": False,
                "hops_count": 0,
                "score_kind": "unvalidated_review_score",
                "confidence_kind": "assigned_reliability_prior_not_measured",
                "threat_probability_validated": False,
                "signals": ["Mail route and link evidence unavailable."]
            }

        hops = InfrastructureLineage.parse_received_hops(headers_text) if has_headers else []
        confidence = 0.80 if hops else (0.45 if urls else 0.20)

        # 1. Inspect Originating Hop (first MTA hop at bottom of received headers)
        if hops:
            origin_hop = hops[-1]["raw_hop"].lower()
            origin_ip = hops[-1]["ip"]

            # Check disposable/bulletproof VPS hosting
            for vps_pat in DISPOSABLE_HOSTING_PATTERNS:
                if re.search(vps_pat, origin_hop):
                    threat_score += 0.35
                    signals.append(f"Supplied Received header matches hosting pattern '{vps_pat}'; trusted origin and hosting abuse are not verified.")
                    break

            # Check residential VPN / dynamic pool relays
            for res_pat in RESIDENTIAL_RELAY_PATTERNS:
                if re.search(res_pat, origin_hop):
                    threat_score += 0.40
                    signals.append(f"Supplied Received header matches consumer relay pattern '{res_pat}'; proxy/VPN use is not verified.")
                    break

        # 2. Suspicious TLD Infrastructure in URLs
        if urls:
            suspicious_tld_found = False
            for u in urls[:5]:
                for tld in [".xyz", ".biz", ".top", ".tk", ".pw", ".ru", ".cc", ".link", ".click"]:
                    if (urlparse(u).hostname or "").lower().rstrip(".").endswith(tld):
                        suspicious_tld_found = True
                        break
            if suspicious_tld_found:
                threat_score += 0.25
                signals.append("An embedded link hostname ends in a configured TLD review list; domain reputation or disposability is not established.")

        if url_risk_score is not None and urls:
            url_risk = min(max(float(url_risk_score), 0.0), 100.0) / 100.0
            if url_risk >= 0.45:
                threat_score = max(threat_score, url_risk)
                confidence = max(confidence, 0.65)
                signals.append("URL analysis identified suspicious link or domain signals.")

        threat_score = min(max(threat_score, 0.0), 1.0)

        return {
            "pillar": "Pillar 2: Infrastructure Lineage",
            "score": round(threat_score * 100, 1),
            "threat_prob": round(threat_score, 3),
            "confidence": confidence,
            "available": True,
            "hops_count": len(hops),
            "score_kind": "unvalidated_review_score",
            "confidence_kind": "assigned_reliability_prior_not_measured",
            "threat_probability_validated": False,
            "signals": signals if signals else [
                "No suspicious patterns found in the available mail route and link data."
            ]
        }

    @staticmethod
    def analyze_web(
        domain: str,
        dns_ephemerality: Optional[Dict[str, Any]] = None,
        asn_intel: Optional[Dict[str, Any]] = None,
        geo: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Web Mode: Evaluates Autonomous System Brand Incongruity (ABCI),
        DNS Ephemerality / Fast-Flux botnet cycling, and low-reputation hosting relays.
        """
        signals = []
        threat_score = 0.0
        confidence = 0.85

        domain_clean = domain.lower().strip() if domain else ""
        if "://" in domain_clean:
            domain_clean = domain_clean.split("://")[-1].split("/")[0].split(":")[0]
        if domain_clean.startswith("www."):
            domain_clean = domain_clean[4:]

        dns = dns_ephemerality or {}
        asn = asn_intel or {}

        # 1. Autonomous System Brand Incongruity (ABCI)
        if asn.get("is_incongruent"):
            threat_score += 0.55
            signals.extend(asn.get("signals", []))
        elif asn.get("is_hosting"):
            # Check disposable / bulletproof VPS
            isp_lower = f"{asn.get('isp', '')} {asn.get('as_name', '')}".lower()
            for vps in DISPOSABLE_HOSTING_PATTERNS:
                if vps in isp_lower:
                    threat_score += 0.30
                    signals.append(f"Disposable VPS Hosting: Hosted on unmanaged provider '{asn.get('isp')}' ({asn.get('asn')})")
                    break

        # 2. DNS Ephemerality & Fast-Flux Index
        # Note: weight is 0.30 (not 0.45) — many legitimate ISPs worldwide use short TTLs,
        # so this alone should never be a decisive signal without other corroborating evidence.
        ephem_idx = dns.get("ephemerality_index", 0.0)
        if ephem_idx > 0:
            threat_score += ephem_idx * 0.30
            signals.extend(dns.get("signals", []))

        # 3. Truly disposable / high-abuse TLDs only
        # Does NOT include legitimate national ccTLDs (.pk, .ru, .cn, .in, .uk, etc.)
        # or popular gTLDs (.biz, .info) used by real businesses.
        suspicious_tlds = [".xyz", ".top", ".tk", ".pw", ".cc", ".link", ".click", ".gq", ".ml", ".ga", ".cf", ".work", ".live"]
        trusted_cctlds  = {
            ".pk", ".in", ".uk", ".au", ".ca", ".de", ".fr", ".jp", ".br", ".mx", ".za", ".ng",
            ".ae", ".sa", ".bd", ".ru", ".cn", ".tr", ".it", ".es", ".nl", ".se", ".no", ".dk",
            ".nz", ".sg", ".my", ".ph", ".id", ".th", ".vn", ".eg", ".biz", ".info",
            ".com.pk", ".net.pk", ".org.pk", ".co.uk", ".co.in", ".co.au",
        }
        uses_trusted = any(domain_clean.endswith(tld) for tld in trusted_cctlds)
        if not uses_trusted and any(domain_clean.endswith(tld) for tld in suspicious_tlds):
            threat_score += 0.25
            matching = next(tld for tld in suspicious_tlds if domain_clean.endswith(tld))
            signals.append(f"Low-Reputation TLD: Domain uses '{matching}' commonly favored by disposable phishing campaigns.")

        threat_score = min(max(threat_score, 0.0), 1.0)
        if not dns.get("available") and not asn.get("available"):
            confidence = 0.40

        return {
            "pillar": "Pillar 2: Infrastructure Lineage",
            "score": round(threat_score * 100, 1),
            "threat_prob": round(threat_score, 3),
            "confidence": confidence,
            "asn": asn.get("asn", "Unknown"),
            "isp": asn.get("isp", "Unknown"),
            "ephemerality_index": ephem_idx,
            "evidence_complete": bool(dns.get("available") and asn.get("available")),
            "signals": signals or (["No infrastructure threat indicators in completed checks."] if dns.get("available") and asn.get("available") else ["Infrastructure assessment incomplete: DNS or ASN evidence unavailable."])
        }


# ==============================================================================
# PILLAR 3: SEMANTIC DECEPTION (Psychological Intent & Multilingual Vectors)
# ==============================================================================
class SemanticDeception:
    """
    Synthesizes psychological social engineering, zero-link BEC, and vision quishing.
    """

    @staticmethod
    def analyze(
        bec_result: Dict[str, Any],
        content_result: Dict[str, Any],
        vector_result: Dict[str, Any],
        vision_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        signals = []
        threat_score = 0.0
        confidence = 0.90

        # 1. Zero-Link BEC Intent
        bec_score = bec_result.get("bec_score", 0) / 100.0
        if bec_result.get("is_bec") or bec_score >= 0.40:
            threat_score = max(threat_score, bec_score)
            signals.append(f"Possible intent: {bec_result.get('intent_category', 'BEC')} (unvalidated review score {int(bec_score*100)}/100; not established coercion or a calibrated probability)")

        # 2. Multilingual Urgency & Social Engineering Lures
        observed_content_score = content_result.get("score")
        content_score = observed_content_score / 100.0 if observed_content_score is not None else 0.0
        if content_result.get("has_urgency_language"):
            threat_score = max(threat_score, content_score * 0.85)
            signals.append("Account-action or urgency wording matched configured patterns; legitimate notifications can use the same wording.")

        if not content_result.get("is_english", True) and content_result.get("keyword_hits"):
            lang = content_result.get("language", "Non-English")
            signals.append(f"Configured account-action/urgency terms matched in {lang}; intent is not established by these terms.")

        # 3. ChromaDB Semantic Zero-Day Campaign Vector Similarity
        if vector_result.get("has_match") and vector_result.get("match_percentage", 0) >= 60:
            vec_prob = vector_result["match_percentage"] / 100.0
            threat_score = max(threat_score, vec_prob * 0.90)
            signals.append(f"Semantic Campaign Resonance: {vector_result['match_percentage']}% similarity to '{vector_result.get('matched_title')}'")

        # 4. Quishing & Visual Deception (OpenCV 5 QR code)
        if vision_result.get("has_quishing"):
            threat_score = max(threat_score, 0.90)
            signals.append("QR destination extracted from an image; QR presence alone does not establish maliciousness.")

        threat_score = min(max(threat_score, 0.0), 1.0)

        return {
            "pillar": "Pillar 3: Semantic Deception",
            "score": round(threat_score * 100, 1),
            "threat_prob": round(threat_score, 3),
            "confidence": confidence,
            "intent_category": bec_result.get("intent_category", "Clean"),
            "score_kind": "unvalidated_review_score",
            "confidence_kind": "assigned_reliability_prior_not_measured",
            "threat_probability_validated": False,
            "signals": signals if signals else ["No configured content-intent indicators matched; this is not proof of legitimacy."]
        }

    @staticmethod
    def analyze_web(
        html_analysis: Optional[Dict[str, Any]] = None,
        url: str = "",
        domain: str = ""
    ) -> Dict[str, Any]:
        """
        Web Mode: Evaluates DOM sinks, fake credential input elements,
        brand spoofing cues, and psychological urgency cues in page content.
        """
        signals = []
        threat_score = 0.0
        confidence = 0.85

        html = html_analysis or {}
        url_lower = url.lower() if url else ""

        # 1. HTML DOM Brand Mismatch (Page mimics brand logo/text but domain is unofficial)
        if html.get("brand_mismatch"):
            threat_score += 0.55
            signals.append("DOM Brand Impersonation: Page renders protected brand assets on an unverified domain.")

        # 2. URL Semantic Deception Cues
        security_cue = re.search(
            r"(?:^|[-_.//?&=])(secure|login|signin|sign-in|verify|verification|account|support|billing|update|auth|portal|banking|wallet|recover)(?:$|[-_.//?&=])",
            url_lower
        )
        if security_cue and (html.get("brand_mismatch") or html.get("obfuscated_js")):
            cue_term = security_cue.group(1)
            threat_score += 0.25
            signals.append(f"Semantic Deception Cue: URL contains high-risk auth action keyword '{cue_term}'.")

        # 3. Credential Harvesting Form (Password field over insecure/mismatched endpoint)
        if html.get("has_password_field"):
            if not url_lower.startswith("https://"):
                threat_score += 0.50
                signals.append("Insecure Credential Harvest: Password submission form detected over plain HTTP.")
            elif html.get("brand_mismatch"):
                threat_score += 0.40
                signals.append("Brand Credential Harvesting: Login form on brand-mismatched unofficial domain.")
            elif html.get("obfuscated_js"):
                threat_score += 0.35
                signals.append("Evasive Credential Ingestion: Form embedded alongside obfuscated client scripts.")
            # On standard HTTPS with no brand mismatch or evasion, authentication forms are normal

        # 4. Client-Side Evasion / Obfuscated JS
        if html.get("obfuscated_js"):
            threat_score += 0.40
            signals.append("Defensive Evasion: Obfuscated or packed client-side JavaScript detected in page DOM.")

        threat_score = min(max(threat_score, 0.0), 1.0)
        if not html.get("success"):
            confidence = 0.40

        return {
            "pillar": "Pillar 3: Semantic Deception",
            "score": round(threat_score * 100, 1),
            "threat_prob": round(threat_score, 3),
            "confidence": confidence,
            "has_credential_form": bool(html.get("has_password_field")),
            "brand_deception": bool(html.get("brand_mismatch")),
            "evidence_complete": bool(html.get("success")),
            "signals": signals or (["No semantic credential harvesting lures or DOM deception detected."] if html.get("success") else ["Page analysis unavailable; absence of DOM findings does not establish safety."])
        }


# ==============================================================================
# PILLAR 4: EVIDENTIAL FUSION (Mathematical Synthesis Without Hardcoded Weights)
# ==============================================================================
class EvidentialFusion:
    """
    Dempster-Shafer & Bayesian evidential reasoning.
    Combines masses of evidence across active modalities, dynamically handles missing
    evidence (no URLs, no headers, no images) without arbitrary hardcoded splits.
    """

    @staticmethod
    def combine_two_sources(
        m1: Tuple[float, float, float],
        m2: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Combines two Dempster-Shafer mass functions (Phish, Clean, Uncertainty).
        m = (p, c, u) where p = m({P}), c = m({C}), u = m(Theta), and p + c + u = 1.0.
        """
        p1, c1, u1 = m1
        p2, c2, u2 = m2

        # Conflict factor K
        k = (p1 * c2) + (c1 * p2)
        if k >= 0.999:
            # Extreme conflicting evidence guard
            return (0.5, 0.5, 0.0)

        one_minus_k = 1.0 - k

        # Combined masses
        p_comb = ((p1 * p2) + (p1 * u2) + (u1 * p2)) / one_minus_k
        c_comb = ((c1 * c2) + (c1 * u2) + (u1 * c2)) / one_minus_k
        u_comb = (u1 * u2) / one_minus_k

        return (p_comb, c_comb, u_comb)

    @staticmethod
    def synthesize(
        provenance: Dict[str, Any],
        infrastructure: Dict[str, Any],
        semantic: Dict[str, Any],
        context: str = "email"
    ) -> Dict[str, Any]:
        """
        Synthesizes the 3 source pillars into a single unified verdict.
        Handles missing modalities dynamically by allocating uncertainty to Theta.
        Context can be 'email' or 'web'.
        """
        sources = [provenance, infrastructure, semantic]
        masses = []

        for src in sources:
            p_risk = src.get("threat_prob", 0.0)
            conf = src.get("confidence", 0.70)
            # Mass distribution:
            # m({Phish}) = p_risk * conf
            # m({Clean}) = (1.0 - p_risk) * conf
            # m(Theta)   = 1.0 - conf (Uncertainty / Missing modality)
            m_phish = p_risk * conf
            # Partial checks may retain observed positive findings, but missing
            # observations cannot contribute affirmative clean evidence.
            m_clean = (1.0 - p_risk) * conf if src.get("evidence_complete", context != "web") else 0.0
            m_theta = 1.0 - m_phish - m_clean
            masses.append((m_phish, m_clean, m_theta))

        # Iteratively fuse masses
        fused = masses[0]
        for m in masses[1:]:
            fused = EvidentialFusion.combine_two_sources(fused, m)

        fused_phish, fused_clean, fused_theta = fused

        # Posterior probability of threat (Pignistic probability transformation)
        # Posterior probability of threat (Pignistic probability transformation)
        # BetP(Phish) = m(Phish) + (m(Theta) * Prior)
        # Prior reduced to 0.10 for web — a lower base rate prevents uncertainty from
        # pushing legitimate regional/ccTLD domains into false-positive phishing verdicts.
        prior_phish = 0.10 if context == "web" else 0.12
        final_prob = fused_phish + (fused_theta * prior_phish)
        final_score = int(round(final_prob * 100))

        # Score floors — only applied when there is real, confirmed multi-signal evidence:
        # Floor 1: Cryptographic TLS lineage mismatch (the cert's org literally doesn't match the domain)
        if provenance.get("tls_lineage_mismatch") and provenance.get("lookalike_domain"):
            final_score = max(final_score, 80)
        # Floor 2: Confirmed lookalike + active credential harvesting DOM (clear attack chain)
        elif provenance.get("lookalike_domain") and semantic.get("has_credential_form") and semantic.get("brand_deception"):
            final_score = max(final_score, 85)
        # Infrastructure score alone (without confirmed brand lookalike) must NOT set a floor —
        # legitimate regional ISPs score high on infrastructure signals without being phishing.

        final_score = min(max(final_score, 0), 100)

        return {
            "pillar": "Pillar 4: Evidential Fusion",
            "context": context,
            "final_score": final_score,
            "threat_probability": round(final_prob, 4),
            "belief_phish": round(fused_phish, 4),
            "belief_clean": round(fused_clean, 4),
            "uncertainty": round(fused_theta, 4),
            "modality_contributions": {
                "provenance_matrix": provenance.get("score", 0),
                "infrastructure_lineage": infrastructure.get("score", 0),
                "semantic_deception": semantic.get("score", 0)
            }
        }


# ==============================================================================
# CROSS-ENTITY CONTRADICTION MATRIX (Topological Deception & Lineage Check)
# ==============================================================================
class CrossEntityContradictionEngine:
    """
    OriginTrace Cross-Entity Contradiction Matrix:
    Evaluates topological and cryptographic alignment across 5 structural layers:
      1. Claimed Identity: Display name and Header From address
      2. Cryptographic Auth: DKIM domain and SPF envelope alignment (RFC 7489 DMARC alignment)
      3. Anchor Link Hypertext: Visible anchor text vs actual href destination
      4. Delivery Destination: Redirect destination host vs claimed identity
      5. Infrastructure ASN: Hosting Autonomous System vs official brand ASN

    Reports observed identity/link relationships. Different hosts, asserted signatures,
    delegation and shared infrastructure do not constitute verified deception.
    Only an explicit displayed URL whose hostname differs from its href is a
    link-destination review signal; even that relationship does not prove phishing.
    """

    OFFICIAL_BRAND_ASNS = {
        "microsoft": ["as8075", "microsoft"],
        "google": ["as15169", "google"],
        "apple": ["as714", "apple"],
        "amazon": ["as16509", "amazon"],
        "paypal": ["as17012", "paypal"],
        "facebook": ["as32934", "meta", "facebook"],
        "chase": ["jpmorgan chase", "chase"],
        "netflix": ["as2906", "netflix"],
        "bankofamerica": ["bank of america"],
        "wellsfargo": ["wells fargo"],
        "citibank": ["citigroup", "citibank"]
    }

    OFFICIAL_BRAND_DOMAINS = {
        "paypal": ["paypal.com", "paypal.me", "paypal-communication.com"],
        "microsoft": ["microsoft.com", "office.com", "live.com", "outlook.com", "azure.com", "office365.com", "microsoftonline.com", "msn.com", "skype.com"],
        "google": ["google.com", "gmail.com", "youtube.com", "google.co.uk", "google.ca"],
        "apple": ["apple.com", "icloud.com"],
        "amazon": ["amazon.com", "amazon.co.uk", "amazon.de", "aws.amazon.com"],
        "netflix": ["netflix.com"],
        "chase": ["chase.com", "jpmorganchase.com"],
        "bankofamerica": ["bankofamerica.com", "bofa.com"],
        "wellsfargo": ["wellsfargo.com"],
        "citibank": ["citi.com", "citibank.com"],
        "docusign": ["docusign.com", "docusign.net"],
        "dropbox": ["dropbox.com"],
        "adobe": ["adobe.com"],
        "facebook": ["facebook.com", "fb.com", "meta.com"],
        "instagram": ["instagram.com"],
        "linkedin": ["linkedin.com"],
        "github": ["github.com", "github.io"],
        "spotify": ["spotify.com"],
        "binance": ["binance.com"],
        "coinbase": ["coinbase.com"],
    }

    TRUSTED_CDNS = ["cloudflare", "akamai", "fastly", "amazon", "aws", "google", "microsoft", "azure"]

    @staticmethod
    def _extract_domain(url_or_email: str) -> str:
        s = (url_or_email or "").strip().lower()
        if "@" in s:
            s = s.split("@")[-1].split(">")[0]
        if "://" in s:
            s = s.split("://")[-1].split("/")[0]
        s = s.split(":")[0].strip()
        if s.startswith("www."):
            s = s[4:]
        return s

    @classmethod
    def _is_official_domain(cls, brand: str, domain: str) -> bool:
        if not brand or not domain:
            return False
        d = domain.lower().strip()
        legit_list = cls.OFFICIAL_BRAND_DOMAINS.get(brand, [f"{brand}.com"])
        for legit in legit_list:
            if d == legit or d.endswith("." + legit):
                return True
        return False

    @classmethod
    def analyze(
        cls,
        sender_header: str = "",
        raw_headers: str = "",
        spf: str = "",
        dkim: str = "",
        dmarc: str = "",
        urls: Optional[List[str]] = None,
        anchor_links: Optional[List[Dict[str, str]]] = None,
        destination_domain: str = "",
        destination_asn: str = "",
        destination_isp: str = "",
        tls_cert_org: str = ""
    ) -> Dict[str, Any]:
        mismatches = []
        observations = []
        score = 0
        claimed_brand = None

        # ── Layer 1: Display Name vs From Address Identity Check ──────────
        disp_name, from_addr = parseaddr(sender_header) if sender_header else ("", "")
        from_domain = _mailbox_hostname(from_addr)

        # Check if Display Name impersonates a protected brand
        claimed_brand = _display_brand(disp_name)

        if claimed_brand and from_domain:
            # Check if From domain is legitimate for that brand
            is_legit_from = cls._is_official_domain(claimed_brand, from_domain)
            if not is_legit_from:
                observations.append({
                    "layer": "Layer 1: Identity Provenance",
                    "type": "DISPLAY_NAME_DOMAIN_DIFFERENCE",
                    "verified": False,
                    "claimed": f"Brand: {claimed_brand.title()} ('{disp_name}')",
                    "actual": f"Sender Domain: @{from_domain}",
                    "evidence": f"Display name contains '{claimed_brand.title()}', while the From hostname is '{from_domain}', outside the configured brand-domain list. Sender authorization, delegation and impersonation have not been verified."
                })

        # ── Layer 2: Cryptographic DKIM / SPF Alignment (RFC 7489) ─────────
        dkim_domains = _reported_dkim_domains(raw_headers)
        for dkim_domain in dkim_domains:
            observations.append({
                    "layer": "Layer 2: Cryptographic Authorization",
                    "type": "REPORTED_DKIM_SIGNING_DOMAIN",
                    "verified": False,
                    "claimed": f"From hostname: '{from_domain or 'unavailable'}'",
                    "actual": f"DKIM-Signature asserts d='{dkim_domain}'",
                    "evidence": f"A DKIM-Signature field asserts signing domain '{dkim_domain}'. Signature validity and DMARC alignment were not checked; this is not a cryptographic pass or failure."
                })

        # ── Layer 3: Visible Anchor Text vs Actual Href Destination ────────
        if anchor_links:
            for link in anchor_links:
                anchor_text = (link.get("text") or "").strip().lower()
                href = (link.get("href") or "").strip().lower()

                # If anchor text displays a full URL or domain (e.g. "https://paypal.com/signin")
                text_dom = _identity_hostname(anchor_text, bare=True)
                href_dom = _identity_hostname(href)

                if text_dom and href_dom and text_dom != href_dom:
                    score += 50
                    mismatches.append({
                        "layer": "Layer 3: Hyperlink Anchor Text",
                        "type": "VISIBLE_LINK_HOST_DIFFERENCE",
                        "severity": "REVIEW",
                        "verified": False,
                        "claimed": f"Visual Anchor Target: {text_dom}",
                        "actual": f"Actual Route Target: {href_dom}",
                        "evidence": f"Displayed URL hostname '{text_dom}' differs from href hostname '{href_dom}'. Redirects and delegation have not been verified; this difference requires destination review and does not establish phishing."
                    })
                    break

        # ── Layer 4: Claimed Brand vs In-Body Link Destination Domains ─────
        target_urls = urls or []
        if claimed_brand and target_urls:
            has_brand_dest = False
            unaligned_destinations = []
            for u in target_urls:
                u_dom = _identity_hostname(u)
                if not u_dom:
                    continue
                is_legit = cls._is_official_domain(claimed_brand, u_dom)
                if is_legit:
                    has_brand_dest = True
                else:
                    unaligned_destinations.append(u_dom)

            if unaligned_destinations and not has_brand_dest:
                observations.append({
                    "layer": "Layer 4: Delivery Destination",
                    "type": "BRAND_LINK_HOST_RELATIONSHIP",
                    "verified": False,
                    "claimed": f"{claimed_brand.title()} Official Destination",
                    "actual": f"Observed link hostname: {unaligned_destinations[0]}",
                    "evidence": f"The display name contains '{claimed_brand.title()}' and the observed links use hosts outside its configured brand-domain list. Tracking, partners, delegation and ownership are not verified; different hosts alone do not establish deception."
                })

        # ── Layer 5: Target Brand vs Infrastructure Routing ASN ─────────────
        dest_asn = destination_asn.lower().strip()
        dest_isp = destination_isp.lower().strip()
        asn_str = f"{dest_asn} {dest_isp}".strip()

        if claimed_brand and asn_str:
            official_asns = cls.OFFICIAL_BRAND_ASNS.get(claimed_brand, [])
            is_official_net = any(net in asn_str for net in official_asns)
            is_trusted_cdn = any(cdn in asn_str for cdn in cls.TRUSTED_CDNS)

            # Check if hosted on disposable VPS
            is_disposable = any(re.search(pat, asn_str) for pat in DISPOSABLE_HOSTING_PATTERNS)

            if is_disposable and not (is_official_net or is_trusted_cdn):
                observations.append({
                    "layer": "Layer 5: Infrastructure Routing ASN",
                    "type": "HOSTING_PROVIDER_LIST_MATCH",
                    "verified": False,
                    "claimed": f"{claimed_brand.title()} Official Autonomous System",
                    "actual": f"Hosted on: {destination_isp or destination_asn}",
                    "evidence": f"Reported destination infrastructure '{destination_isp or destination_asn}' matches a configured hosting-provider pattern. Hosting does not establish impersonation or abuse; brand routing authorization is unverified."
                })

        score = min(100, score)
        if score >= 75:
            severity = "CRITICAL"
        elif score >= 50:
            severity = "HIGH"
        elif score >= 25:
            severity = "MODERATE"
        else:
            severity = "LOW" if score > 0 else "NONE"

        is_contradictory = score >= 50

        summary = (
            f"Displayed link destination requires review ({len(mismatches)} observed hostname differences). No deception or authentication verdict is established."
            if is_contradictory else
            "No displayed URL/href hostname difference found in available evidence. Sender identity, link delegation and authentication remain unverified."
        )

        return {
            "contradiction_score": score,
            "severity": severity,
            "is_contradictory": is_contradictory,
            "claimed_brand": claimed_brand,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "observations": observations,
            "reported_dkim_domains": dkim_domains,
            "authentication_verified": False,
            "score_kind": "unvalidated_link_review_score",
            "threat_probability_validated": False,
            "alignment_summary": summary,
            "signals": [m["evidence"] for m in mismatches]
        }


# ==============================================================================
# ADAPTIVE MULTI-FIDELITY CASCADE ROUTER (Cost-Aware Early Exit Engine)
# ==============================================================================
class AdaptiveCascadeRouter:
    """Compatibility router. Web scans delegate to the shared URL pipeline.

    No clean verdict is established from host popularity or DNS-only checks.
    The legacy email route is not used by the production email endpoint.
    """

    @staticmethod
    def _extract_domain(target: str) -> str:
        s = (target or "").strip().lower()
        if "@" in s:
            s = s.split("@")[-1].split(">")[0]
        if "://" in s:
            s = s.split("://")[-1].split("/")[0]
        s = s.split(":")[0].strip()
        if s.startswith("www."):
            s = s[4:]
        return s

    TOP_AUTHORITATIVE_DOMAINS = {
        "google.com", "microsoft.com", "apple.com", "github.com",
        "amazon.com", "cloudflare.com", "wikipedia.org", "python.org",
        "youtube.com", "linkedin.com", "twitter.com", "x.com"
    }

    @classmethod
    def route_scan(
        cls,
        target_type: str,  # "url", "domain", or "email"
        raw_target: str,
        sender_header: str = "",
        raw_headers: str = "",
        spf: str = "",
        dkim: str = "",
        dmarc: str = "",
        anchor_links: Optional[List[Dict[str, str]]] = None,
        email_content: str = "",
        force_deep: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the adaptive multi-fidelity routing cascade.
        Returns execution tier, early-exit flag, risk score, contradiction report,
        and evidential uncertainty metrics.
        """
        # Web routing uses the same completed-evidence policy as the URL endpoint.
        # Host popularity and DNS alone cannot establish a clean URL verdict.
        if target_type in ("url", "domain"):
            from utils.url_scanner import scan_url
            result = scan_url(raw_target)
            return {**result, "risk_score": result.get("final_score", 0),
                    "tier_executed": "TIER_2_DEEP_FORENSICS", "early_exit": False}

        import time
        start_t = time.perf_counter()
        domain = cls._extract_domain(raw_target)

        # ── TIER 0: Instant Cache & Authority Check (< 1 ms) ──────────────
        if not force_deep and raw_target:
            # 2. Threat Cache Check (< 1 ms)
            try:
                from data.store import get_threat_cache
                cached = get_threat_cache(raw_target)
                if cached:
                    # If confirmed malicious in threat cache, exit early as PHISHING immediately
                    if cached.get("is_malicious") or cached.get("overall") == "MALICIOUS" or (cached.get("overall_score", 0) >= 70):
                        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
                        return {
                            "tier_executed": "TIER_0_CACHE",
                            "early_exit": True,
                            "latency_ms": elapsed_ms,
                            "status": "PHISHING",
                            "risk_score": max(85, cached.get("overall_score", 90)),
                            "epistemic_uncertainty": 0.02,
                            "verdict_state": "PHISHING",
                            "routing_reason": "Hit persistent threat cache (confirmed malicious indicator).",
                            "cached_record": cached
                        }
                    # If a completed deep forensic scan record exists with CLEAN verdict
                    elif cached.get("full_forensic_complete") and cached.get("verdict_state") == "CLEAN":
                        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
                        return {
                            "tier_executed": "TIER_0_CACHE",
                            "early_exit": True,
                            "latency_ms": elapsed_ms,
                            "status": "CLEAN",
                            "risk_score": cached.get("risk_score", 0),
                            "epistemic_uncertainty": 0.02,
                            "verdict_state": "CLEAN",
                            "routing_reason": "Hit persistent threat cache (pre-computed deep forensic record).",
                            "cached_record": cached
                        }
            except Exception as e:
                logger.debug(f"[CascadeRouter] Tier 0 cache lookup bypassed: {e}")

        # ── TIER 1: Lightweight Live Inspection (< 5 ms) ───────────────────
        tier1_flags = []
        requires_deep = False

        # 1. Fast DNS Ephemerality & Fast-Flux Check via DoH
        dns_res = {}
        if domain and domain not in ["localhost", "127.0.0.1"]:
            try:
                from utils.threat_intel import calculate_dns_ephemerality
                dns_res = calculate_dns_ephemerality(domain)
                if dns_res.get("available"):
                    min_ttl = dns_res.get("min_ttl")
                    if min_ttl is not None and min_ttl <= 60:
                        requires_deep = True
                        tier1_flags.append(f"Fast-Flux DNS Anomaly: Host TTL is {min_ttl}s (aggressive botnet cycling).")
                    elif dns_res.get("ephemerality_index", 0.0) >= 0.35:
                        requires_deep = True
                        tier1_flags.append("Ephemeral DNS Architecture: Host displays low TTL and disposable NS records.")
            except Exception as e:
                logger.debug(f"[CascadeRouter] Tier 1 DNS DoH check skipped: {e}")

        # 2. Raw Byte Morphology & Homoglyph Lookalike
        is_lookalike, matched_brand, lookalike_type = ProvenanceMatrix.check_lookalike_domain(domain)
        if is_lookalike:
            requires_deep = True
            tier1_flags.append(f"Byte Morphology Spoof: Domain targets brand '{matched_brand}' via {lookalike_type}.")

        # 3. TLD Tier Risk Check
        tld = domain.split(".")[-1] if "." in domain else ""
        from services.scan_engine import _TLD_RISK
        tld_risk = _TLD_RISK.get(tld, 0.1)
        if tld_risk >= 0.80:
            requires_deep = True
            tier1_flags.append(f"Abuse-Prone TLD: Top-level domain (.{tld}) carries high abuse rate ({tld_risk}).")

        # 4. Email Layer 1/2 Preliminary Check
        if target_type == "email":
            dkim_str = str(dkim).upper()
            spf_str = str(spf).upper()
            if dkim_str in ["FAIL", "PERMERROR", "FAILED"] or spf_str in ["FAIL", "SOFTFAIL"]:
                requires_deep = True
                tier1_flags.append(f"Cryptographic Auth Disconnect: DKIM={dkim_str}, SPF={spf_str}.")

            # Preliminary Contradiction Check
            contra_pre = CrossEntityContradictionEngine.analyze(
                sender_header=sender_header,
                raw_headers=raw_headers,
                spf=spf,
                dkim=dkim,
                dmarc=dmarc,
                urls=[raw_target] if raw_target else [],
                anchor_links=anchor_links
            )
            if contra_pre.get("is_contradictory") or contra_pre.get("contradiction_score", 0) >= 35:
                requires_deep = True
                tier1_flags.append(f"Cross-Entity Contradiction Detected: Score {contra_pre['contradiction_score']}/100.")

        # ── TIER 2: Deep Forensics & Evidential Reasoning (< 35 ms) ────────
        # Escalated due to anomaly, contradiction, or missing modality
        tls_res = {}
        asn_res = {}
        if domain and domain not in ["localhost", "127.0.0.1"]:
            try:
                from utils.threat_intel import analyze_tls_cert_lineage, check_asn_brand_incongruity
                tls_res = analyze_tls_cert_lineage(domain)
                asn_res = check_asn_brand_incongruity(domain)
            except Exception as e:
                logger.debug(f"[CascadeRouter] Tier 2 network lookups bypassed: {e}")

        # Full 5-Layer Cross-Entity Contradiction Analysis
        contra_full = CrossEntityContradictionEngine.analyze(
            sender_header=sender_header,
            raw_headers=raw_headers,
            spf=spf,
            dkim=dkim,
            dmarc=dmarc,
            urls=[raw_target] if raw_target else [],
            anchor_links=anchor_links,
            destination_domain=domain,
            destination_asn=asn_res.get("asn", ""),
            destination_isp=asn_res.get("isp", ""),
            tls_cert_org=tls_res.get("subject_org", "")
        )

        # Build Evidential Mass Inputs
        prov_score = 10.0
        if is_lookalike:
            prov_score += 55.0
        if tls_res.get("lineage_mismatch"):
            prov_score += 35.0
        if contra_full.get("is_contradictory"):
            prov_score = max(prov_score, float(contra_full["contradiction_score"]))

        prov_pillar = {
            "score": min(100.0, prov_score),
            "threat_prob": min(1.0, prov_score / 100.0),
            "confidence": 0.90 if tls_res.get("available") else 0.80,
            "lookalike_domain": is_lookalike,
            "tls_lineage_mismatch": tls_res.get("lineage_mismatch", False)
        }

        infra_score = 10.0
        if tld_risk >= 0.80:
            infra_score += 45.0
        if dns_res.get("ephemerality_index", 0.0) >= 0.45:
            infra_score += 35.0
        if asn_res.get("is_incongruent"):
            infra_score += 45.0
        elif asn_res.get("is_hosting"):
            infra_score += 15.0

        infra_pillar = {
            "score": min(100.0, infra_score),
            "threat_prob": min(1.0, infra_score / 100.0),
            "confidence": 0.88 if (dns_res.get("available") or asn_res.get("available")) else 0.65
        }

        semantic_score = 10.0
        if target_type == "email" and email_content:
            from services.bec_detector import predict_bec_intent
            bec = predict_bec_intent(email_content, urls_found=1 if raw_target else 0)
            semantic_score = float(bec.get("risk_score", 15.0))

        sem_pillar = {
            "score": min(100.0, semantic_score),
            "threat_prob": min(1.0, semantic_score / 100.0),
            "confidence": 0.85 if (target_type == "email" and email_content) else 0.0
        }

        # Synthesize via Dempster-Shafer Evidential Fusion
        fused = EvidentialFusion.synthesize(
            provenance=prov_pillar,
            infrastructure=infra_pillar,
            semantic=sem_pillar,
            context="email" if target_type == "email" else "web"
        )

        final_score = fused["final_score"]
        uncertainty = fused["uncertainty"]

        # Evidential Decision Logic with 3-Way State
        if contra_full.get("is_contradictory") and contra_full.get("severity") == "CRITICAL":
            final_score = max(final_score, 88)
            verdict_state = "PHISHING"
            status = "PHISHING"
        elif final_score >= 65:
            verdict_state = "PHISHING"
            status = "PHISHING"
        elif uncertainty >= 0.35 and (30 <= final_score <= 65):
            verdict_state = "INSUFFICIENT_EVIDENCE"
            status = "SUSPICIOUS"
        else:
            verdict_state = "CLEAN"
            status = "CLEAN"

        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return {
            "tier_executed": "TIER_2_DEEP_FORENSICS",
            "early_exit": False,
            "latency_ms": elapsed_ms,
            "status": status,
            "risk_score": final_score,
            "verdict_state": verdict_state,
            "epistemic_uncertainty": uncertainty,
            "contradiction_analysis": contra_full,
            "tier1_escalation_triggers": tier1_flags,
            "evidential_fusion": fused,
            "routing_reason": f"Escalated to Tier 2 Deep Forensics: {', '.join(tier1_flags) if tier1_flags else 'Deep forensic verification requested.'}"
        }


# ==============================================================================
# CAMPAIGN ATTRIBUTION GRAPH (Zero-Day Infrastructure Clustering)
# ==============================================================================

# Correct display names for protected brands; .title() mangles camelCase (paypal→Paypal).
_BRAND_DISPLAY_NAMES: Dict[str, str] = {
    "paypal":       "PayPal",
    "microsoft":    "Microsoft",
    "google":       "Google",
    "apple":        "Apple",
    "amazon":       "Amazon",
    "netflix":      "Netflix",
    "chase":        "Chase",
    "bankofamerica": "Bank of America",
    "wellsfargo":   "Wells Fargo",
    "citibank":     "Citibank",
    "docusign":     "DocuSign",
    "dropbox":      "Dropbox",
    "adobe":        "Adobe",
    "facebook":     "Facebook",
    "instagram":    "Instagram",
    "linkedin":     "LinkedIn",
    "irs":          "IRS",
    "github":       "GitHub",
    "spotify":      "Spotify",
    "binance":      "Binance",
    "coinbase":     "Coinbase",
}


class CampaignAttributor:
    """
    Campaign Attribution & Infrastructure Clustering:
    Generates a deterministic attack infrastructure signature by hashing:
    - Target Brand
    - ASN / Hosting Provider
    - TLS Certificate SHA-256 Fingerprint / Issuer
    - Primary Nameserver

    Clusters repeated attacks together into named campaigns in the database,
    providing instant cross-modal attribution across Email, URL, and Domain scans.
    """

    @staticmethod
    def generate_fingerprint(
        brand: str,
        asn: str,
        cert_fingerprint: str,
        issuer: str
    ) -> str:
        raw = f"{brand.lower().strip()}|{asn.strip()}|{cert_fingerprint.strip() or issuer.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def attribute(
        domain: str,
        brand: Optional[str] = None,
        asn: str = "",
        isp: str = "",
        tls_fingerprint: str = "",
        tls_issuer: str = "",
        is_threat: bool = False,
        session = None
    ) -> Dict[str, Any]:
        """
        Correlates the scanned artifact with existing campaigns or establishes
        a new campaign cluster in the database.
        """
        brand_name = brand or "Generic Phish"
        fp = CampaignAttributor.generate_fingerprint(
            brand=brand_name,
            asn=asn,
            cert_fingerprint=tls_fingerprint,
            issuer=tls_issuer
        )

        # Use curated display name to preserve camelCase; fall back to title-case for unknowns.
        brand_display = _BRAND_DISPLAY_NAMES.get(brand_name.lower().strip(), brand_name.title())
        cluster_name = f"{brand_display} Infrastructure Cluster" if brand_name != "Generic Phish" else f"Infrastructure Cluster ({asn or isp or 'VPS'})"
        
        result = {
            "campaign_id": f"CAMP-{fp[:8].upper()}",
            "campaign_name": cluster_name,
            "fingerprint": fp,
            "target_brand": brand_name,
            "asn": asn or isp or "Unknown",
            "is_attributed": is_threat,
            "attribution_confidence": 0.92 if (brand and (asn or tls_fingerprint)) else 0.70,
            "cluster_size": 1,
            "summary": f"Attributed to {cluster_name} via ASN & TLS infrastructure fingerprinting."
        }

        if session and is_threat:
            try:
                from data.models import Campaign, utc_now
                from sqlmodel import select

                statement = select(Campaign).where(Campaign.name == cluster_name)
                campaign = session.exec(statement).first()
                if campaign:
                    campaign.targets += 1
                    campaign.last_seen = utc_now()
                    session.add(campaign)
                    session.commit()
                    session.refresh(campaign)
                    result["cluster_size"] = campaign.targets
                else:
                    new_camp = Campaign(
                        name=cluster_name,
                        type="Credential Harvesting" if "login" in domain or "verify" in domain else "Brand Impersonation",
                        source=domain,
                        severity="critical",
                        targets=1,
                        first_seen=utc_now(),
                        last_seen=utc_now()
                    )
                    session.add(new_camp)
                    session.commit()
                    result["cluster_size"] = 1
            except Exception as e:
                logger.debug(f"[CampaignAttributor] DB persistence skipped: {e}")

        return result
