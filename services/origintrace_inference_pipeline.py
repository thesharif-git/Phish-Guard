"""
OriginTrace Unified Inference Pipeline (Historical Benchmark Evaluation Harness)
Preserved for reproducibility of Step 5 benchmark results (October 1, 2026).

NOTE: This file is NOT active in production.
Active production scanning is implemented by:
  - Email Scanner: services/scan_engine.py & services/conformal_evidential_engine.py
  - URL & Domain Scanners: utils/url_scanner.py & services/domain_scan_policy.py
  - Multi-Pillar OriginTrace: services/origintrace_pillars.py
"""

import os
import sys
import time
import math
import re
import json
import hashlib
import posixpath
import unicodedata
import urllib.parse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.sparse import hstack, csr_matrix
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.origintrace_pillars import EvidentialFusion, ProvenanceMatrix

TLD_RISK = {
    "tk": 1.0, "ml": 1.0, "ga": 1.0, "gq": 1.0, "cf": 1.0,
    "xyz": 0.9, "top": 0.9, "pw": 0.9, "cc": 0.85, "icu": 0.85,
    "club": 0.8, "online": 0.8, "site": 0.8, "click": 0.85,
    "link": 0.8, "download": 0.9, "zip": 0.9, "review": 0.8,
    "com": 0.1, "org": 0.05, "net": 0.1, "edu": 0.0, "gov": 0.0,
    "io": 0.1, "co": 0.1, "app": 0.1, "dev": 0.05
}
KNOWN_SHORTENERS = re.compile(r"bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|rb\.gy")
BRAND_KEYWORDS = re.compile(r"paypal|apple|microsoft|google|facebook|amazon|netflix|chase|wellsfargo|binance|coinbase")
OFFICIAL_DOMAINS = re.compile(r"^(paypal|apple|microsoft|google|amazon|chase)\.(com|org|net)$")

HOMOGLYPH_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'і': 'i', 'ј': 'j', 'ѕ': 's',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T',
    'Х': 'X', 'Ү': 'Y'
}

def _entropy(s: str) -> float:
    if not s: return 0.0
    freq: Dict[str, int] = {}
    for c in s: freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())

def _tld_risk(netloc: str) -> float:
    tld = netloc.split(".")[-1].lower() if "." in netloc else netloc.lower()
    return TLD_RISK.get(tld, 0.2)

def _char_continuation_rate(s: str) -> float:
    if not s: return 0.0
    max_run, cur = 1, 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        max_run = max(max_run, cur)
    return max_run / len(s)

def _unique_char_ratio(s: str) -> float:
    return len(set(s)) / len(s) if s else 0.0

def canonicalize_email_text(text: str) -> str:
    if not isinstance(text, str): return ""
    try:
        text = urllib.parse.unquote(text)
    except Exception:
        pass
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xad]', '', text)
    for k, v in HOMOGLYPH_MAP.items():
        text = text.replace(k, v)
    return text

def extract_email_anomaly_features(text: str) -> List[float]:
    if not isinstance(text, str): return [0.0] * 7
    zw_count = len(re.findall(r'[\u200b\u200c\u200d\ufeff\xad]', text))
    hg_count = sum(1 for c in text if c in HOMOGLYPH_MAP)
    hex_count = len(re.findall(r'%[0-9a-fA-F]{2}', text))
    length = max(1, len(text))
    non_ascii_count = sum(ord(c) > 127 for c in text)
    b_entropy = _entropy(text)
    return [
        float(zw_count),
        float(hg_count),
        float(hex_count),
        round(hg_count / length, 5),
        round(hex_count / length, 5),
        round(non_ascii_count / length, 5),
        round(b_entropy, 4)
    ]

def normalize_url_text(url: str) -> str:
    if not isinstance(url, str): return ""
    s = url.strip().lower()
    s = re.sub(r'^https?://', '', s)
    s = re.sub(r'^www\.', '', s)
    return s

def canonicalize_url(url: str) -> str:
    if not isinstance(url, str): return ""
    s = url.strip()
    p = urllib.parse.urlparse(s if s.startswith("http") else "http://" + s)
    raw_path = urllib.parse.unquote(p.path or "")
    canon_path = posixpath.normpath(raw_path) if raw_path else ""
    if raw_path.endswith("/") and not canon_path.endswith("/"):
        canon_path += "/"
    canon_host = unicodedata.normalize('NFKC', urllib.parse.unquote(p.netloc).lower())
    canon_query = urllib.parse.unquote(p.query or "")
    return urllib.parse.urlunparse((p.scheme or "http", canon_host, canon_path, "", canon_query, ""))

def extract_url_features(url: str) -> Dict[str, Any]:
    from urllib.parse import urlparse
    if not isinstance(url, str): url = ""
    url = url.strip()
    try:
        p = urlparse(url if url.startswith("http") else "http://" + url)
        netloc, path, query, scheme = p.netloc.lower().split(":")[0], p.path, p.query, p.scheme.lower()
    except Exception:
        netloc = path = query = scheme = ""
    clean_domain = re.sub(r"^www\.", "", netloc)
    parts = clean_domain.split(".")
    tld = parts[-1] if parts else ""
    norm = re.sub(r"^https?://", "", url.lower())
    norm = re.sub(r"^www\.", "", norm)
    dc = sum(c.isdigit() for c in norm)
    tl = len(norm) or 1
    has_brand = int(bool(BRAND_KEYWORDS.search(clean_domain)) and not bool(OFFICIAL_DOMAINS.match(clean_domain)))
    return {
        "url_length": len(norm), "entropy": round(_entropy(norm), 4),
        "digit_count": dc, "digit_ratio": round(dc / tl, 4),
        "special_char_count": sum(not c.isalnum() for c in norm),
        "hyphen_count": norm.count("-"), "at_count": norm.count("@"),
        "dot_count": clean_domain.count("."), "path_length": len(path),
        "query_length": len(query), "path_depth": path.count("/"),
        "query_param_count": query.count("&") + (1 if query else 0),
        "subdomain_count": max(0, clean_domain.count(".") - 1),
        "domain_length": len(clean_domain), "is_https": int(scheme == "https"),
        "has_ip": int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", clean_domain))),
        "has_shortener": int(bool(KNOWN_SHORTENERS.search(norm))),
        "tld_risk_score": round(_tld_risk(clean_domain), 3),
        "has_suspicious_tld": int(TLD_RISK.get(tld, 0) >= 0.75),
        "has_brand_in_url": has_brand, "has_hex_encoding": int("%" in url),
        "has_double_slash": int("//" in path),
        "char_continuation": round(_char_continuation_rate(norm), 4),
        "unique_char_ratio": round(_unique_char_ratio(norm[:50]), 4),
    }

def extract_domain_features(url_or_domain: str) -> Dict[str, Any]:
    from urllib.parse import urlparse
    if not isinstance(url_or_domain, str): url_or_domain = ""
    s = url_or_domain.strip().lower()
    try:
        p = urlparse(s if s.startswith("http") else "http://" + s)
        domain = p.netloc.split(":")[0] if p.netloc else s.split("/")[0]
    except Exception:
        domain = s.split("/")[0]
    clean_domain = re.sub(r"^www\.", "", domain).split(":")[0]
    parts = clean_domain.split(".")
    tld = parts[-1] if parts else ""
    sld = parts[-2] if len(parts) >= 2 else parts[0] if parts else ""
    has_brand = int(bool(BRAND_KEYWORDS.search(clean_domain)) and not bool(OFFICIAL_DOMAINS.match(clean_domain)))
    return {
        "domain_length": len(clean_domain), "domain_entropy": round(_entropy(clean_domain), 4),
        "domain_digit_count": sum(c.isdigit() for c in clean_domain),
        "domain_hyphen_count": clean_domain.count("-"),
        "domain_dot_count": clean_domain.count("."),
        "subdomain_depth": max(0, clean_domain.count(".") - 1),
        "has_ip": int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", clean_domain))),
        "tld_risk_score": round(_tld_risk(clean_domain), 3),
        "has_suspicious_tld": int(TLD_RISK.get(tld, 0) >= 0.75),
        "is_free_tld": int(tld in ["tk", "ml", "ga", "cf", "gq"]),
        "brand_impersonation": has_brand,
        "brand_in_subdomain": int(bool(BRAND_KEYWORDS.search(".".join(parts[:-2])))),
        "has_numbers_in_domain": int(any(c.isdigit() for c in clean_domain)),
        "has_punycode": int(clean_domain.startswith("xn--")),
        "unique_char_ratio": round(_unique_char_ratio(clean_domain), 4),
        "vowel_ratio": round(sum(c in "aeiou" for c in clean_domain) / max(1, len(clean_domain)), 4),
        "consonant_cluster": 0, "dga_score": 0.5, "homoglyph_flag": 0,
        "security_keyword": int(any(k in sld for k in ["login", "verify", "secure", "account"])),
        "sld_length": len(sld), "sld_digit_ratio": round(sum(c.isdigit() for c in sld) / max(1, len(sld)), 4),
        "tld_is_numeric": int(tld.isdigit()), "repeating_segment": 0, "known_tld_score": 0
    }

def extract_url_dual_representation(url: str) -> List[float]:
    c_u = canonicalize_url(url)
    raw = str(url)
    hex_count = len(re.findall(r'%[0-9a-fA-F]{2}', raw))
    length = max(1, len(raw))
    struct_dict = extract_url_features(c_u)
    return list(struct_dict.values()) + [float(hex_count), float(hex_count / length)]

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

class OriginTracePipeline:
    """
    Deployable, frozen OriginTrace-AdaptiveGraph Inference Pipeline.
    Loads verified model artifacts and executes reproducible scoring on pure CPU.
    """
    _instance: Optional['OriginTracePipeline'] = None

    def __init__(self, model_dir: Optional[str] = None):
        import warnings
        warnings.warn(
            "OriginTracePipeline is a historical benchmark harness (Step 5) and is not active in production.",
            DeprecationWarning,
            stacklevel=2
        )
        self.model_dir = model_dir or os.path.join(PROJECT_ROOT, "models", "frozen_pipeline")
        self.email_vectorizer = None
        self.email_specialist = None
        self.url_vectorizer = None
        self.url_specialist = None
        self.domain_specialist = None
        self.monolith_model = None
        self.config: Dict[str, Any] = {}
        self.hashes_verified: bool = False
        self._load_pipeline()

    @classmethod
    def get_instance(cls, model_dir: Optional[str] = None) -> 'OriginTracePipeline':
        if cls._instance is None:
            cls._instance = OriginTracePipeline(model_dir)
        return cls._instance

    def _load_pipeline(self):
        config_path = os.path.join(self.model_dir, "pipeline_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Frozen pipeline configuration not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Verify SHA-256 Hashes
        expected_hashes = self.config.get("artifact_sha256_hashes", {})
        for fname, exp_hash in expected_hashes.items():
            fpath = os.path.join(self.model_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing expected artifact: {fpath}")
            act_hash = compute_sha256(fpath)
            if act_hash != exp_hash:
                raise ValueError(f"Artifact hash mismatch for {fname}! Expected {exp_hash}, got {act_hash}")

        self.hashes_verified = True

        # Load models and vectorizers
        self.email_vectorizer = joblib.load(os.path.join(self.model_dir, "email_vectorizer.joblib"))
        self.email_specialist = joblib.load(os.path.join(self.model_dir, "email_specialist.joblib"))
        self.url_vectorizer = joblib.load(os.path.join(self.model_dir, "url_vectorizer.joblib"))
        self.url_specialist = joblib.load(os.path.join(self.model_dir, "url_specialist.joblib"))
        self.domain_specialist = joblib.load(os.path.join(self.model_dir, "domain_specialist.joblib"))
        self.monolith_model = joblib.load(os.path.join(self.model_dir, "monolith_model.joblib"))

        # Thresholds
        thresh = self.config.get("operating_thresholds", {})
        self.thresh_05 = float(thresh.get("strict_blocking_tau_05", 0.7862))
        self.thresh_20 = float(thresh.get("deep_inspection_tau_20", 0.3238))
        self.thresh_bal = float(thresh.get("balanced_optimum_tau", 0.7007))

        # Escalation rules
        esc = self.config.get("cascade_escalation_rules", {})
        self.conflict_threshold = float(esc.get("conflict_metric_threshold", 0.40))
        self.unc_range = esc.get("uncertainty_range", [0.40, 0.60])

    def score_email_raw(self, text: str) -> float:
        c_text = canonicalize_email_text(text)
        tfidf = self.email_vectorizer.transform([c_text]).astype(np.float32)
        anom = csr_matrix(np.array([extract_email_anomaly_features(text)], dtype=np.float32))
        feat = hstack([tfidf, anom])
        return float(self.email_specialist.booster_.predict(feat)[0])

    def score_url_raw(self, url: str) -> float:
        norm_u = normalize_url_text(url)
        morph = self.url_vectorizer.transform([norm_u]).astype(np.float32)
        struct = csr_matrix(np.array([extract_url_dual_representation(url)], dtype=np.float32))
        feat = hstack([morph, struct])
        return float(self.url_specialist.booster_.predict(feat)[0])

    def score_domain_raw(self, domain_or_url: str) -> float:
        dom_feats = np.array([list(extract_domain_features(domain_or_url).values())], dtype=np.float32)
        return float(self.domain_specialist.booster_.predict(dom_feats)[0])

    def score_monolith_raw(self, email_text: str, url: str) -> float:
        c_text = canonicalize_email_text(email_text)
        em_tfidf = self.email_vectorizer.transform([c_text]).astype(np.float32)
        em_anom = csr_matrix(np.array([extract_email_anomaly_features(email_text)], dtype=np.float32))
        em_feat = hstack([em_tfidf, em_anom])

        norm_u = normalize_url_text(url)
        u_morph = self.url_vectorizer.transform([norm_u]).astype(np.float32)
        u_struct = csr_matrix(np.array([extract_url_dual_representation(url)], dtype=np.float32))
        u_feat = hstack([u_morph, u_struct])

        dom_feat = csr_matrix(np.array([list(extract_domain_features(url).values())], dtype=np.float32))

        # Ensure correct column shapes
        n_samples = 1
        mono_vec = hstack([em_feat, u_feat, dom_feat])
        return float(self.monolith_model.booster_.predict(mono_vec)[0])

    def predict(
        self,
        scan_type: str,  # "url", "domain", "email", or "multimodal"
        raw_input: str,
        email_content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        force_tier2: bool = False
    ) -> Dict[str, Any]:
        """
        Unified deployable inference pipeline execution.
        """
        t0 = time.perf_counter()
        metadata = metadata or {}
        scan_type = scan_type.lower().strip()

        if scan_type == "url":
            raw_url = raw_input.strip()
            # Fast-Path Pure CPU Inference
            p_u = self.score_url_raw(raw_url)
            conf_u = float(max(p_u, 1.0 - p_u))

            p_d = self.score_domain_raw(raw_url)
            conf_d = float(max(p_d, 1.0 - p_d))

            fusion_res = EvidentialFusion.synthesize(
                provenance={"threat_prob": p_d, "confidence": conf_d},
                infrastructure={"threat_prob": p_u, "confidence": conf_u},
                semantic={"threat_prob": 0.5, "confidence": 0.0},
                context="url"
            )
            final_prob = fusion_res["threat_probability"]
            final_score = fusion_res["final_score"]
            uncertainty = fusion_res["uncertainty"]

            elapsed_ms = (time.perf_counter() - t0) * 1000

            return {
                "scan_type": "url",
                "target": raw_url,
                "threat_probability": round(final_prob, 4),
                "risk_score": final_score,
                "tier_executed": "TIER_1_LIGHTWEIGHT",
                "early_exit": True,
                "is_escalated": False,
                "uncertainty": round(uncertainty, 4),
                "belief_phish": round(fusion_res["belief_phish"], 4),
                "belief_clean": round(fusion_res["belief_clean"], 4),
                "operating_verdicts": {
                    "strict_blocking_05": int(final_prob >= self.thresh_05),
                    "deep_inspection_20": int(final_prob >= self.thresh_20),
                    "balanced_optimum": int(final_prob >= self.thresh_bal)
                },
                "latency_ms": round(elapsed_ms, 3),
                "hashes_verified": self.hashes_verified
            }

        elif scan_type == "domain":
            raw_domain = raw_input.strip()
            p_d = self.score_domain_raw(raw_domain)
            final_prob = p_d
            final_score = int(round(p_d * 100))
            elapsed_ms = (time.perf_counter() - t0) * 1000

            return {
                "scan_type": "domain",
                "target": raw_domain,
                "threat_probability": round(final_prob, 4),
                "risk_score": final_score,
                "tier_executed": "TIER_1_LIGHTWEIGHT",
                "early_exit": True,
                "is_escalated": False,
                "uncertainty": 0.05,
                "operating_verdicts": {
                    "strict_blocking_05": int(final_prob >= self.thresh_05),
                    "deep_inspection_20": int(final_prob >= self.thresh_20),
                    "balanced_optimum": int(final_prob >= self.thresh_bal)
                },
                "latency_ms": round(elapsed_ms, 3),
                "hashes_verified": self.hashes_verified
            }

        elif scan_type == "email":
            raw_em = (raw_input or email_content).strip()
            p_em = self.score_email_raw(raw_em)
            final_prob = p_em
            final_score = int(round(p_em * 100))
            elapsed_ms = (time.perf_counter() - t0) * 1000

            return {
                "scan_type": "email",
                "target_length": len(raw_em),
                "threat_probability": round(final_prob, 4),
                "risk_score": final_score,
                "tier_executed": "TIER_1_LIGHTWEIGHT",
                "early_exit": True,
                "is_escalated": False,
                "uncertainty": 0.05,
                "operating_verdicts": {
                    "strict_blocking_05": int(final_prob >= self.thresh_05),
                    "deep_inspection_20": int(final_prob >= self.thresh_20),
                    "balanced_optimum": int(final_prob >= self.thresh_bal)
                },
                "latency_ms": round(elapsed_ms, 3),
                "hashes_verified": self.hashes_verified
            }

        elif scan_type == "multimodal":
            raw_em = (email_content or "").strip()
            raw_url = raw_input.strip()
            has_em = bool(raw_em)

            # 1. Tier 1 Fast Consensus
            p_em = self.score_email_raw(raw_em) if has_em else 0.50
            conf_em = float(max(p_em, 1.0 - p_em)) if has_em else 0.0

            p_u = self.score_url_raw(raw_url)
            conf_u = float(max(p_u, 1.0 - p_u))

            p_d = self.score_domain_raw(raw_url)
            conf_d = float(max(p_d, 1.0 - p_d))

            if has_em:
                # Historical Step 5 benchmark weighting formula (superseded in production by services/origintrace_pillars.py)
                p_t1_score = 0.45 * p_em + 0.35 * p_u + 0.20 * p_d
                m_phish_em = p_em * conf_em
                m_clean_em = (1.0 - p_em) * conf_em
                m_phish_u = p_u * conf_u
                m_clean_u = (1.0 - p_u) * conf_u
                conflict_k = (m_clean_em * m_phish_u) + (m_phish_em * m_clean_u)
            else:
                p_t1_score = 0.60 * p_u + 0.40 * p_d
                conflict_k = 0.0

            # Gating Rule: Conflict > 0.40 OR Epistemic Uncertainty in [0.40, 0.60]
            needs_escalation = bool(force_tier2 or conflict_k > self.conflict_threshold or (self.unc_range[0] <= p_t1_score <= self.unc_range[1]))

            if needs_escalation:
                tier_executed = "TIER_2_DEEP_FORENSICS"
                is_escalated = True
                dom = raw_url.split("://")[-1].split("/")[0].split(":")[0].lstrip("www.")
                is_lookalike, brand_target, reason = ProvenanceMatrix.check_lookalike_domain(dom)

                # Tier 2 Payload Asymmetric Resolution
                if is_lookalike or (p_u >= 0.75 and p_d >= 0.45) or p_u >= 0.85:
                    final_prob = max(p_u, 0.90)
                elif p_u <= 0.35 or (has_em and p_em <= 0.15 and p_u < 0.65 and not is_lookalike):
                    final_prob = min(p_u, p_em, 0.10)
                else:
                    final_prob = p_t1_score
            else:
                tier_executed = "TIER_1_LIGHTWEIGHT"
                is_escalated = False
                final_prob = p_t1_score

            final_score = int(round(final_prob * 100))
            elapsed_ms = (time.perf_counter() - t0) * 1000

            return {
                "scan_type": "multimodal",
                "threat_probability": round(final_prob, 4),
                "risk_score": final_score,
                "tier_executed": tier_executed,
                "early_exit": not is_escalated,
                "is_escalated": is_escalated,
                "conflict_metric": round(conflict_k, 4),
                "tier1_fast_score": round(p_t1_score, 4),
                "operating_verdicts": {
                    "strict_blocking_05": int(final_prob >= self.thresh_05),
                    "deep_inspection_20": int(final_prob >= self.thresh_20),
                    "balanced_optimum": int(final_prob >= self.thresh_bal)
                },
                "latency_ms": round(elapsed_ms, 3),
                "hashes_verified": self.hashes_verified
            }

        else:
            raise ValueError(f"Unknown scan_type: {scan_type}. Expected 'url', 'domain', 'email', or 'multimodal'.")
