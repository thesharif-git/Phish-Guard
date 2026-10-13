"""Subjective-logic fusion with split-conformal prediction sets.
Reliability constants assign ignorance mass; they are not learned uncertainty.
Coverage/benign Type I claims require exchangeability and a frozen scorer.
There is no FDR guarantee or demonstrated novelty claim.
"""

import os
import json
import math
import hashlib
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFORMAL_PATH = os.path.join(MODELS_DIR, "conformal_params.json")


# ==============================================================================
# 1. DIRICHLET SUBJECTIVE OPINION
# ==============================================================================

class SubjectiveOpinion:
    """
    Represents an evidential belief opinion over binary state space Theta = {Clean, Phish}.
    Satisfies: b_phish + b_clean + u = 1.0, with b_phish, b_clean, u >= 0.
    """
    __slots__ = ("b_phish", "b_clean", "u", "source")

    def __init__(self, b_phish: float, b_clean: float, u: float, source: str = "unknown"):
        total = b_phish + b_clean + u
        if total <= 0:
            self.b_phish = 0.0
            self.b_clean = 0.0
            self.u = 1.0
        else:
            self.b_phish = max(0.0, b_phish / total)
            self.b_clean = max(0.0, b_clean / total)
            self.u = max(0.0, u / total)
        self.source = source

    @classmethod
    def from_probability(cls, prob_phish: float, confidence: float = 0.85, source: str = "ml") -> "SubjectiveOpinion":
        """
        Converts a calibrated probability p in [0, 1] and reliability confidence c in [0, 1]
        into a Dirichlet subjective opinion.
        When confidence -> 0 (missing modality / no evidence), u -> 1.0.
        """
        p = min(max(float(prob_phish), 0.0), 1.0)
        c = min(max(float(confidence), 0.0), 0.999)

        if c <= 0.001:
            # Completely missing evidence
            return cls(b_phish=0.0, b_clean=0.0, u=1.0, source=source)

        # Dirichlet evidence mapping: e = 2 * c / (1 - c)
        evidence_scale = (2.0 * c) / (1.0 - c)
        e_phish = p * evidence_scale
        e_clean = (1.0 - p) * evidence_scale
        s = e_phish + e_clean + 2.0  # Dirichlet strength

        b_phish = e_phish / s
        b_clean = e_clean / s
        u = 2.0 / s

        return cls(b_phish=b_phish, b_clean=b_clean, u=u, source=source)

    @classmethod
    def vacuous(cls, source: str = "vacuous") -> "SubjectiveOpinion":
        """Returns a completely ignorant opinion (u = 1.0, b_phish = 0, b_clean = 0)."""
        return cls(0.0, 0.0, 1.0, source=source)

    @property
    def expected_probability(self) -> float:
        """Pignistic transformation: E[P(Phish)] = b_phish + (u * 0.5)."""
        return self.b_phish + (self.u * 0.5)

    @property
    def aleatoric_uncertainty(self) -> float:
        """Ambiguity / conflict between clean and phish evidence."""
        return min(self.b_phish, self.b_clean) * 2.0

    @property
    def epistemic_uncertainty(self) -> float:
        """Lack of evidence mass (ignorance)."""
        return self.u

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "b_phish": round(self.b_phish, 4),
            "b_clean": round(self.b_clean, 4),
            "epistemic_uncertainty": round(self.u, 4),
            "aleatoric_uncertainty": round(self.aleatoric_uncertainty, 4),
            "expected_prob": round(self.expected_probability, 4)
        }


# ==============================================================================
# 2. EVIDENTIAL FUSION OPERATOR
# ==============================================================================

class EvidentialFusionEngine:
    """
    Implements Dempster's combination rule over subjective opinions with mathematical closure.
    Naturally satisfies Missing Modality Invariance: fusing any opinion with a vacuous
    opinion (u = 1.0) preserves the original opinion exactly.
    """

    @staticmethod
    def combine_two(op1: SubjectiveOpinion, op2: SubjectiveOpinion) -> SubjectiveOpinion:
        # Conflict factor K
        k = (op1.b_phish * op2.b_clean) + (op1.b_clean * op2.b_phish)
        if k >= 0.9999:
            # Extreme conflicting evidence guard
            return SubjectiveOpinion(0.5, 0.5, 0.0, source=f"{op1.source}+{op2.source}")

        denom = 1.0 - k
        b_p = ((op1.b_phish * op2.b_phish) + (op1.b_phish * op2.u) + (op1.u * op2.b_phish)) / denom
        b_c = ((op1.b_clean * op2.b_clean) + (op1.b_clean * op2.u) + (op1.u * op2.b_clean)) / denom
        u = (op1.u * op2.u) / denom

        return SubjectiveOpinion(b_p, b_c, u, source=f"{op1.source}+{op2.source}")

    @classmethod
    def fuse_opinions(cls, opinions: List[SubjectiveOpinion]) -> SubjectiveOpinion:
        """Iteratively fuses multiple evidence opinions."""
        active = [op for op in opinions if op.u < 0.999]
        if not active:
            return SubjectiveOpinion.vacuous("fused_vacuous")

        fused = active[0]
        for op in active[1:]:
            fused = cls.combine_two(fused, op)
        return fused


# ==============================================================================
# 3. SPLIT-CONFORMAL PREDICTOR
# ==============================================================================

# ==============================================================================
# 3. SPLIT-CONFORMAL PREDICTOR (Distribution-Free Risk Control)
# ==============================================================================

class ConformalPredictor:
    """Split-conformal sets and a separate conservative benign-class action gate.

    Coverage and benign Type I bounds require independent calibration observations
    exchangeable with deployment observations and an unchanged scoring pipeline.
    They do not imply FDR control or protection under distribution shift.
    """
    SCHEMA_VERSION = 2

    def __init__(self, alpha_coverage=0.05, alpha_fpr=0.01, *, parameter_path=CONFORMAL_PATH):
        if not 0 < alpha_coverage < 1 or not 0 < alpha_fpr < 1:
            raise ValueError("alpha values must be in (0, 1)")
        self.alpha_coverage, self.alpha_fpr = alpha_coverage, alpha_fpr
        self.parameter_path = parameter_path
        self.q_coverage = self.q_quarantine = float("inf")
        self.is_calibrated = False
        self.pipeline_id = None
        self.calibration_rejection_reason = None
        self._load_parameters()

    def _deployment_admission_rejection(self, parameter_bytes, params):
        """Check production admission attestations and exact artifact integrity.

        These checks do not independently verify labels or establish statistical
        performance. They prevent a matching pipeline name from reactivating
        historical/development thresholds without a corresponding admitted run.
        Explicit research paths and in-memory calibration remain separate.
        """
        if Path(self.parameter_path).resolve() != Path(CONFORMAL_PATH).resolve():
            return None
        evaluation_path = Path(CONFORMAL_PATH).with_name("conformal_evaluation_results.json")
        try:
            evaluation_bytes = evaluation_path.read_bytes()
        except OSError:
            return "deployment_evaluation_unavailable"
        try:
            evaluation = json.loads(evaluation_bytes)
        except (ValueError, UnicodeError):
            return "invalid_deployment_evaluation_record"
        if not isinstance(evaluation, dict):
            return "invalid_deployment_evaluation_record"
        if evaluation.get("schema_version") != self.SCHEMA_VERSION:
            return "deployment_evaluation_schema_mismatch"
        if evaluation.get("pipeline_id") != params["pipeline_id"]:
            return "deployment_evaluation_pipeline_mismatch"
        if evaluation.get("complete") is not True:
            return "deployment_evaluation_incomplete"
        if evaluation.get("development_only", False) is not False:
            return "development_evaluation_not_approved"
        if evaluation.get("independent_label_review_completed") is not True:
            return "independent_label_review_incomplete"
        if evaluation.get("deployment_generalization_established") is not True:
            return "deployment_generalization_unestablished"
        if evaluation.get("calibration_params_sha256") != hashlib.sha256(parameter_bytes).hexdigest():
            return "deployment_calibration_hash_mismatch"
        return None

    def _load_parameters(self):
        # A failed reload must not leave an earlier accepted artifact active.
        self.is_calibrated = False
        self.pipeline_id = None
        self.q_coverage = self.q_quarantine = float("inf")
        self.calibration_rejection_reason = None
        try:
            parameter_bytes = Path(self.parameter_path).read_bytes()
        except OSError:
            self.calibration_rejection_reason = "calibration_parameters_unavailable"
            return
        try:
            params = json.loads(parameter_bytes)
            if not isinstance(params, dict):
                raise ValueError("Invalid parameter record")
            if params.get("schema_version") != self.SCHEMA_VERSION:
                self.calibration_rejection_reason = "calibration_parameter_schema_mismatch"
                return  # Legacy thresholds were calibrated against a different scorer.
            q_coverage = float("inf") if params["q_coverage"] is None else float(params["q_coverage"])
            q_quarantine = float("inf") if params["q_quarantine"] is None else float(params["q_quarantine"])
            alpha_coverage = float(params["alpha_coverage"])
            alpha_fpr = float(params["alpha_fpr"])
            pipeline_id = params["pipeline_id"]
            if not isinstance(pipeline_id, str) or not pipeline_id.strip():
                raise ValueError("Invalid pipeline identity")
            if not (0 < alpha_coverage < 1 and 0 < alpha_fpr < 1):
                raise ValueError("Invalid alpha")
            if any(math.isnan(q) or q < 0 for q in (q_coverage, q_quarantine)):
                raise ValueError("Invalid quantile")
            rejection = self._deployment_admission_rejection(parameter_bytes, params)
            if rejection is not None:
                self.calibration_rejection_reason = rejection
                return
            self.q_coverage, self.q_quarantine = q_coverage, q_quarantine
            self.alpha_coverage, self.alpha_fpr = alpha_coverage, alpha_fpr
            self.pipeline_id = pipeline_id
            self.is_calibrated = True
            self.calibration_rejection_reason = None
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            self.is_calibrated = False
            self.calibration_rejection_reason = "invalid_calibration_parameters"

    @staticmethod
    def _quantile(scores, alpha):
        n = len(scores)
        k = math.ceil((n + 1) * (1 - alpha))
        # The (n+1)th order statistic is +infinity, never clamped to the maximum.
        return float(np.sort(scores)[k - 1]) if 1 <= k <= n else float("inf")

    def save_parameters(self):
        params = {"schema_version": self.SCHEMA_VERSION,
                  "q_coverage": self.q_coverage if math.isfinite(self.q_coverage) else None,
                  "q_quarantine": self.q_quarantine if math.isfinite(self.q_quarantine) else None,
                  "alpha_coverage": self.alpha_coverage, "alpha_fpr": self.alpha_fpr,
                  "pipeline_id": self.pipeline_id,
                  "uncertainty_kind": "assigned_subjective_logic_ignorance"}
        target = Path(self.parameter_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(params, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(target)

    def calibrate(self, val_scores, val_labels, *, pipeline_id, persist=True):
        scores = np.asarray(val_scores, dtype=float)
        labels = np.asarray(val_labels)
        if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
            raise ValueError("Nonempty matching one-dimensional calibration arrays required")
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Scores must be finite probabilities")
        if not np.all(np.isin(labels, [0, 1])) or not pipeline_id:
            raise ValueError("Binary labels and pipeline identity required")
        self.q_coverage = self._quantile(np.where(labels == 1, 1-scores, scores), self.alpha_coverage)
        self.q_quarantine = self._quantile(scores[labels == 0], self.alpha_fpr)
        self.pipeline_id, self.is_calibrated = pipeline_id, True
        self.calibration_rejection_reason = None
        if persist:
            self.save_parameters()

    def predict_set(self, threat_prob, *, pipeline_id=None):
        p = float(threat_prob)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Threat probability must be finite and in [0, 1]")
        if not self.is_calibrated or pipeline_id != self.pipeline_id:
            return {"CLEAN", "PHISHING"}, "SUSPICIOUS", "WARNING_BANNER"
        pred_set = set()
        if p <= self.q_coverage:
            pred_set.add("CLEAN")
        if 1-p <= self.q_coverage:
            pred_set.add("PHISHING")
        # Strict > handles ties conservatively. There is no fallback that bypasses this gate.
        if pred_set == {"PHISHING"} and p > self.q_quarantine:
            return pred_set, "PHISHING", "AUTO_QUARANTINE"
        if pred_set == {"CLEAN"}:
            return pred_set, "CLEAN", "SAFE_DELIVERY"
        # Includes empty sets and phishing singletons that fail the benign action gate.
        return pred_set, "SUSPICIOUS", "WARNING_BANNER"


# Global Singleton
CONFORMAL_PREDICTOR = ConformalPredictor()


# ==============================================================================
# 4. HIGH-LEVEL MULTI-MODAL SYNTHESIS INTERFACE
# ==============================================================================

def synthesize_email_security_opinion(
    content_prob: Optional[float] = None,
    content_conf: float = 0.85,
    url_prob: Optional[float] = None,
    url_conf: float = 0.0,
    intent_prob: Optional[float] = None,
    intent_conf: float = 0.0,
    contradiction_prob: Optional[float] = None,
    contradiction_conf: float = 0.0,
    vision_prob: Optional[float] = None,
    vision_conf: float = 0.0,
    has_hard_threat: bool = False,
    predictor=None,
    pipeline_id=None,
    heuristic_evidence=None
) -> Dict[str, Any]:
    """
    Synthesizes multi-sensor email intelligence into a single evidential verdict.
    Handles missing links, missing intent, vision/quishing, and hard threats with mathematical closure.
    """
    # Until a fitted fusion mapping is independently validated, retain the model's
    # calibrated probability. Legacy auxiliary arguments are informational only.
    evidence = dict(heuristic_evidence or {})
    for name, value in [("url", url_prob), ("intent", intent_prob),
                        ("contradiction", contradiction_prob), ("vision", vision_prob)]:
        if value is not None:
            evidence[name + "_unvalidated_score"] = value
    statistical = predictor or CONFORMAL_PREDICTOR
    if content_prob is None:
        threat_prob = None
        pred_set, verdict, action = {"CLEAN", "PHISHING"}, "SUSPICIOUS", "WARNING_BANNER"
    else:
        threat_prob = float(content_prob)
        pred_set, verdict, action = statistical.predict_set(threat_prob, pipeline_id=pipeline_id)
    basis = "email_model_conformal" if threat_prob is not None else "model_unavailable"
    review_required = (evidence.get("url_risk_score", 0) >= 45 or
                       evidence.get("bec_detected", False) or
                       evidence.get("contradiction_detected", False) or
                       evidence.get("qr_detected", False) or
                       bool(evidence.get("threat_matches")) or has_hard_threat or
                       any(v is not None and v >= .45 for v in (url_prob, intent_prob, contradiction_prob, vision_prob)))
    if review_required and action != "AUTO_QUARANTINE":
        verdict, action, basis = "SUSPICIOUS", "WARNING_BANNER", "separate_evidence_review"
    valid = bool(threat_prob is not None and statistical.is_calibrated and pipeline_id == statistical.pipeline_id)
    if threat_prob is not None and not valid and basis == 'email_model_conformal':
        basis = 'calibration_unavailable_or_pipeline_mismatch'
    return {
        "uncertainty_kind": "not_estimated",
        "calibration_valid": valid,
        "calibration_rejection_reason": None if valid else getattr(statistical, "calibration_rejection_reason", None),
        "final_score": int(round(threat_prob * 100)) if threat_prob is not None else None,
        "threat_probability": threat_prob,
        "verdict": verdict, "action": action,
        "conformal_prediction_set": sorted(pred_set),
        "epistemic_uncertainty": None, "aleatoric_uncertainty": None,
        "belief_phish": None, "belief_clean": None,
        "modality_opinions": [], "heuristic_evidence": evidence,
        "decision_basis": basis,
    }
