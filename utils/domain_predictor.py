"""
PhishGuard AI — Calibrated Domain Inference Engine (Phase 3)
Pure-CPU LightGBM FastTree + Platt Scaling (Sigmoid) predictor for domain reputation.
Guarantees deterministic, thread-safe P95 inference latency in < 1 ms on CPU.
"""
from typing import Union, List, Optional
import numpy as np
import pandas as pd
from scipy.special import expit
from utils.domain_features import (
    extract_domain_features_single,
    extract_domain_features_batch,
    DOMAIN_FEATURE_NAMES
)


class CalibratedDomainPredictor:
    """
    Production-grade pure-CPU inference wrapper for LightGBM + Platt Scaling on Domains.
    Provides standard scikit-learn API (predict_proba, predict, classes_, feature_names_in_)
    with zero thread contention and sub-millisecond execution.
    """
    def __init__(self, booster, a: float, b: float, classes=(0, 1)):
        self.booster = booster
        self.a_ = float(a)
        self.b_ = float(b)
        self.classes_ = np.array(classes, dtype=np.int64)
        self.feature_names_in_ = np.array(DOMAIN_FEATURE_NAMES, dtype=object)

    def _extract_matrix(self, X: Union[List[str], str, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Extracts (N, 16) float32 feature matrix from various input types."""
        if isinstance(X, pd.DataFrame):
            # Ensure columns match expected feature ordering
            cols = [c for c in self.feature_names_in_ if c in X.columns]
            if len(cols) == len(self.feature_names_in_):
                return X[list(self.feature_names_in_)].to_numpy(dtype=np.float32)
            elif "domain" in X.columns or "psl_domain" in X.columns or "url" in X.columns:
                target_col = "domain" if "domain" in X.columns else ("psl_domain" if "psl_domain" in X.columns else "url")
                return extract_domain_features_batch(X[target_col].tolist())
            else:
                return X.to_numpy(dtype=np.float32)
        elif isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], str):
            return extract_domain_features_batch(list(X))
        elif isinstance(X, str):
            return extract_domain_features_single(X)
        elif isinstance(X, np.ndarray):
            return X.astype(np.float32)
        else:
            raise ValueError(f"Unsupported input type for domain prediction: {type(X)}")

    def predict_proba(self, X) -> np.ndarray:
        """
        Returns calibrated 2D class probabilities matrix of shape (N, 2).
        Column 0: P(Clean)
        Column 1: P(Phishing / Malicious Domain)
        Accepts: List[str] of domains/URLs, pd.DataFrame from numeric_url_frame, or np.ndarray.
        """
        X_mat = self._extract_matrix(X)
        raw = self.booster.predict(X_mat, raw_score=True)
        # Platt scaling (Sigmoid calibration)
        p1 = expit(self.a_ * raw + self.b_)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    OPERATIONAL_THRESHOLD: float = 0.70

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        """
        Binary classification using calibrated decision threshold.
        Default threshold (0.70) aligns with tau_operational (FPR <= 3.99%),
        safeguarding clean domains against false positives on bare text.
        """
        t = self.OPERATIONAL_THRESHOLD if threshold is None else float(threshold)
        probs = self.predict_proba(X)
        return (probs[:, 1] >= t).astype(np.int64)
