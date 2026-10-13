"""Fit separate research candidates using audited data; never replace live models.

This is a reproducible legacy baseline, not a state-of-the-art or release claim.
No URL labels are promoted to domain labels. The final_holdout split is not used
for training, calibration, threshold selection, or this development evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_curve
from sklearn.pipeline import FeatureUnion, Pipeline

from training.prepared_corpus import load_prepared
from training.url_representation import URLSurfaceView


def metrics(y, probabilities, threshold):
    tn, fp, fn, tp = confusion_matrix(y, probabilities >= threshold, labels=[0, 1]).ravel()
    return {"threshold": threshold, "n": len(y),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "fpr": float(fp / (fp + tn)) if fp + tn else None,
            "accuracy": float((tp + tn) / len(y)),
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0,
            "average_precision": float(average_precision_score(y, probabilities))}


def thresholds(y, probabilities):
    fpr, recall, tau = roc_curve(y, probabilities, drop_intermediate=False)
    selected = {}
    for name, budget in (("strict", 0.005), ("inspect", 0.02)):
        valid = np.flatnonzero((fpr <= budget) & np.isfinite(tau))
        # If no finite threshold qualifies, a value above 1 represents no blocks.
        selected[name] = float(tau[valid[np.argmax(recall[valid])]]) if len(valid) else 1.000001
    finite = np.flatnonzero(np.isfinite(tau))
    selected["balanced"] = float(tau[finite[np.argmax((recall - fpr)[finite])]])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/origintrace_prepared")
    parser.add_argument("--output", type=Path, default=ROOT / "models/audited_candidates/normalized")
    parser.add_argument("--url-view", choices=("scheme-independent", "raw"), default="scheme-independent")
    args = parser.parse_args()
    if args.output.resolve() in {(ROOT / "models/frozen_pipeline").resolve(), (ROOT / "models").resolve()}:
        parser.error("Candidate output must not replace the production model directory")
    data, audit = load_prepared(args.corpus)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "corpus_manifest_sha256": audit["manifest_sha256"],
               "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "dependencies": {name: importlib.metadata.version(name)
                                for name in ("scikit-learn", "numpy", "joblib")},
               "purpose": "development baseline only; no final-holdout evaluation or model promotion",
               "modalities": {}}
    for modality in ("email", "url", "domain"):
        parts = {name: data.get(modality + "_" + name, [])
                 for name in ("train", "calibration", "validation", "development_test", "regression")}
        if not all(parts[name] and {row["label"] for row in parts[name]} == {0, 1}
                   for name in ("train", "calibration", "validation", "development_test")):
            results["modalities"][modality] = {"status": "blocked",
                                                "reason": "Verified task-specific two-class splits are unavailable"}
            continue
        print("Training audited " + modality + " baseline", flush=True)
        # Complementary word/character views; model selection belongs in development.
        views = [("characters", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                               max_features=12000, sublinear_tf=True))]
        if modality == "email":
            views.append(("words", TfidfVectorizer(ngram_range=(1, 2), max_features=12000,
                                                  sublinear_tf=True)))
        steps = [("url_surface", URLSurfaceView())] if modality == "url" and args.url_view == "scheme-independent" else []
        estimator = Pipeline(steps + [
            ("views", FeatureUnion(views)),
            ("classifier", LogisticRegression(C=2.0, solver="liblinear", max_iter=500, random_state=42))
        ])
        text = lambda name: [row["raw_input"] for row in parts[name]]
        labels = lambda name: np.array([row["label"] for row in parts[name]])
        estimator.fit(text("train"), labels("train"))
        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
        calibrated.fit(text("calibration"), labels("calibration"))
        selected = thresholds(labels("validation"), calibrated.predict_proba(text("validation"))[:, 1])
        model_path = args.output / (modality + "_baseline.joblib")
        joblib.dump(calibrated, model_path, compress=3)
        scores = {}
        # Preserve the more independent final holdout for a later frozen comparison.
        for split in ("development_test", "regression"):
            if not parts[split]:
                continue
            p = calibrated.predict_proba(text(split))[:, 1]
            scores[split] = {name: metrics(labels(split), p, tau) for name, tau in selected.items()}
            with (args.output / (modality + "_" + split + "_predictions.jsonl")).open("w", encoding="utf-8") as stream:
                for row, probability in zip(parts[split], p):
                    stream.write(json.dumps({"id": row["id"], "group_id": row["group_id"],
                                             "label": row["label"], "probability": float(probability)}) + "\n")
        results["modalities"][modality] = {"status": "evaluated_development_candidate",
                                            "representation": args.url_view + " URL surface" if modality == "url" else "word and character email views",
                                            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                            "thresholds": selected, "results": scores,
                                            "split_counts": {k: len(v) for k, v in parts.items()}}
    (args.output / "candidate_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved development results to " + str(args.output / "candidate_results.json"), flush=True)


if __name__ == "__main__":
    main()
