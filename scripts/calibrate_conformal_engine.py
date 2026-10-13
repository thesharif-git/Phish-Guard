"""Calibrate and evaluate the shared email probability branch.
No dataset edits or model retraining. Historical test reuse is development
validation, not an untouched final research holdout. External sensors must be
frozen/snapshotted before invoking theoretical exchangeability claims.
"""
import json
import sys
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.scan_engine import predict_email_probabilities, load_ml_model
from services import scan_engine
from training.email_pipeline import validate_email_splits
from services.security_pipeline import security_pipeline_id
from services.conformal_evidential_engine import ConformalPredictor



def score_split(frame, predictor):
    """Calibrate the shared statistical branch, excluding separate review policies."""
    scores, sets, actions = [], [], []
    identity = security_pipeline_id()
    for offset in range(0, len(frame), 256):
        batch = frame.iloc[offset:offset+256]
        # evaluate_email_security calls score_content(body + " " + subject).
        probabilities = predict_email_probabilities([str(t) + " " for t in batch.raw_text])
        for probability in probabilities:
            prediction, _, action = predictor.predict_set(float(probability), pipeline_id=identity)
            scores.append(float(probability)); sets.append(sorted(prediction)); actions.append(action)
        print(f"Scored probability branch {min(offset+256, len(frame))}/{len(frame)}", flush=True)
    return np.asarray(scores), sets, np.asarray(actions)


def evaluation_metrics(labels, scores, sets, actions):
    y = np.asarray(labels)
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    def tier(action):
        mask = actions == action
        return {"count": int(mask.sum()), "phishing": int((mask & (y == 1)).sum()),
                "benign": int((mask & (y == 0)).sum())}
    quarantine, warning, delivery = (tier(a) for a in ("AUTO_QUARANTINE", "WARNING_BANNER", "SAFE_DELIVERY"))
    ratio = lambda n, d: n/d if d else None
    quarantine.update(recall=ratio(quarantine["phishing"], positives),
                      empirical_fdr=ratio(quarantine["benign"], quarantine["count"]),
                      benign_fpr=ratio(quarantine["benign"], negatives))
    warning["additional_warning_recall"] = ratio(warning["phishing"], positives)
    return {"test_samples": len(y), "phishing": positives, "benign": negatives,
            "empirical_set_coverage": float(np.mean([("PHISHING" if label else "CLEAN") in ps for label, ps in zip(y, sets)])),
            "empty_set_count": sum(not ps for ps in sets),
            "roc_auc": float(roc_auc_score(y, scores)), "brier_score": float(brier_score_loss(y, scores)),
            "auto_quarantine": quarantine, "warning_banner": warning, "safe_delivery": delivery}

def run_calibration_and_evaluation(*, development_only=False, output_directory=None):
    # The measured source-transfer failure is an admission failure, not something
    # another legacy quantile fit can repair. Keep research fits away from production.
    if not development_only:
        raise ValueError('Production Email recalibration is blocked pending reviewed source/time generalization. Legacy development fits require explicit development_only=True and cannot overwrite deployed calibration.')
    destination = Path(output_directory) if output_directory else ROOT / 'benchmarks/email_development_calibration' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / 'benchmarks').resolve()):
        raise ValueError('Development calibration output must remain in the benchmarks directory')
    destination.mkdir(parents=True, exist_ok=False)
    path = ROOT / "data/clean_email_corpus.csv"
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    validate_email_splits(frame)
    load_ml_model()
    # Thread oversubscription adds large overhead to these small sparse batches.
    # This changes execution resources, not learned parameters or features.
    for calibrated in scan_engine.ML_MODEL.calibrated_classifiers_:
        estimator = calibrated.estimator
        if hasattr(estimator, 'named_steps'):
            estimator.named_steps['classifier'].set_params(n_jobs=1)
        else:
            estimator.set_params(n_jobs=1)
    val, test = frame[frame.split == "val"], frame[frame.split == "test"]
    identity = security_pipeline_id()
    predictor = ConformalPredictor(parameter_path=destination / "conformal_params.json")
    started = time.perf_counter()
    scores, _, _ = score_split(val, predictor)
    predictor.calibrate(scores, val.label.to_numpy(), pipeline_id=identity, persist=False)
    test_scores, sets, actions = score_split(test, predictor)
    if identity != security_pipeline_id() or dataset_hash != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("Pipeline or dataset changed during calibration")
    results = {"schema_version": 2, "pipeline_id": identity, "dataset_sha256": dataset_hash,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "scoring_and_calibration_seconds": time.perf_counter() - started,
               "execution_threads_per_tree_model": 1,
               "validation_samples": len(val), "test_samples": len(test),
               "dataset_label_scope": "legacy_source_binary_development; spam/phishing separation is not independently adjudicated",
               "independent_label_review_completed": False,
               "deployment_generalization_established": False,
               "evaluation_scope": "shared email probability branch on existing body-only development split; separate heuristic review policies not evaluated",
               "theoretical_claims": "Conditional on independent exchangeable calibration observations and unchanged model inputs; no FDR guarantee; heuristic policies are outside the statistical claim",
               "test_metrics": evaluation_metrics(test.label, test_scores, sets, actions)}
    predictor.save_parameters()
    output = destination / "conformal_evaluation_results.json"
    output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Email development calibration only; production admission is unresolved')
    parser.add_argument('--development-only', action='store_true')
    parser.add_argument('--output-directory')
    arguments = parser.parse_args()
    run_calibration_and_evaluation(development_only=arguments.development_only, output_directory=arguments.output_directory)
