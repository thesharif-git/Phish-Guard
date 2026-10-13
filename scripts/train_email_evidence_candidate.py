"""Email-only reviewed-evidence candidates, never automatic deployment.

Unapproved review corpora are rejected before output creation or model fitting.
The fixed arms run serially and never contact URLs, mailboxes or external APIs.
"""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from scipy.stats import beta
from training.email_evidence_model import (
    EMAIL_EVIDENCE_MODEL_INPUT_KIND, EMAIL_INPUT_CONTRACT_VERSION, MODEL_PARAMS,
    TASK_LABELS, contract_source_hashes, email_evidence_estimator,
    library_versions, load_reviewed_email_evidence, predict_email_evidence_probabilities,
)

ARMS = (
    dict(name="word_only", character_features=False, evidence_features=False),
    dict(name="text_only", character_features=True, evidence_features=False),
    dict(name="structured", character_features=True, evidence_features=True),
)
VALIDATION_FPR_BUDGET = .01


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def validate_output(out):
    out = Path(out).resolve()
    try:
        relative = out.relative_to((ROOT / "benchmarks").resolve())
    except ValueError as error:
        raise ValueError("Email research output must be a new workspace benchmarks child") from error
    if not relative.parts or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError("Do not overwrite an existing Email experiment")
    return out


def _validated_arrays(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Require matching binary labels and probabilities")
    if np.isinf(p).any() or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError("Invalid candidate probability")
    return y.astype(int), p


def _interval(k, n):
    if not n:
        return None
    return [0. if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1. if k == n else float(beta.ppf(.975, k + 1, n - k))]


def metrics(y, p, threshold=.5):
    y, p = _validated_arrays(y, p)
    supported = np.isfinite(p)
    result = dict(total_rows=len(y), scored_rows=int(supported.sum()),
                  unsupported_rows=int((~supported).sum()), threshold=float(threshold))
    result["unsupported_by_class"] = {str(label): int(((y == label) & ~supported).sum()) for label in (0, 1)}
    y, p = y[supported], p[supported]
    if not len(y):
        return result
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel())
    result.update(confusion_matrix=[[tn, fp], [fn, tp]], accuracy=(tn + tp) / len(y),
        precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        fpr=fp / (tn + fp) if tn + fp else None, fnr=fn / (tp + fn) if tp + fn else None,
        roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        average_precision=float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
        brier=float(brier_score_loss(y, p)), fpr_95ci=_interval(fp, tn + fp), recall_95ci=_interval(tp, tp + fn))
    ece, bins = 0., []
    for index in range(10):
        selected = (p >= index / 10) & (p < (index + 1) / 10 if index < 9 else p <= 1)
        if selected.any():
            predicted, observed = float(p[selected].mean()), float(y[selected].mean())
            ece += selected.mean() * abs(predicted - observed)
            bins.append(dict(count=int(selected.sum()), mean_raw_probability=predicted, positive_fraction=observed))
    result.update(ece_10_equal_width=float(ece), reliability_bins=bins,
                  calibration_status="uncalibrated_diagnostics_only",
                  interval_assumption="Binomial row independence; campaign/source dependence may invalidate intervals")
    return result


def validation_threshold(y, p, budget=VALIDATION_FPR_BUDGET):
    y, p = _validated_arrays(y, p)
    if not 0 <= budget < 1:
        raise ValueError("Empirical validation budget must be in [0, 1)")
    negatives = p[(y == 0) & np.isfinite(p)]
    if not len(negatives):
        raise ValueError("No supported validation negatives; no threshold can be frozen")
    allowed = math.floor(budget * len(negatives))
    threshold = float(np.nextafter(np.sort(negatives)[::-1][allowed], np.inf))
    assert int((negatives >= threshold).sum()) <= allowed
    return threshold


def _split_report(records, p, threshold):
    y = np.asarray([TASK_LABELS[r["task_label"]] for r in records])
    result = dict(at_0_5=metrics(y, p), at_validation_budget=metrics(y, p, threshold),
                  per_source={}, per_task_label={}, availability_subgroups={})
    for source in sorted(set(row["source"] for row in records)):
        mask = np.asarray([row["source"] == source for row in records])
        result["per_source"][source] = dict(at_0_5=metrics(y[mask], p[mask]),
                                           at_validation_budget=metrics(y[mask], p[mask], threshold))
    for label in sorted(set(row["task_label"] for row in records)):
        mask = np.asarray([row["task_label"] == label for row in records])
        result["per_task_label"][label] = dict(at_0_5=metrics(y[mask], p[mask]),
                                               at_validation_budget=metrics(y[mask], p[mask], threshold))
    for field in ("sender", "headers", "links", "trusted_authentication"):
        for present in (False, True):
            mask = np.asarray([row["email_input"]["availability"][field] == present for row in records])
            if mask.any():
                result["availability_subgroups"][field + ("_present" if present else "_missing")] = dict(
                    at_0_5=metrics(y[mask], p[mask]), at_validation_budget=metrics(y[mask], p[mask], threshold))
    return result


def train_candidate(corpus_path, manifest_path, output):
    # Admission completes before outputs, saved models, or any training action.
    output = validate_output(output)
    records, manifest, admission = load_reviewed_email_evidence(corpus_path, manifest_path)
    source_hashes = contract_source_hashes()
    source_hashes["scripts/train_email_evidence_candidate.py"] = _digest(Path(__file__))
    protected_names = ("models/phish_model.pkl", "models/vectorizer.pkl", "models/url_model.pkl",
                       "models/domain_model.pkl", "models/conformal_params.json", "models/conformal_evaluation_results.json")
    protected = {name: _digest(ROOT / name) for name in protected_names if (ROOT / name).exists()}
    input_hashes = {str(Path(corpus_path).resolve()): _digest(corpus_path),
                    str(Path(manifest_path).resolve()): _digest(manifest_path)}
    protocol = dict(schema_version=3, input_kind=EMAIL_EVIDENCE_MODEL_INPUT_KIND,
        input_contract_version=EMAIL_INPUT_CONTRACT_VERSION,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), corpus_sha256=manifest["corpus_sha256"],
        manifest_sha256=_digest(manifest_path), arms=ARMS, classifier_params=MODEL_PARAMS,
        feature_limits=dict(word=12000, character=16000, identity=8000, min_df=3),
        validation_fpr_budget=VALIDATION_FPR_BUDGET, source_hashes=source_hashes,
        protected_hashes=protected, input_hashes=input_hashes, library_versions=library_versions(),
        admission=admission, automatic_promotion=False, calibration_fitting=False,
        limitations=["Recorded review attestations require actual independent human review",
                     "No trusted receiver authentication provider exists in contract v1",
                     "Future deployment needs separate held-out acceptance and calibration",
                     "Neither validation thresholds nor this training establish an FDR guarantee"])
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "protocol.json", protocol)
    protocol_hash = _digest(output / "protocol.json")
    parts = {split: [row for row in records if row["split"] == split]
             for split in ("train", "val", "source_test", "time_test")}
    result = dict(complete=False, schema_version=3, approved_for_deployment=False,
                  deployment_generalization_established=False, calibration_fitted=False,
                  protocol_sha256=protocol_hash, models={}, paired_comparisons={})
    fitted, all_predictions = {}, {}
    try:
        # Freeze every fitted arm before any test scores are obtained.
        for spec in ARMS:
            _save(output / "run_status.json", dict(stage="fitting", model=spec["name"]))
            model = email_evidence_estimator(**{key: spec[key] for key in ("character_features", "evidence_features")})
            x = [row["email_input"] for row in parts["train"]]
            y = np.asarray([TASK_LABELS[row["task_label"]] for row in parts["train"]])
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x, y)
            convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
            if convergence:
                raise ValueError("Candidate solver did not converge: " + "; ".join(convergence))
            if not model.named_steps["features"].supported(x).all():
                raise ValueError("Approved training contains unsupported feature rows; review representation coverage")
            artifact = dict(schema_version=3, input_kind=EMAIL_EVIDENCE_MODEL_INPUT_KIND,
                input_contract_version=EMAIL_INPUT_CONTRACT_VERSION, model=model,
                approved_for_deployment=False, deployment_generalization_established=False,
                probability_calibrated=False, protocol_sha256=protocol_hash,
                corpus_sha256=manifest["corpus_sha256"], contract_source_hashes=contract_source_hashes(),
                library_versions=protocol["library_versions"], admission=admission)
            path = output / (spec["name"] + ".pkl")
            joblib.dump(artifact, path)
            fitted[spec["name"]] = dict(fit_seconds=time.perf_counter() - started,
                artifact_sha256=_digest(path), artifact_mib=path.stat().st_size / 2**20,
                feature_count=model.named_steps["features"].transform(x[:1]).shape[1],
                convergence_warnings=convergence, n_iter=model.named_steps["classifier"].n_iter_.tolist())
            del model, artifact
            gc.collect()
        _save(output / "fitted_models.json", fitted)
        for spec in ARMS:
            name = spec["name"]
            path = output / (name + ".pkl")
            if _digest(path) != fitted[name]["artifact_sha256"]:
                raise ValueError("Candidate changed after fit freeze")
            model = joblib.load(path)["model"]
            value = dict(**fitted[name], splits={})
            predictions = {}
            threshold = None
            for split in ("val", "source_test", "time_test"):
                part = parts[split]
                inputs = [row["email_input"] for row in part]
                started = time.perf_counter()
                p = predict_email_evidence_probabilities(model, inputs)
                elapsed = time.perf_counter() - started
                if split == "val":
                    threshold = validation_threshold([TASK_LABELS[row["task_label"]] for row in part], p)
                    value["validation_threshold"] = threshold
                    _save(output / (name + "_threshold.json"), dict(threshold=threshold,
                        selected_split="val", empirical_fpr_budget=VALIDATION_FPR_BUDGET,
                        protocol_sha256=protocol_hash, artifact_sha256=fitted[name]["artifact_sha256"]))
                predictions[split] = p
                value["splits"][split] = _split_report(part, p, threshold)
                value["splits"][split]["batch_inference_seconds"] = elapsed
                content, identity = model.named_steps["features"].channel_support(inputs)
                with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                    for row, probability, content_ok, identity_ok in zip(part, p, content, identity):
                        stream.write(json.dumps(dict(model=name, split=split, source=row["source"], row_id=row["id"],
                            task_label=row["task_label"], label=TASK_LABELS[row["task_label"]],
                            probability=float(probability) if np.isfinite(probability) else None,
                            state="SCORED" if np.isfinite(probability) else "UNSUPPORTED_REVIEW_REQUIRED",
                            content_supported=bool(content_ok), identity_link_supported=bool(identity_ok),
                            availability=row["email_input"]["availability"],
                            input_contract_sha256=row["input_contract_sha256"])) + "\n")
            all_predictions[name] = predictions
            result["models"][name] = value
            _save(output / "evaluation_checkpoint.json", result)
            del model
            gc.collect()
        # Only common supported rows are used for model contrasts; the full
        # cohorts above retain every missing score and source-specific failure.
        for split in ("source_test", "time_test"):
            y = np.asarray([TASK_LABELS[row["task_label"]] for row in parts[split]])
            common = np.logical_and.reduce([np.isfinite(all_predictions[spec["name"]][split]) for spec in ARMS])
            result["paired_comparisons"][split] = dict(total_rows=len(y), common_supported_rows=int(common.sum()),
                models={spec["name"]: metrics(y[common], all_predictions[spec["name"]][split][common],
                    result["models"][spec["name"]]["validation_threshold"]) for spec in ARMS})
        if any(_digest(ROOT / name) != value for name, value in protected.items()):
            raise ValueError("Protected production artifact changed during Email training")
        if any(_digest(path) != value for path, value in input_hashes.items()):
            raise ValueError("Reviewed corpus/manifest changed during Email training")
        current_sources = contract_source_hashes()
        current_sources["scripts/train_email_evidence_candidate.py"] = _digest(Path(__file__))
        if current_sources != source_hashes:
            raise ValueError("Frozen Email input/training sources changed during the experiment")
        result.update(complete=True, production_artifacts_unchanged=True,
                      input_artifacts_unchanged=True, source_artifacts_unchanged=True,
                      no_candidate_promoted=True)
        _save(output / "results.json", result)
        _save(output / "run_status.json", dict(stage="completed", models=len(result["models"])))
        return result
    except Exception as error:
        _save(output / "run_status.json", dict(stage="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_candidate(args.corpus, args.manifest, args.output)
