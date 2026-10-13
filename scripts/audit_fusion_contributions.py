"""Paired fusion ablation. Writes audit artifacts only, never production parameters."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from services import conformal_evidential_engine as fusion
from services import scan_engine as engine
from services.security_pipeline import security_pipeline_id
from scripts.calibrate_conformal_engine import evaluation_metrics

OUT = ROOT / "benchmarks/fusion_contributions"
KEYS = ["content_prob", "content_conf", "url_prob", "url_conf", "intent_prob",
        "intent_conf", "contradiction_prob", "contradiction_conf", "vision_prob",
        "vision_conf", "has_hard_threat"]
VARIANTS = ["content_model", "content_prior", "plus_url", "plus_bec",
            "plus_contradiction", "plus_vision", "full_fusion"]


def variant_scores(inputs):
    """Use the actual fusion function; change only which recorded inputs it receives."""
    scores = {"content_model": inputs["content_prob"]}
    staged = dict(content_prob=inputs["content_prob"], content_conf=inputs["content_conf"])
    groups = [("content_prior", []), ("plus_url", ["url_prob", "url_conf"]),
              ("plus_bec", ["intent_prob", "intent_conf"]),
              ("plus_contradiction", ["contradiction_prob", "contradiction_conf"]),
              ("plus_vision", ["vision_prob", "vision_conf"]),
              ("full_fusion", ["has_hard_threat"])]
    for name, keys in groups:
        staged.update({key: inputs[key] for key in keys})
        scores[name] = ORIGINAL_SYNTHESIS(**staged)["threat_probability"]
    return scores


ORIGINAL_SYNTHESIS = fusion.synthesize_email_security_opinion


def analyze(frame, records, manifest):
    metrics, decisions, changes = {}, {}, {}
    test_records = [r for r in records if r["split"] == "test"]
    val_records = [r for r in records if r["split"] == "val"]
    val_y = np.array([r["label"] for r in val_records])
    test_y = np.array([r["label"] for r in test_records])
    matrix = {name: np.array([r["variants"][name] for r in test_records]) for name in VARIANTS}
    for name in VARIANTS:
        predictor = fusion.ConformalPredictor(parameter_path=OUT / "unused_parameters.json")
        val = np.array([r["variants"][name] for r in val_records])
        predictor.calibrate(val, val_y, pipeline_id=name, persist=False)
        predictions = [predictor.predict_set(p, pipeline_id=name) for p in matrix[name]]
        sets = [sorted(item[0]) for item in predictions]
        actions = np.array([item[2] for item in predictions])
        result = evaluation_metrics(test_y, matrix[name], sets, actions)
        # Also compare score-only recall at the same validation benign Type I budget.
        selected = matrix[name] > predictor.q_quarantine
        result["score_only_at_validation_fpr_budget"] = {
            "budget": .01, "threshold": predictor.q_quarantine,
            "test_recall": float(selected[test_y == 1].mean()),
            "test_fpr": float(selected[test_y == 0].mean())}
        result["quarantine_precision"] = (1-result["auto_quarantine"]["empirical_fdr"]
                                           if result["auto_quarantine"]["count"] else None)
        result["warning_volume_fraction"] = result["warning_banner"]["count"] / len(test_y)
        metrics[name], decisions[name] = result, actions.tolist()
    for before, after in zip(VARIANTS, VARIANTS[1:]):
        delta = matrix[after]-matrix[before]
        changes[after] = {"score_changed": int(np.sum(abs(delta) > 1e-12)),
                          "mean_probability_change": float(delta.mean()),
                          "brier_change": metrics[after]["brier_score"]-metrics[before]["brier_score"],
                          "action_changed": sum(a != b for a,b in zip(decisions[before], decisions[after]))}
    review = []
    for i, record in enumerate(test_records):
        action = decisions["full_fusion"][i]
        if (record["label"] == 0 and action == "AUTO_QUARANTINE") or (record["label"] == 1 and action == "SAFE_DELIVERY"):
            row = frame.loc[record["row_id"]]
            review.append({**record, "actions": {name: decisions[name][i] for name in VARIANTS},
                           "source": str(row.source), "campaign_cluster_id": str(row.campaign_cluster_id),
                           "excerpt": str(row.raw_text)[:1600]})
    timings = {}
    for sensor in sorted({key for r in records for key in r["timings_ms"]}):
        samples = [r["timings_ms"].get(sensor, 0) for r in records[1:]]
        timings[sensor] = {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples,95))}
    results = {"manifest": manifest, "complete": True, "metrics": metrics,
               "incremental_changes": changes, "timings": timings, "error_count": len(review),
               "limitations": ["Existing development split, not a new final holdout.",
                   "Paired comparisons hold observed sensor evidence fixed; no reduced-pipeline throughput claim.",
                   "Body-only corpus cannot evaluate sender/header or attachment-dependent sensors fully.",
                   "Fresh live sensor observations may differ from the previous run.",
                   "Validation-selected equal nominal FPR budgets do not force equal realized test FPR."]}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "error_review.json").write_text(json.dumps(review, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Controlled fusion comparison", "", "Dataset and trained models were unchanged. No production calibration was written.", "",
             "| Variant | AUC | Brier | Quarantine recall | Benign FPR | Warning volume |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        q = m["auto_quarantine"]
        lines.append(f"| {name} | {m['roc_auc']:.4f} | {m['brier_score']:.4f} | {100*q['recall']:.2f}% | {100*q['benign_fpr']:.2f}% | {100*m['warning_volume_fraction']:.2f}% |")
    lines += ["", "Incremental changes and measured stage timing are in results.json. Error excerpts and per-stage decisions are in error_review.json.", "", *results["limitations"]]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/clean_email_corpus.csv"
    manifest = {"pipeline_id": security_pipeline_id(), "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "body-only val/test; sequential CPU with one native worker"}
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Audit inputs changed; use a new output directory rather than mix captures")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frame = pd.read_csv(path)
    capture = OUT / "sensor_capture.jsonl"
    records = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()] if capture.exists() else []
    done = {r["row_id"] for r in records}
    engine.load_ml_model()
    for model in [engine.ML_MODEL, engine.URL_MODEL, engine.DOMAIN_MODEL]:
        for classifier in getattr(model, "calibrated_classifiers_", []):
            classifier.estimator.set_params(n_jobs=1)
    from services import bec_detector, vector_store
    current = {}
    def timed(original, name):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                current.setdefault("timings_ms", {})[name] = (time.perf_counter()-start)*1000
        return call
    for name in ["score_content", "score_urls", "score_headers"]:
        setattr(engine, name, timed(getattr(engine,name), name))
    bec_detector.predict_bec_intent = timed(bec_detector.predict_bec_intent, "bec_intent")
    vector_store.search_similar_campaigns = timed(vector_store.search_similar_campaigns, "campaign_similarity")
    def capture_synthesis(**kwargs):
        current["inputs"] = {k: kwargs[k] for k in KEYS}
        return ORIGINAL_SYNTHESIS(**kwargs)
    fusion.synthesize_email_security_opinion = capture_synthesis
    selected = frame[frame.split.isin(["val", "test"])]
    for row_id, row in selected.iterrows():
        if row_id in done:
            continue
        current.clear()
        start = time.perf_counter()
        result = engine.evaluate_email_security(str(row.raw_text))
        if result["sensor_errors"]:
            raise RuntimeError(f"Row {row_id}: {result['sensor_errors']}")
        current["timings_ms"]["full_pipeline"] = (time.perf_counter()-start)*1000
        variants = variant_scores(current["inputs"])
        if abs(variants["full_fusion"]-result["conformal_res"]["threat_probability"]) > 1e-12:
            raise AssertionError("Recorded-input replay disagrees with production")
        record = {"row_id": int(row_id), "split": row.split, "label": int(row.label),
                  "inputs": current["inputs"], "variants": variants, "timings_ms": current["timings_ms"],
                  "signals": {"content": result["content_result"].get("flags", []),
                              "bec": result["bec_result"].get("signals", []),
                              "contradiction": result["contra_result"].get("signals", []),
                              "threat_matches": result["threat_result"].get("matches", [])}}
        with capture.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False)+"\n")
        records.append(record)
        if len(records) % 50 == 0:
            print(f"Captured {len(records)}/{len(selected)}", flush=True)
    if manifest["pipeline_id"] != security_pipeline_id() or manifest["dataset_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("Inputs changed during audit")
    analyze(frame, records, manifest)


if __name__ == "__main__":
    main()
