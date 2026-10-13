"""Production artifact admission; fixtures do not establish model performance."""
import hashlib
import json

import pytest

from services import conformal_evidential_engine as engine


@pytest.fixture
def production_paths(tmp_path, monkeypatch):
    params = tmp_path / "models" / "conformal_params.json"
    params.parent.mkdir()
    monkeypatch.setattr(engine, "CONFORMAL_PATH", str(params))
    return params, params.with_name("conformal_evaluation_results.json")


def write_params(path, **changes):
    record = {"schema_version": 2, "pipeline_id": "fixture-pipeline",
              "q_coverage": .2, "q_quarantine": .8,
              "alpha_coverage": .05, "alpha_fpr": .01}
    record.update(changes)
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


def write_evaluation(params_path, evaluation_path, **changes):
    record = {"schema_version": 2, "pipeline_id": "fixture-pipeline", "complete": True,
              "independent_label_review_completed": True,
              "deployment_generalization_established": True,
              "calibration_params_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest()}
    record.update(changes)
    evaluation_path.write_text(json.dumps(record), encoding="utf-8")
    return record


def assert_review_fallback(predictor, reason):
    assert not predictor.is_calibrated
    assert predictor.calibration_rejection_reason == reason
    assert predictor.predict_set(.99, pipeline_id="fixture-pipeline") == (
        {"CLEAN", "PHISHING"}, "SUSPICIOUS", "WARNING_BANNER")


def test_matching_pipeline_id_alone_cannot_reactivate_historical_thresholds(production_paths):
    params, _ = production_paths
    write_params(params)
    predictor = engine.ConformalPredictor(parameter_path=params)
    assert_review_fallback(predictor, "deployment_evaluation_unavailable")
    result = engine.synthesize_email_security_opinion(
        content_prob=.99, predictor=predictor, pipeline_id="fixture-pipeline")
    assert result["action"] == "WARNING_BANNER"
    assert result["calibration_valid"] is False
    assert result["calibration_rejection_reason"] == "deployment_evaluation_unavailable"


def test_matching_artifact_and_admission_flags_load_production_thresholds(production_paths):
    params, evaluation = production_paths
    write_params(params)
    write_evaluation(params, evaluation)
    predictor = engine.ConformalPredictor(parameter_path=params)
    assert predictor.is_calibrated
    assert predictor.calibration_rejection_reason is None
    assert predictor.predict_set(.99, pipeline_id="fixture-pipeline")[2] == "AUTO_QUARANTINE"
    assert predictor.predict_set(.01, pipeline_id="fixture-pipeline")[2] == "SAFE_DELIVERY"


@pytest.mark.parametrize("changes,reason", [
    ({"schema_version": 1}, "deployment_evaluation_schema_mismatch"),
    ({"pipeline_id": "historical-pipeline"}, "deployment_evaluation_pipeline_mismatch"),
    ({"complete": False}, "deployment_evaluation_incomplete"),
    ({"development_only": True}, "development_evaluation_not_approved"),
    ({"independent_label_review_completed": False}, "independent_label_review_incomplete"),
    ({"independent_label_review_completed": "true"}, "independent_label_review_incomplete"),
    ({"independent_label_review_completed": None}, "independent_label_review_incomplete"),
    ({"deployment_generalization_established": False}, "deployment_generalization_unestablished"),
    ({"deployment_generalization_established": 1}, "deployment_generalization_unestablished"),
    ({"deployment_generalization_established": None}, "deployment_generalization_unestablished"),
    ({"calibration_params_sha256": None}, "deployment_calibration_hash_mismatch"),
    ({"calibration_params_sha256": "wrong"}, "deployment_calibration_hash_mismatch"),
])
def test_incomplete_or_development_evaluation_is_rejected(production_paths, changes, reason):
    params, evaluation = production_paths
    write_params(params)
    write_evaluation(params, evaluation, **changes)
    assert_review_fallback(engine.ConformalPredictor(parameter_path=params), reason)


@pytest.mark.parametrize("missing", ["complete", "independent_label_review_completed", "deployment_generalization_established", "calibration_params_sha256"])
def test_required_evaluation_attestation_cannot_be_omitted(production_paths, missing):
    params, evaluation = production_paths
    write_params(params)
    record = write_evaluation(params, evaluation)
    del record[missing]
    evaluation.write_text(json.dumps(record), encoding="utf-8")
    predictor = engine.ConformalPredictor(parameter_path=params)
    reasons = {"complete": "deployment_evaluation_incomplete",
               "independent_label_review_completed": "independent_label_review_incomplete",
               "deployment_generalization_established": "deployment_generalization_unestablished",
               "calibration_params_sha256": "deployment_calibration_hash_mismatch"}
    assert_review_fallback(predictor, reasons[missing])


def test_binding_covers_exact_parameter_bytes(production_paths):
    params, evaluation = production_paths
    write_params(params)
    write_evaluation(params, evaluation)
    # Even semantically identical JSON has changed since the evaluated artifact.
    params.write_bytes(params.read_bytes() + b"\n")
    assert_review_fallback(engine.ConformalPredictor(parameter_path=params),
                           "deployment_calibration_hash_mismatch")


def test_failed_reload_cannot_keep_previous_production_admission(production_paths):
    params, evaluation = production_paths
    write_params(params)
    write_evaluation(params, evaluation)
    predictor = engine.ConformalPredictor(parameter_path=params)
    assert predictor.is_calibrated
    evaluation.unlink()
    predictor._load_parameters()
    assert_review_fallback(predictor, "deployment_evaluation_unavailable")


@pytest.mark.parametrize("contents", [b"{broken", b"[]", b"null", b"\xff"])
def test_malformed_evaluation_cannot_activate_production_params(production_paths, contents):
    params, evaluation = production_paths
    write_params(params)
    evaluation.write_bytes(contents)
    assert_review_fallback(engine.ConformalPredictor(parameter_path=params),
                           "invalid_deployment_evaluation_record")


@pytest.mark.parametrize("contents", [b"{broken", b"[]", b"null", b"\xff"])
def test_malformed_parameters_fall_back_to_review(production_paths, contents):
    params, _ = production_paths
    params.write_bytes(contents)
    assert_review_fallback(engine.ConformalPredictor(parameter_path=params),
                           "invalid_calibration_parameters")


def test_custom_development_path_roundtrip_does_not_require_production_admission(production_paths, tmp_path):
    path = tmp_path / "development_params.json"
    predictor = engine.ConformalPredictor(parameter_path=path)
    predictor.calibrate([.01, .02, .98, .99], [0, 0, 1, 1], pipeline_id="development")
    loaded = engine.ConformalPredictor(parameter_path=path)
    assert loaded.is_calibrated
    assert loaded.pipeline_id == "development"
    assert loaded.calibration_rejection_reason is None


def test_explicit_in_memory_calibration_remains_available_without_saved_admission(production_paths):
    params, _ = production_paths
    predictor = engine.ConformalPredictor(parameter_path=params)
    predictor.calibrate([.01, .02, .98, .99], [0, 0, 1, 1],
                        pipeline_id="research", persist=False)
    assert predictor.is_calibrated
    assert predictor.calibration_rejection_reason is None
    assert not params.exists()
