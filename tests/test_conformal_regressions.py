import numpy as np
import pytest
from services.conformal_evidential_engine import ConformalPredictor, SubjectiveOpinion, EvidentialFusionEngine
from services.security_pipeline import numeric_url_frame

@pytest.fixture
def predictor(tmp_path):
    return ConformalPredictor(parameter_path=tmp_path / "params.json")

def test_missing_and_legacy_calibration_abstains(predictor):
    assert predictor.predict_set(.99)[2] == "WARNING_BANNER"

def test_small_calibration_uses_infinite_order_statistic(predictor):
    predictor.calibrate([.1, .9], [0, 1], pipeline_id="frozen", persist=False)
    assert np.isinf(predictor.q_coverage)
    assert np.isinf(predictor.q_quarantine)
    assert predictor.predict_set(.99, pipeline_id="frozen")[2] == "WARNING_BANNER"

def test_ties_cannot_quarantine_and_singleton_does_not_bypass_gate(predictor):
    predictor.is_calibrated = True
    predictor.pipeline_id = "frozen"
    predictor.q_coverage = .2
    predictor.q_quarantine = .95
    assert predictor.predict_set(.95, pipeline_id="frozen") == ({"PHISHING"}, "SUSPICIOUS", "WARNING_BANNER")
    assert predictor.predict_set(.96, pipeline_id="frozen")[2] == "AUTO_QUARANTINE"

def test_empty_set_is_not_safe_delivery(predictor):
    predictor.is_calibrated = True
    predictor.pipeline_id = "frozen"
    predictor.q_coverage = .2
    assert predictor.predict_set(.5, pipeline_id="frozen") == (set(), "SUSPICIOUS", "WARNING_BANNER")

def test_changed_pipeline_abstains(predictor):
    predictor.calibrate([.1]*100 + [.9]*100, [0]*100 + [1]*100, pipeline_id="old", persist=False)
    assert predictor.predict_set(.99, pipeline_id="new")[2] == "WARNING_BANNER"

def test_roundtrip_preserves_quantile_precision(predictor):
    scores = np.linspace(.000001, .333333333, 200)
    predictor.calibrate(scores, np.zeros(200), pipeline_id="frozen")
    loaded = ConformalPredictor(parameter_path=predictor.parameter_path)
    assert loaded.q_quarantine == predictor.q_quarantine
    assert loaded.q_coverage == predictor.q_coverage

@pytest.mark.parametrize("scores,labels", [([float("nan")], [0]), ([1.1], [0]), ([.1], [2]), ([], [])])
def test_invalid_calibration_rejected(predictor, scores, labels):
    with pytest.raises(ValueError):
        predictor.calibrate(scores, labels, pipeline_id="frozen", persist=False)

def test_vacuous_identity():
    opinion = SubjectiveOpinion.from_probability(.8, .85)
    result = EvidentialFusionEngine.combine_two(opinion, SubjectiveOpinion.vacuous())
    assert (result.b_phish, result.b_clean, result.u) == pytest.approx((opinion.b_phish, opinion.b_clean, opinion.u))

def test_url_feature_order_and_failure_visibility():
    class Model:
        feature_names_in_ = np.array(["b", "a"])
    frame = numeric_url_frame(Model(), [{"a": 1, "b": 2}])
    assert frame.columns.tolist() == ["b", "a"]
    assert frame.to_numpy().tolist() == [[2., 1.]]
    with pytest.raises(ValueError):
        numeric_url_frame(Model(), [{"a": 1}])


def test_shared_scorer_preserves_content_probability_and_exposes_failure(monkeypatch, predictor):
    import ast
    import sys
    import types
    from collections import Counter
    from pathlib import Path
    import services.conformal_evidential_engine as fusion
    from utils.email_input_contract import build_email_input, contract_sha256
    # Execute the actual shared scorer with deterministic sensors, without DB/network.
    tree = ast.parse(Path("services/scan_engine.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_email_security")
    namespace = dict(score_content=lambda text: {"ml_probability": .99, "ml_powered": True},
                     score_urls=lambda text: {"score": 0, "inference_errors": []},
                     score_headers=lambda text: {}, build_threat_intel=lambda *args: {"matches": []},
                     extract_urls=lambda text: [], security_pipeline_id=lambda: "frozen",
                     logger=types.SimpleNamespace(debug=lambda *args: None),
                     build_email_input=build_email_input, contract_sha256=contract_sha256, Counter=Counter,
                     EMAIL_INPUT_KIND='legacy_tfidf_structural',
                     STRUCTURED_EMAIL_INPUT_KIND='structured_email_evidence_v1')
    vector = types.ModuleType("services.vector_store")
    vector.search_similar_campaigns = lambda text: {"has_match": False}
    vector.index_phishing_email = lambda *args: None
    bec = types.ModuleType("services.bec_detector")
    bec.predict_bec_intent = lambda *args, **kwargs: {"is_bec": False}
    pillars = types.ModuleType("services.origintrace_pillars")
    class Sensor:
        @staticmethod
        def analyze(**kwargs): return {}
    for name in ["ProvenanceMatrix", "InfrastructureLineage", "SemanticDeception", "EvidentialFusion", "CrossEntityContradictionEngine"]:
        setattr(pillars, name, Sensor)
    for name, module in [("services.vector_store", vector), ("services.bec_detector", bec), ("services.origintrace_pillars", pillars)]:
        monkeypatch.setitem(sys.modules, name, module)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "shared-scorer", "exec"), namespace)
    result = namespace["evaluate_email_security"]("No urgency or links", predictor=predictor)
    assert result["sensor_errors"] == []
    assert result["conformal_res"]["threat_probability"] == pytest.approx(.99)
    namespace["score_urls"] = lambda text: {"score": 0, "inference_errors": ["URL schema mismatch"]}
    result = namespace["evaluate_email_security"]("example", predictor=predictor)
    assert "URL schema mismatch" in result["sensor_errors"]
    assert result["security_action"] == "WARNING_BANNER"
    assert result["conformal_res"]["calibration_valid"] is False
    predictor.is_calibrated = True
    predictor.pipeline_id = "frozen"
    predictor.q_coverage = .2
    predictor.q_quarantine = .95
    result = namespace["evaluate_email_security"]("example", predictor=predictor)
    assert result["security_action"] == "AUTO_QUARANTINE"
    assert result["conformal_res"]["threat_probability"] == .99
    assert result["conformal_res"]["evidence_incomplete"] is True


@pytest.mark.parametrize("p", [.02, .99])
def test_probability_is_unchanged_by_priors_and_auxiliary_scores(predictor, p):
    from services.conformal_evidential_engine import synthesize_email_security_opinion
    result = synthesize_email_security_opinion(content_prob=p, content_conf=.4,
        url_prob=.9, url_conf=.8, intent_prob=.95, intent_conf=.85,
        has_hard_threat=True, predictor=predictor)
    assert result["threat_probability"] == p
    assert result["uncertainty_kind"] == "not_estimated"


def test_separate_evidence_requests_review_without_changing_probability(predictor):
    from services.conformal_evidential_engine import synthesize_email_security_opinion
    predictor.is_calibrated = True; predictor.pipeline_id = "frozen"
    predictor.q_coverage = .2; predictor.q_quarantine = .95
    result = synthesize_email_security_opinion(content_prob=.02, predictor=predictor,
        pipeline_id="frozen", heuristic_evidence={"bec_detected": True, "bec_risk_score": 80})
    assert result["threat_probability"] == .02
    assert result["conformal_prediction_set"] == ["CLEAN"]
    assert result["action"] == "WARNING_BANNER"
    assert result["decision_basis"] == "separate_evidence_review"


@pytest.mark.parametrize("value,expected", [
    ("https://login.accounts.example.co.uk/verify", "example.co.uk"),
    ("www.example.com", "example.com"), ("https://example.com/", "example.com")])
def test_domain_model_receives_registrable_domain(value, expected):
    from services.security_pipeline import domain_model_input
    assert domain_model_input(value) == expected


def test_root_url_reaches_url_specialist_and_domain_is_separate():
    import textwrap
    import types
    from pathlib import Path
    from urllib.parse import urlparse
    import pandas as pd
    from services.security_pipeline import numeric_url_frame, domain_model_input
    source = Path("utils/url_scanner.py").read_text(encoding="utf-8")
    block = source[source.index("    # 4. ML Predictions"):source.index("    # 5. Parallel Multi-Modal")]
    calls = []
    class Model:
        feature_names_in_ = np.array(["a"])
        def __init__(self, name): self.name = name
        def predict_proba(self, frame):
            calls.append(self.name)
            return np.array([[.05, .95]])
    def domain_features(value):
        assert value == "example.co.uk"
        return {"a": 1}
    scan = types.SimpleNamespace(URL_MODEL=Model("url"), DOMAIN_MODEL=Model("domain"),
        extract_url_features=lambda value: {"a": 1}, extract_domain_features=domain_features,
        numeric_url_frame=numeric_url_frame, domain_model_input=domain_model_input)
    namespace = dict(scan_engine=scan, url="https://login.example.co.uk/", pd=pd,
                     urlparse=urlparse, logger=types.SimpleNamespace(error=lambda *args: None))
    exec(textwrap.dedent(block), namespace)
    assert calls == ["url", "domain"]
    assert namespace["url_model_probability"] == .95
