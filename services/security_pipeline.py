"""Version binding for calibration and numeric URL model inputs."""
import hashlib
from importlib.metadata import version, PackageNotFoundError
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def security_pipeline_id():
    paths = [ROOT / p for p in (
        "services/scan_engine.py", "services/conformal_evidential_engine.py",
        "services/security_pipeline.py", "services/bec_detector.py",
        "services/domain_scan_policy.py",
        "services/web_evidence_policy.py",
        "utils/email_parser.py", "services/inbox_connector.py", "routers/api.py",
        "utils/email_authentication.py",
        "services/origintrace_pillars.py", "services/vision_scanner.py",
        "services/vector_store.py", "utils/url_scanner.py", "utils/threat_intel.py",
        "utils/feature_extractor.py", "utils/text_sanitizer.py", "utils/language_detector.py", "core/config.py",
        "training/email_pipeline.py",
        "utils/email_input_contract.py", "utils/email_candidate_text.py", "utils/jsonl.py",
        "training/email_evidence_model.py", "training/email_evidence.py",
        "models/phish_model.pkl", "models/vectorizer.pkl", "models/url_model.pkl",
        "models/domain_model.pkl" )]
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.exists())
    return _fingerprint(signature)

@lru_cache(maxsize=2)
def _fingerprint(signature):
    h = hashlib.sha256()
    for package, package_version in _scoring_runtime_versions():
        h.update((package + '=' + package_version).encode())
    for name, _, _ in signature:
        h.update(Path(name).relative_to(ROOT).as_posix().encode())
        with open(name, "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _scoring_runtime_versions():
    values = []
    for package in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'joblib',
                    'beautifulsoup4', 'tldextract'):
        try:
            values.append((package, version(package)))
        except PackageNotFoundError:
            values.append((package, 'unavailable'))
    return tuple(values)


def validate_structured_email_artifact(artifact):
    """Admit a future reviewed bundle only when its input implementation matches.

    Research exports are explicitly unapproved. Merely being schema 3 does not
    authorize deployment or establish external generalization/calibration.
    """
    from utils.email_input_contract import EMAIL_INPUT_CONTRACT_VERSION
    from training.email_evidence_model import (
        EMAIL_EVIDENCE_MODEL_INPUT_KIND, contract_source_hashes, library_versions,
    )
    expected = {'schema_version': 3, 'input_kind': EMAIL_EVIDENCE_MODEL_INPUT_KIND,
                'input_contract_version': EMAIL_INPUT_CONTRACT_VERSION}
    if not isinstance(artifact, dict) or any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported structured Email artifact contract')
    if (artifact.get('approved_for_deployment') is not True or
            artifact.get('deployment_generalization_established') is not True):
        raise ValueError('Structured Email artifact lacks deployment approval and established generalization')
    if artifact.get('contract_source_hashes') != contract_source_hashes(ROOT):
        raise ValueError('Structured Email artifact source hashes do not match current scoring inputs')
    if artifact.get('library_versions') != library_versions():
        raise ValueError('Structured Email artifact runtime versions do not match')
    # Retain bindings to the independently reviewed training protocol/corpus.
    import re
    for key in ('protocol_sha256', 'corpus_sha256'):
        if not isinstance(artifact.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', artifact[key]):
            raise ValueError('Structured Email artifact lacks a valid ' + key)
    model = artifact.get('model')
    classes = np.asarray(getattr(model, 'classes_', []))
    if classes.shape != (2,) or not np.array_equal(classes, [0, 1]):
        raise ValueError('Structured Email model requires ordered binary 0/1 classes')
    steps = getattr(model, 'named_steps', {})
    if not callable(getattr(steps.get('features'), 'supported', None)) or not callable(getattr(model, 'predict_proba', None)):
        raise ValueError('Structured Email model lacks its contract feature transform')
    return model

def numeric_url_frame(model, records):
    frame = pd.DataFrame(records)
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        calibrated = getattr(model, "calibrated_classifiers_", [])
        if calibrated:
            names = getattr(calibrated[0].estimator, "feature_name_", None)
    if names is not None:
        names = list(names)
        if set(names) != set(frame.columns):
            raise ValueError("URL feature names do not match the trained model")
        frame = frame.loc[:, names]
    else:
        # Refuse to infer ordering for a nameless artifact.
        raise ValueError("URL model lacks a named feature schema")
    frame = frame.astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite URL model features")
    return frame


def domain_model_input(value):
    """Registrable domain contract used by the domain training corpus; no PSL fetch."""
    from urllib.parse import urlsplit
    import tldextract
    parsed = urlsplit(value if "://" in value else "https://" + value)
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("Missing domain hostname")
    extracted = _offline_psl()(host)
    return extracted.top_domain_under_public_suffix or host

@lru_cache(maxsize=1)
def _offline_psl():
    import tldextract
    return tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
