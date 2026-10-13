"""Email-only controlled retraining. Source labels stay source labels.

No training admission attestation or production artifact is fabricated. External
comparison seeds are used only for a separately disclosed research evaluation,
never added to training. No URLs, mailboxes, or remote services are contacted.
"""
import hashlib
import json
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from training.email_pipeline import email_estimator, grouped_email_folds, validate_email_splits
from utils.text_sanitizer import clean_and_mask_email_text
from utils.email_parser import visible_html_text
from services.conformal_evidential_engine import ConformalPredictor
from scripts.calibrate_conformal_engine import evaluation_metrics
from services import scan_engine as engine
from services.security_pipeline import security_pipeline_id

OUT = ROOT / 'benchmarks/email_retraining_20261003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    target = OUT / name
    pending = target.with_suffix(target.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(target)


def status(stage, **kwargs):
    save('run_status.json', {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **kwargs})
    print(stage + ' ' + json.dumps(kwargs), flush=True)


fold_number = 0
def progress_callback(env):
    global fold_number
    if env.iteration == 0:
        fold_number += 1
    if env.iteration % 100 == 0 or env.iteration + 1 == env.end_iteration:
        status('training', fold=fold_number, iteration=env.iteration + 1, iterations=env.end_iteration)


def metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'rows': len(y), 'positive': int(y.sum()), 'negative': int((y == 0).sum()),
            'accuracy': float(accuracy_score(y, pred)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            'pr_auc_average_precision': float(average_precision_score(y, p)) if y.sum() else None,
            'brier': float(brier_score_loss(y, p)),
            'threshold_0_5': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}}


def probability_batch(texts, model=None):
    output = []
    for offset in range(0, len(texts), 128):
        batch = texts[offset:offset + 128]
        values = engine.predict_email_probabilities(batch) if model is None else model.predict_proba(batch)[:, 1]
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Invalid candidate probabilities')
        output.extend(map(float, values))
    return np.asarray(output)


def calibrated_actions(y, p, predictor, identity):
    sets, actions = [], []
    for value in p:
        ps, _, action = predictor.predict_set(float(value), pipeline_id=identity)
        sets.append(ps)
        actions.append(action)
    return evaluation_metrics(y, p, sets, np.array(actions)), actions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Experiment already frozen; use its existing artifacts, do not overwrite or refit after test inspection')
    protected_names = ['data/clean_email_corpus.csv', 'models/phish_model.pkl', 'models/vectorizer.pkl',
                       'models/url_model.pkl', 'models/domain_model.pkl', 'models/conformal_params.json', 'models/conformal_evaluation_results.json']
    protected = {p: sha(ROOT / p) for p in protected_names}
    df = pd.read_csv(ROOT / 'data/clean_email_corpus.csv', keep_default_na=False)
    validate_email_splits(df)
    original = json.loads((ROOT / 'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json').read_text(encoding='utf-8'))
    from utils.jsonl import read_jsonl
    inventory = read_jsonl(ROOT / 'benchmarks/email_comparison_source_review/comparison_candidates.jsonl')
    external, seen = [], set()
    old_masked = set(df.raw_text.map(clean_and_mask_email_text))
    exclusions = []
    for record in inventory:
        row = original[record['source_row_index']]
        body, subject = row['Body'], row.get('Subject') or ''
        # Preserve provenance; visible HTML uses the same Email ingestion helper.
        if any(tag in body.lower() for tag in ('<html', '<body', '<p>', '<div', '<a ')):
            body = visible_html_text(body)
        text = body + ' ' + subject
        masked = clean_and_mask_email_text(text)
        reason = 'insufficient_text' if len(masked) <= 5 else 'exact_existing_overlap' if masked in old_masked else 'exact_external_repeat' if masked in seen else None
        if reason:
            exclusions.append({'review_id': record['review_id'], 'reason': reason})
            continue
        seen.add(masked)
        external.append({'review_id': record['review_id'], 'text': text, 'body': body, 'subject': subject,
                         'sender': row.get('Sender') or '', 'html_body': row['Body'] if body != row['Body'] else '',
                         'source_label': row['Type'], 'label': int(row['Type'] == 'Phishing'),
                         'source_claimed_year': row['Year'], 'label_authority': 'PhishFuzzer author-reported Manual original seed',
                         'independent_review_by_this_project': False})
    protocol = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'protected_hashes': protected,
        'purpose': 'Controlled correction of preprocessing isolation on legacy source labels, not certified phishing/BEC retraining',
        'candidate': 'Same LightGBM hyperparameters and 5 grouped sigmoid calibration folds; vocabulary/IDF fitted inside each fold',
        'external_role': 'unseen-to-model research comparison using author labels, no training or tuning on these rows',
        'external_count': len(external), 'external_classes': dict(Counter(r['source_label'] for r in external)),
        'external_exclusions': exclusions, 'external_class_mapping': {'Phishing': 1, 'Spam': 0, 'Valid': 0},
        'external_sha256': hashlib.sha256(json.dumps(external, sort_keys=True).encode()).hexdigest(),
        'source_sha256': sha(ROOT / 'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json'),
        'external_claimed_year_class_counts': dict(Counter(str((r['source_claimed_year'], r['source_label'])) for r in external)),
        'selection': 'No test-based hyperparameter search; evaluate one correction candidate against frozen existing model',
        'production_promotion': 'Not automatic. Candidate remains unapproved; task-label and operational validation are not replaced by this experiment',
        'alpha_coverage': .05, 'alpha_benign_fpr': .01, 'baseline_threshold': .5,
        'limitations': ['Legacy labels can conflate spam and phishing', 'External author labels/years are not independent adjudications',
                       'External JSON is not original MIME or verified receiver authentication', 'Explicit external commercial reuse terms not established',
                       'Repeatedly inspected old splits are development measurements', 'Recorded groups do not prove true independent campaigns']}
    save('protocol.json', protocol)
    (OUT / 'external_inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in external), encoding='utf-8')
    status('data_contract_validated', legacy_rows=len(df), external_rows=len(external))
    engine.load_ml_model()
    baseline = engine.ML_MODEL
    for calibrated in baseline.calibrated_classifiers_:
        calibrated.estimator.set_params(n_jobs=1)
    base = clone(baseline.calibrated_classifiers_[0].estimator).set_params(n_jobs=1, verbosity=-1)
    train = df[df.split == 'train'].reset_index(drop=True)
    texts = (train.raw_text + ' ').tolist()
    folds = grouped_email_folds(texts, train.label.to_numpy(), train.campaign_cluster_id.to_numpy())
    candidate = CalibratedClassifierCV(estimator=email_estimator(base), method='sigmoid', cv=folds, n_jobs=1)
    start = time.perf_counter()
    candidate.fit(texts, train.label.to_numpy(), classifier__callbacks=[progress_callback])
    fit_seconds = time.perf_counter() - start
    status('fit_completed', fit_seconds=fit_seconds)
    artifact = {'schema_version': 2, 'input_kind': 'raw_email_model_text', 'approved_for_deployment': False,
                'model': candidate, 'label_scope': 'legacy_source_binary_development', 'independent_label_review_completed': False,
                'corpus_sha256': protected['data/clean_email_corpus.csv'], 'protocol_sha256': sha(OUT / 'protocol.json')}
    joblib.dump(artifact, OUT / 'candidate_email_model.pkl')
    save('training_metadata.json', {'fit_seconds': fit_seconds, 'training_rows': len(train),
        'folds': len(folds), 'candidate_sha256': sha(OUT / 'candidate_email_model.pkl')})
    from scripts.resume_email_retraining_evaluation import main as evaluate_saved_candidate
    evaluate_saved_candidate()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    try:
        main()
    except Exception as e:
        if OUT.exists():
            status('failed', error=repr(e))
        raise
