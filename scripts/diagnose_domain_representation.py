"""Bounded domain representation comparison; never writes production artifacts.

Uses existing development splits, not a fresh generalization benchmark.
Fits text features on training only; selects regularization and thresholds on
validation only, then evaluates the selected candidate on development test.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from training.train_url_and_domain_models import extract_domain_features
from services.security_pipeline import numeric_url_frame

OUT = ROOT / 'benchmarks/domain_representation'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def threshold(y, p):
    fpr, tpr, values = roc_curve(y, p, drop_intermediate=False)
    valid = np.flatnonzero((fpr <= .01) & np.isfinite(values))
    return float(values[valid[np.argmax(tpr[valid])]]) if len(valid) else 1.000001

def metrics(y, p, tau):
    pred = p >= tau
    return dict(auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                threshold=tau, recall=float(pred[y == 1].mean()),
                fpr=float(pred[y == 0].mean()), n=len(y))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / 'data/clean_domain_corpus.csv'
    model_path = ROOT / 'models/domain_model.pkl'
    before = {str(p.relative_to(ROOT)): digest(p) for p in (data_path, model_path)}
    df = pd.read_csv(data_path)
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ('train', 'val', 'test')}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(parts[left].psl_domain) & set(parts[right].psl_domain):
            raise ValueError('Domain identity leakage')
    if df.psl_domain.duplicated().any():
        raise ValueError('Duplicate domain rows')
    print('Loading baseline and extracting numeric features', flush=True)
    base = joblib.load(model_path)
    for c in base.calibrated_classifiers_:
        c.estimator.set_params(n_jobs=1)
    baseline = {}
    for s in ('val', 'test'):
        frame = numeric_url_frame(base, [extract_domain_features(d) for d in parts[s].psl_domain])
        baseline[s] = base.predict_proba(frame)[:, 1]
    train, cal = train_test_split(parts['train'], test_size=.2, random_state=42,
                                 stratify=parts['train'].label)
    candidates = []
    fitted = []
    for strength in (.1, 1.0, 10.0):
        print(f'Training character candidate C={strength}', flush=True)
        estimator = Pipeline([
            ('characters', TfidfVectorizer(analyzer='char', ngram_range=(2, 5),
                                          min_df=2, max_features=30000, sublinear_tf=True)),
            ('classifier', LogisticRegression(C=strength, solver='liblinear', max_iter=500,
                                              random_state=42))])
        estimator.fit(train.psl_domain, train.label)
        model = CalibratedClassifierCV(FrozenEstimator(estimator), method='sigmoid')
        model.fit(cal.psl_domain, cal.label)
        vp = model.predict_proba(parts['val'].psl_domain)[:, 1]
        result = metrics(parts['val'].label.to_numpy(), vp, threshold(parts['val'].label, vp))
        candidates.append(dict(C=strength, validation=result))
        fitted.append(model)
    # Predeclared selection: recall at <=1% empirical validation FPR; AUC tie-break.
    selected = max(range(len(candidates)), key=lambda i: (
        candidates[i]['validation']['recall'], candidates[i]['validation']['auc']))
    winner = fitted[selected]
    yp = winner.predict_proba(parts['test'].psl_domain)[:, 1]
    y = parts['test'].label.to_numpy()
    chosen = metrics(y, yp, candidates[selected]['validation']['threshold'])
    old = metrics(y, baseline['test'], threshold(parts['val'].label, baseline['val']))
    started = time.perf_counter()
    for _ in range(100):
        winner.predict_proba([parts['test'].psl_domain.iloc[0]])
    latency = (time.perf_counter() - started) * 10
    after = {str(p.relative_to(ROOT)): digest(p) for p in (data_path, model_path)}
    if before != after:
        raise RuntimeError('Inputs changed during comparison')
    joblib.dump(winner, OUT / 'character_candidate.joblib', compress=3)
    result = dict(complete=True, scope='existing development corpus; labels not independently verified',
                  input_hashes=before, split_counts={s: len(p) for s, p in parts.items()},
                  candidates=candidates, selected_C=candidates[selected]['C'], baseline_test=old,
                  candidate_test=chosen, candidate_single_mean_ms=latency,
                  production_model_changed=False)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
