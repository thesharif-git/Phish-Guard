"""Additional fixed-model diagnostics; all original external failures remain reported."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from utils.feature_extractor import extract_features
from utils.text_sanitizer import clean_and_mask_email_text
from scripts.audit_email_generalization_root import measure, OUT, OLD


def main():
    decisions = pd.read_csv(OLD / 'decisions.csv')
    external = decisions[decisions.split == 'external'].dropna(subset=['baseline_probability', 'candidate_probability'])
    real_valid = external[external.source_label != 'Spam']
    y = (real_valid.source_label == 'Phishing').astype(int)
    results = {'scope': 'Development diagnostics only; not relabeling, retraining, calibration or replacing the 291-row external benchmark',
        'phishing_vs_valid_excluding_spam_sensitivity_analysis': {name: measure(y, real_valid[name+'_probability']) for name in ('baseline', 'candidate')}}
    from utils.jsonl import read_jsonl
    overlaps = read_jsonl(OUT / 'full_message_overlap_review.jsonl')
    exclude = {str(r['row']) for r in overlaps if r['full_unique_trigram_jaccard'] >= .9}
    tests = decisions[decisions.split == 'test']
    disjoint = tests[~tests.row.astype(str).isin(exclude)]
    results['internal_sensitivity_excluding_14_high_overlap_test_rows'] = {name: measure(disjoint.source_label.astype(int), disjoint[name+'_probability']) for name in ('baseline', 'candidate')}
    # A source archive footer changes no sender, URL, authentication or payload.
    # This is an input sensitivity probe; synthetic mutations are never called real gold emails.
    rows = read_jsonl(OLD / 'external_inputs.jsonl')
    supported_ids = set(external.row.astype(str))
    rows = [r for r in rows if r['review_id'] in supported_ids]
    texts = [r['text'] for r in rows]
    modified = [t + '\nArchive reference: Enron 2001.' for t in texts]
    vectorizer = joblib.load(ROOT / 'models/vectorizer.pkl')
    model = joblib.load(ROOT / 'models/phish_model.pkl')
    for fold in model.calibrated_classifiers_:
        fold.estimator.set_params(n_jobs=1)
    def predict(inputs):
        x = hstack([vectorizer.transform([clean_and_mask_email_text(t) for t in inputs]), csr_matrix(extract_features(inputs))]).tocsr()
        return model.predict_proba(x)[:, 1]
    before, after = predict(texts), predict(modified)
    delta = after-before
    results['archive_footer_probe'] = {'n': len(rows), 'footer': 'Archive reference: Enron 2001.',
        'mean_signed_probability_change': float(delta.mean()), 'mean_absolute_probability_change': float(np.abs(delta).mean()),
        'max_absolute_probability_change': float(np.abs(delta).max()),
        'changed_decisions_at_0_5': int(((before >= .5) != (after >= .5)).sum()),
        'phishing_to_nonphishing_changes': int(((before >= .5) & (after < .5)).sum()),
        'scope': 'Synthetic representation-sensitivity probe, not a validation result or proposed preprocessing fix'}
    (OUT / 'archive_footer_probe.jsonl').write_text(''.join(json.dumps({'id': r['review_id'], 'source_label': r['source_label'], 'before': float(p), 'after': float(q)})+'\n' for r,p,q in zip(rows,before,after)),encoding='utf-8')
    (OUT / 'failure_attribution.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
