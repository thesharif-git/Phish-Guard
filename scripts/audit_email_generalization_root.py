"""Fixed-model Email diagnostics. Never trains, calibrates, or promotes a model."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import hashlib
import html
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
from utils.feature_extractor import extract_features
from utils.text_sanitizer import clean_and_mask_email_text

OUT = ROOT / 'benchmarks/email_generalization_root_20261003'
OLD = ROOT / 'benchmarks/email_retraining_20261003'
STRUCTURAL = ['url_count', 'has_ip_url', 'has_suspicious_tld', 'has_shortener',
              'urgent_words', 'exclamation_count', 'dollar_signs', 'all_caps_words',
              'brand_keywords', 'has_html', 'html_link_count', 'text_length',
              'word_count', 'avg_word_len', 'question_marks', 'comma_count']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def progress(stage):
    save('run_status.json', {'stage': stage, 'time': time.time()})
    print(stage, flush=True)


def measure(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p)
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= .5, labels=[0, 1]).ravel())
    ece = 0.
    bins = []
    for lo in np.arange(10) / 10:
        selected = (p >= lo) & (p < lo + .1 if lo < .9 else p <= 1)
        if selected.any():
            observed, predicted = float(y[selected].mean()), float(p[selected].mean())
            ece += selected.mean() * abs(observed - predicted)
            bins.append({'count': int(selected.sum()), 'mean_probability': predicted, 'observed_phishing_fraction': observed})
    return {'n': len(y), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'accuracy': (tn + tp) / len(y), 'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'fpr': fp / (tn + fp) if tn + fp else None, 'fnr': fn / (tp + fn) if tp + fn else None,
            'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            'brier': float(brier_score_loss(y, p)), 'ece_10_equal_width': float(ece), 'reliability_bins': bins}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        raise RuntimeError('Do not overwrite the completed diagnostic snapshot')
    protected_names = ['data/clean_email_corpus.csv', 'models/phish_model.pkl', 'models/vectorizer.pkl',
                       'models/url_model.pkl', 'models/domain_model.pkl', 'models/conformal_params.json',
                       'models/conformal_evaluation_results.json', 'benchmarks/email_retraining_20261003/candidate_email_model.pkl']
    protected = {name: digest(ROOT / name) for name in protected_names}
    save('protocol.json', {'scope': 'Previously inspected development data; causal representation probes, not a new holdout',
        'protected': protected, 'seed': 20261003, 'counterfactual': 'HTML wrapper only, identical canonical TF-IDF text required',
        'nearest_neighbor': 'Train-fitted word trigrams on full canonical messages; >=.95 similarity flags review, not proven campaigns',
        'model_changes': False, 'retraining': False, 'calibration': False})
    progress('loading fixed artifacts')
    frame = pd.read_csv(ROOT / 'data/clean_email_corpus.csv', keep_default_na=False)
    from utils.jsonl import read_jsonl
    external = read_jsonl(OLD / 'external_inputs.jsonl')
    decisions = pd.read_csv(OLD / 'decisions.csv')
    supported = decisions[decisions.split == 'external'].dropna(subset=['baseline_probability', 'candidate_probability'])
    supported_ids = set(supported.row.astype(str))
    external = [x for x in external if x['review_id'] in supported_ids]
    vectorizer = joblib.load(ROOT / 'models/vectorizer.pkl')
    baseline = joblib.load(ROOT / 'models/phish_model.pkl')
    candidate = joblib.load(OLD / 'candidate_email_model.pkl')['model']
    for model in (baseline, candidate):
        for fold in model.calibrated_classifiers_:
            est = fold.estimator
            (est.named_steps['classifier'] if hasattr(est, 'named_steps') else est).set_params(n_jobs=1)
    def matrix(texts):
        return hstack([vectorizer.transform([clean_and_mask_email_text(t) for t in texts]), csr_matrix(extract_features(texts))]).tocsr()
    results = {'complete': False, 'source_label_counts': frame.groupby(['source', 'label']).size().reset_index().to_dict('records'),
               'source_years_and_headers': 'Not preserved in the legacy training CSV', 'metrics': {}, 'representation_probes': {}, 'attribution': {}}
    for split in ('test', 'external'):
        part = decisions[decisions.split == split].dropna(subset=['baseline_probability', 'candidate_probability'])
        y = (part.source_label == 'Phishing').astype(int) if split == 'external' else part.source_label.astype(int)
        results['metrics'][split] = {name: measure(y, part[name + '_probability']) for name in ('baseline', 'candidate')}
    progress('probing raw-format dependence with identical canonical content')
    probe_rows = [{'id': r['review_id'], 'text': r['text'], 'label': r['label'], 'scope': 'external'} for r in external]
    sample = frame[frame.split == 'test'].sample(n=300, random_state=20261003)
    probe_rows += [{'id': str(i), 'text': r.raw_text, 'label': int(r.label), 'scope': 'internal_sample'} for i, r in sample.iterrows()]
    # Wrapping already-HTML source text would escape tags into visible text; only keep invariant probes.
    probes = []
    for row in probe_rows:
        wrapped = '<html><body>' + row['text'] + '</body></html>'
        if clean_and_mask_email_text(wrapped) == clean_and_mask_email_text(row['text']):
            probes.append(dict(row, wrapped=wrapped))
    texts, wrapped = [r['text'] for r in probes], [r['wrapped'] for r in probes]
    original_x, wrapped_x = matrix(texts), matrix(wrapped)
    # The diagnostic must change only structural columns; TF-IDF is byte-for-byte identical.
    assert (original_x[:, :len(vectorizer.vocabulary_)] != wrapped_x[:, :len(vectorizer.vocabulary_)]).nnz == 0
    audit_rows = []
    for name, model in [('baseline', baseline), ('candidate', candidate)]:
        before = model.predict_proba(original_x if name == 'baseline' else texts)[:, 1]
        after = model.predict_proba(wrapped_x if name == 'baseline' else wrapped)[:, 1]
        by_scope = {}
        for scope in ('external', 'internal_sample'):
            selected = np.array([r['scope'] == scope for r in probes])
            delta = after[selected] - before[selected]
            by_scope[scope] = {'n': int(selected.sum()), 'mean_absolute_probability_change': float(np.abs(delta).mean()),
                'max_absolute_probability_change': float(np.abs(delta).max()),
                'decisions_changed_at_0_5': int(((before[selected] >= .5) != (after[selected] >= .5)).sum()),
                'before': measure([r['label'] for r in probes if r['scope'] == scope], before[selected]),
                'after': measure([r['label'] for r in probes if r['scope'] == scope], after[selected])}
        results['representation_probes'][name] = by_scope
        for r, p, q in zip(probes, before, after):
            audit_rows.append({'id': r['id'], 'scope': r['scope'], 'model': name, 'before': float(p), 'wrapped': float(q),
                'canonical_text_equal': True, 'original_raw_html': bool(extract_features([r['text']])[0, 9])})
    (OUT / 'format_probe_decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audit_rows), encoding='utf-8')
    progress('measuring learned feature contributions')
    names = list(vectorizer.get_feature_names_out()) + STRUCTURAL
    gain = np.zeros(len(names)); contributions = np.zeros(len(names))
    attribution_x = original_x[:min(len(probes), 200)]
    for fold in baseline.calibrated_classifiers_:
        booster = fold.estimator.booster_
        gain += booster.feature_importance(importance_type='gain')
        shap = booster.predict(attribution_x, pred_contrib=True)
        shap = shap.toarray() if hasattr(shap, 'toarray') else np.asarray(shap)
        contributions += np.abs(shap[:, :-1]).mean(axis=0)
    gain /= len(baseline.calibrated_classifiers_)
    contributions /= len(baseline.calibrated_classifiers_)
    top = np.argsort(contributions)[::-1][:30]
    results['attribution'] = {'scope': 'Mean absolute TreeSHAP logit contribution across fixed folds on first 200 external probes; not causal or calibrated probability weights',
        'top_features': [{'feature': names[i], 'mean_abs_logit_contribution': float(contributions[i]), 'training_gain': float(gain[i])} for i in top],
        'structural_fraction_abs_contribution': float(contributions[-16:].sum()/contributions.sum()),
        'structural_fraction_gain': float(gain[-16:].sum()/gain.sum())}
    progress('checking full-message near-overlap independently of first-80-word SimHash')
    canonical = frame.raw_text.map(clean_and_mask_email_text)
    train = frame[frame.split == 'train']
    audit_vectorizer = TfidfVectorizer(ngram_range=(3, 3), min_df=2, max_features=80000, sublinear_tf=True, dtype=np.float32)
    train_x = audit_vectorizer.fit_transform(canonical.loc[train.index])
    nearest = []
    for scope, identifiers, texts in [('internal_test', frame[frame.split == 'test'].index.tolist(), canonical[frame.split == 'test'].tolist()),
                                      ('external', [r['review_id'] for r in external], [clean_and_mask_email_text(r['text']) for r in external])]:
        query = audit_vectorizer.transform(texts)
        for start in range(0, len(texts), 128):
            similarities = (query[start:start+128] @ train_x.T).toarray()
            best = similarities.argmax(axis=1)
            for j, index in enumerate(best):
                train_id = int(train.index[index]); row = start + j
                nearest.append({'scope': scope, 'row': str(identifiers[row]), 'train_row': train_id,
                    'trigram_cosine': float(similarities[j, index]), 'train_label': int(frame.loc[train_id, 'label']),
                    'train_source': frame.loc[train_id, 'source'], 'training_group': str(frame.loc[train_id, 'campaign_cluster_id'])})
    (OUT / 'nearest_training_messages.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in nearest), encoding='utf-8')
    results['near_overlap'] = {scope: {'rows': len(rows), 'cosine_ge_0_95_review_flags': sum(r['trigram_cosine'] >= .95 for r in rows),
        'cosine_ge_0_8': sum(r['trigram_cosine'] >= .8 for r in rows), 'median_nearest_cosine': float(np.median([r['trigram_cosine'] for r in rows]))}
        for scope in ('internal_test', 'external') for rows in [[r for r in nearest if r['scope'] == scope]]}
    assert all(digest(ROOT / name) == value for name, value in protected.items())
    results.update(complete=True, protected_artifacts_unchanged=True, retrained=False, recalibrated=False,
                   limitations=['Existing external author labels are not independently adjudicated', 'Nearest-neighbor flag is not a verified campaign identity',
                                'HTML probes diagnose representation sensitivity, not a new phishing test set'])
    save('results.json', results)
    progress('completed')
    print(json.dumps({key: results[key] for key in ('metrics', 'near_overlap', 'attribution')}, indent=2), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save('run_status.json', {'stage': 'failed', 'error': repr(error)})
        raise
