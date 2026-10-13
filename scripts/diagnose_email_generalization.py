"""Descriptive, fixed-model diagnostics; no tuning, relabeling or training."""
import json
import re
import time
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scripts.run_email_retraining_experiment import ROOT, OUT, sha
from utils.text_sanitizer import clean_and_mask_email_text
from services import scan_engine as engine


def main():
    frame = pd.read_csv(ROOT / 'data/clean_email_corpus.csv', keep_default_na=False)
    from utils.jsonl import read_jsonl
    external = read_jsonl(OUT / 'external_inputs.jsonl')
    original = json.loads((ROOT / 'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json').read_text(encoding='utf-8'))
    inventory = {r['review_id']: r for r in read_jsonl(ROOT / 'benchmarks/email_comparison_source_review/comparison_candidates.jsonl')}
    decisions = pd.read_csv(OUT / 'decisions.csv')
    decisions = decisions[decisions.split == 'external'].set_index('row')
    engine.load_ml_model()
    for c in engine.ML_MODEL.calibrated_classifiers_:
        c.estimator.set_params(n_jobs=1)
    bundle = joblib.load(OUT / 'candidate_email_model.pkl')
    candidate = bundle['model']
    findings = {'input_hashes': {'corpus': sha(ROOT / 'data/clean_email_corpus.csv'),
                                'external': sha(OUT / 'external_inputs.jsonl')},
                'analysis_scope': 'Descriptive audit on inspected development/source-labeled comparison rows; not a new independent evaluation',
                'legacy_source_by_label': frame.groupby(['source', 'label']).size().to_dict() if 'source' in frame else {},
                'external_by_class': {}, 'error_review': []}
    findings['legacy_source_by_label'] = {str(k): int(v) for k, v in findings['legacy_source_by_label'].items()}
    for cls in ('Phishing', 'Spam', 'Valid'):
        rows = [r for r in external if r['source_label'] == cls]
        body_urls, metadata_urls, metadata_outside_body, attachments = [], [], [], []
        for r in rows:
            original_row = original[inventory[r['review_id']]['source_row_index']]
            field = original_row.get('URL') or []
            urls = field if isinstance(field, list) else [field]
            urls = [u for u in urls if isinstance(u, str) and re.match('https?://', u, re.I)]
            body_urls.append(bool(engine.extract_urls(r['text'])))
            metadata_urls.append(bool(urls))
            metadata_outside_body.append(any(u not in r['text'] for u in urls))
            attachments.append(bool(original_row.get('File')))
        findings['external_by_class'][cls] = {'rows': len(rows),
                'model_text_contains_http_url_rows': sum(body_urls),
                'source_has_http_url_metadata_rows': sum(metadata_urls),
                'source_http_url_metadata_not_in_model_text_rows': sum(metadata_outside_body),
                'source_attachment_names_present_rows': sum(attachments),
                'median_text_characters': float(np.median([len(r['text']) for r in rows])),
                'over_500_characters': sum(len(r['text']) > 500 for r in rows)}
    supported = decisions.dropna(subset=['baseline_probability', 'candidate_probability'])
    y = (supported.source_label == 'Phishing').to_numpy(dtype=int)
    old, new = supported.baseline_probability.to_numpy(), supported.candidate_probability.to_numpy()
    rng = np.random.default_rng(1729)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    samples = []
    for _ in range(1000):
        indices = np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        a, b = roc_auc_score(y[indices], old[indices]), roc_auc_score(y[indices], new[indices])
        samples.append((a, b, b-a))
    samples = np.asarray(samples)
    findings['external_auc_bootstrap_95_intervals'] = {'baseline': np.quantile(samples[:, 0], [.025, .975]).tolist(),
            'candidate': np.quantile(samples[:, 1], [.025, .975]).tolist(),
            'paired_candidate_minus_baseline': np.quantile(samples[:, 2], [.025, .975]).tolist(),
            'resamples': 1000, 'seed': 1729, 'assumption': 'IID rows within source classes; dependence among campaigns would reduce precision'}
    for r in external:
        decision = decisions.loc[r['review_id']]
        error = r['source_label'] == 'Phishing' and decision.baseline_action == 'SAFE_DELIVERY' or r['source_label'] != 'Phishing' and decision.baseline_action == 'AUTO_QUARANTINE'
        if error or pd.isna(decision.baseline_probability):
            findings['error_review'].append({'review_id': r['review_id'], 'source_label': r['source_label'],
                'baseline_probability': None if pd.isna(decision.baseline_probability) else float(decision.baseline_probability),
                'candidate_probability': None if pd.isna(decision.candidate_probability) else float(decision.candidate_probability),
                'baseline_action': decision.baseline_action, 'candidate_action': decision.candidate_action,
                'review_completed': False, 'note': 'Source-label disagreement, not automatic relabeling; trace original JSON using comparison source index'})
    findings['single_message_model_latency_ms'] = {}
    examples = [r['text'] for r in external if not pd.isna(decisions.loc[r['review_id']].baseline_probability)][:10]
    original_model, original_kind = engine.ML_MODEL, engine.EMAIL_INPUT_KIND
    for name, model, kind in [('baseline', original_model, original_kind), ('candidate', candidate, 'raw_email_model_text')]:
        engine.ML_MODEL, engine.EMAIL_INPUT_KIND = model, kind
        for text in examples:
            engine.predict_email_probabilities([text])
        timings = []
        for index in range(100):
            start = time.perf_counter()
            engine.predict_email_probabilities([examples[index % len(examples)]])
            timings.append((time.perf_counter()-start)*1000)
        findings['single_message_model_latency_ms'][name] = {'runs': 100, 'warm': True, 'cpu_threads_per_tree': 1,
                'median': float(np.median(timings)), 'p95': float(np.quantile(timings, .95)), 'p99': float(np.quantile(timings, .99)),
                'scope': 'Shared preprocessing/vocabulary gate/calibrated model only, not network/OriginTrace/attachments'}
    engine.ML_MODEL, engine.EMAIL_INPUT_KIND = original_model, original_kind
    findings['complete'] = True
    (OUT / 'generalization_diagnostics.json').write_text(json.dumps(findings, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in findings.items() if k != 'error_review'}, indent=2))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    main()
