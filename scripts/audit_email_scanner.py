"""Email-only, read-only model/data audit. No training or live network calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from utils.text_sanitizer import clean_and_mask_email_text
from utils.feature_extractor import extract_features, extract_features_single
from services import scan_engine
from services.security_pipeline import security_pipeline_id
from services.conformal_evidential_engine import ConformalPredictor

OUT = ROOT / 'benchmarks/email_scanner_validation'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    OUT.mkdir(parents=True, exist_ok=True)
    protected = ['data/clean_email_corpus.csv', 'models/phish_model.pkl', 'models/vectorizer.pkl',
                 'models/conformal_params.json']
    hashes = {name: digest(ROOT / name) for name in protected}
    df = pd.read_csv(ROOT / protected[0], keep_default_na=False)
    assert set(df.split) == {'train', 'val', 'test'}
    assert set(df.label) == {0, 1}
    masked = df.raw_text.map(clean_and_mask_email_text)
    structural = extract_features(df.raw_text)
    single = np.vstack([extract_features_single(text) for text in df.raw_text])
    preprocessing_mismatches = int((masked != df.masked_text).sum())
    feature_mismatches = int(np.any(single != structural, axis=1).sum())
    assert np.isfinite(structural).all()
    overlaps = {}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlaps[f'{left}/{right}'] = {
            'exact_masked_texts': len(set(masked[df.split == left]) & set(masked[df.split == right])),
            'cluster_ids': len(set(df.loc[df.split == left, 'campaign_cluster_id']) &
                               set(df.loc[df.split == right, 'campaign_cluster_id']))}
    scan_engine.ML_MODEL = joblib.load(ROOT / 'models/phish_model.pkl')
    scan_engine.TFIDF_VECTORIZER = joblib.load(ROOT / 'models/vectorizer.pkl')
    scan_engine.MODEL_LOADED = True
    for calibrated in scan_engine.ML_MODEL.calibrated_classifiers_:
        calibrated.estimator.set_params(n_jobs=1)  # memory only
    vectorizer = scan_engine.TFIDF_VECTORIZER
    analyzer = vectorizer.build_analyzer()
    frequencies = Counter()
    train = df[df.split == 'train']
    for text in train.masked_text:
        frequencies.update(set(analyzer(text)))
    expected_idf = np.array([np.log((1 + len(train)) / (1 + frequencies[token])) + 1
                             for token, _ in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])])
    idf_max_difference = float(np.max(np.abs(expected_idf - vectorizer.idf_)))
    feature_dimensions = len(vectorizer.vocabulary_) + structural.shape[1]
    model_dimensions = [int(c.estimator.n_features_in_) for c in scan_engine.ML_MODEL.calibrated_classifiers_]
    predictor = ConformalPredictor()
    identity = security_pipeline_id()
    assert predictor.pipeline_id == identity and predictor.is_calibrated
    metrics = {}
    decisions = []
    source_rates = train.groupby('source').label.mean().to_dict()
    for split in ('train', 'val', 'test'):
        subset = df[df.split == split]
        values = []
        for start in range(0, len(subset), 256):
            values.extend(scan_engine.predict_email_probabilities([t + ' ' for t in subset.raw_text.iloc[start:start+256]]))
        values = np.array(values)
        actions = [predictor.predict_set(p, pipeline_id=identity)[2] for p in values]
        source_only = subset.source.map(source_rates).to_numpy()
        metrics[split] = {
            'rows': len(subset), 'roc_auc_against_legacy_labels': float(roc_auc_score(subset.label, values)),
            'brier_against_legacy_labels': float(brier_score_loss(subset.label, values)),
            'source_only_auc_train_label_rates': float(roc_auc_score(subset.label, source_only)),
            'actions_by_label': {f'{label}/{action}': sum(y == label and a == action for y, a in zip(subset.label, actions))
                                for label in (0, 1) for action in ('AUTO_QUARANTINE', 'WARNING_BANNER', 'SAFE_DELIVERY')},
        }
        for idx, y, source, p, action in zip(subset.index, subset.label, subset.source, values, actions):
            decisions.append({'corpus_row_index': int(idx), 'legacy_label': int(y), 'source': source,
                              'split': split, 'probability': float(p), 'action': action})
        print(f'Email-only audit: {split} {len(subset)} rows verified', flush=True)
    decisions = pd.DataFrame(decisions)
    decisions.to_csv(OUT / 'model_decisions.csv', index=False)
    # This is a development error-review queue, never a fresh independent holdout.
    test = decisions[decisions.split == 'test']
    queue = []
    for (label, action, source), cell in test.groupby(['legacy_label', 'action', 'source']):
        # Include every false quarantine and every labelled phishing safe delivery.
        errors = (label == 0 and action == 'AUTO_QUARANTINE') or (label == 1 and action == 'SAFE_DELIVERY')
        selected = cell if errors else cell.sample(n=min(12, len(cell)), random_state=20261003)
        for row in selected.to_dict('records'):
            idx = row['corpus_row_index']
            text = df.loc[idx, 'raw_text']
            queue.append({'sample_id': hashlib.sha256(text.encode()).hexdigest(),
                          'corpus_row_index': idx, 'raw_text': text, 'source': source,
                          'source_corpus_sha256': hashes[protected[0]],
                          'review_status': 'UNREVIEWED', 'security_category': None,
                          'evidence_references': [], 'reviewers': [],
                          'original_full_message_available': False,
                          'scope': 'development_review_not_independent_test'})
    with (OUT / 'blind_review_queue.jsonl').open('w', encoding='utf-8') as stream:
        for record in queue:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    # Keep labels/model decisions separate to avoid anchoring the reviewer.
    test.to_csv(OUT / 'review_answer_key.csv', index=False)
    result = {
        'complete': True, 'email_ready_for_independent_accuracy_claim': False,
        'scope': 'Current body/subject statistical branch against repeatedly inspected legacy labels; no live auxiliary performance claim',
        'pipeline_id': identity, 'preserved_hashes': hashes,
        'preprocessing_mismatches': preprocessing_mismatches,
        'batch_vs_single_structural_feature_mismatches': feature_mismatches,
        'split_overlap': overlaps,
        'source_label_split_counts': [{**row, 'count': int(row['count'])} for row in
            df.groupby(['source','label','split']).size().reset_index(name='count').to_dict('records')],
        'raw_empty_rows': int((df.raw_text.str.strip() == '').sum()),
        'short_model_inputs': int((masked.str.len() <= 5).sum()),
        'conflicting_exact_masked_labels': int(df.assign(masked=masked).groupby('masked').label.nunique().gt(1).sum()),
        'idf_train_only_max_absolute_difference': idf_max_difference,
        'feature_dimensions': feature_dimensions, 'model_dimensions': model_dimensions,
        'model_type': type(scan_engine.ML_MODEL).__name__,
        'base_model_types': [type(c.estimator).__name__ for c in scan_engine.ML_MODEL.calibrated_classifiers_],
        'vectorizer_configuration': {name: getattr(vectorizer, name) for name in
            ('ngram_range','max_features','min_df','token_pattern','sublinear_tf')},
        'metrics': metrics, 'blind_review_queue_rows': len(queue),
        'independent_reviewed_labels_available': False,
        'fresh_source_time_holdout_available': False,
        'blocking_evidence_gaps': ['Phishing/BEC versus spam label adjudication',
            'Original complete messages and authenticated sensor snapshots',
            'Fresh source/time-separated holdout for complete production policy',
            'Unvalidated auxiliary BEC/vision/campaign rules and action execution'],
        'trained_models_and_datasets_changed': False,
    }
    assert all(digest(ROOT / name) == h for name, h in hashes.items())
    assert preprocessing_mismatches == feature_mismatches == 0
    assert all(not any(values.values()) for values in overlaps.values())
    assert idf_max_difference < 1e-10 and all(n == feature_dimensions for n in model_dimensions)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('complete','preprocessing_mismatches',
          'batch_vs_single_structural_feature_mismatches','idf_train_only_max_absolute_difference',
          'feature_dimensions','blind_review_queue_rows','email_ready_for_independent_accuracy_claim')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
