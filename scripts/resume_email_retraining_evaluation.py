"""Resume the saved, frozen candidate; never refit or silently drop failures."""
import json
import time
import warnings
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from scripts.run_email_retraining_experiment import OUT, ROOT, sha, save, status, metrics, calibrated_actions, probability_batch
from services import scan_engine as engine
from services.security_pipeline import security_pipeline_id
from services.conformal_evidential_engine import ConformalPredictor


def interval(k, n):
    if not n:
        return None
    return [0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k))]


def evaluate(texts, labels, predictor, identity, model=None):
    support = engine.email_vocabulary_supported(texts, model=model, vectorizer=None if model is not None else engine.TFIDF_VECTORIZER)
    eligible = np.flatnonzero(support)
    p = probability_batch([texts[i] for i in eligible], model)
    measures, scored_actions = calibrated_actions(np.asarray(labels)[eligible], p, predictor, identity)
    all_actions, all_p = ['WARNING_BANNER'] * len(texts), [None] * len(texts)
    for i, probability, action in zip(eligible, p, scored_actions):
        all_p[i], all_actions[i] = float(probability), action
    y = np.asarray(labels)
    # ROC/Brier require a probability: disclose the supported denominator separately.
    output = {'total_rows': len(texts), 'supported_probability_rows': int(support.sum()),
              'unsupported_rows': int((~support).sum()), 'unsupported_policy': 'WARNING_BANNER; no invented probability',
              'supported_metrics': metrics(y[eligible], p), 'supported_calibrated_actions': measures,
              'all_rows_action_counts': {}, 'all_rows_coverage_warning_or_quarantine': None}
    for action in ('AUTO_QUARANTINE', 'WARNING_BANNER', 'SAFE_DELIVERY'):
        selected = np.array(all_actions) == action
        output['all_rows_action_counts'][action] = {'count': int(selected.sum()),
                    'positive': int((selected & (y == 1)).sum()), 'negative': int((selected & (y == 0)).sum())}
    q = output['all_rows_action_counts']['AUTO_QUARANTINE']
    output['quarantine_recall'] = q['positive'] / int((y == 1).sum())
    output['quarantine_recall_95ci'] = interval(q['positive'], int((y == 1).sum()))
    output['benign_quarantine_fpr'] = q['negative'] / int((y == 0).sum())
    output['benign_quarantine_fpr_95ci'] = interval(q['negative'], int((y == 0).sum()))
    return output, all_p, all_actions


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    from utils.jsonl import read_jsonl
    external = read_jsonl(OUT / 'external_inputs.jsonl')
    assert protocol['external_sha256'] == __import__('hashlib').sha256(json.dumps(external, sort_keys=True).encode()).hexdigest()
    artifact = joblib.load(OUT / 'candidate_email_model.pkl')
    assert artifact['protocol_sha256'] == sha(OUT / 'protocol.json') and not artifact['approved_for_deployment']
    assert all(sha(ROOT / p) == h for p, h in protocol['protected_hashes'].items())
    candidate = artifact['model']
    training = json.loads((OUT / 'training_metadata.json').read_text())
    assert training['candidate_sha256'] == sha(OUT / 'candidate_email_model.pkl')
    engine.load_ml_model()
    for c in engine.ML_MODEL.calibrated_classifiers_:
        c.estimator.set_params(n_jobs=1)
    baseline_id = security_pipeline_id()
    candidate_id = 'research-email-candidate:' + sha(OUT / 'candidate_email_model.pkl')
    predictors = {name: ConformalPredictor(parameter_path=OUT / (name + '_conformal_params.json')) for name in ('baseline', 'candidate')}
    frame = pd.read_csv(ROOT / 'data/clean_email_corpus.csv', keep_default_na=False)
    results = {'complete': False, 'protocol_sha256': sha(OUT / 'protocol.json'),
               'candidate_sha256': sha(OUT / 'candidate_email_model.pkl'), 'baseline_pipeline_id': baseline_id,
               'candidate_pipeline_id': candidate_id, 'fit_seconds': training['fit_seconds'],
               'resume_reason': 'Existing model has zero-vocabulary external input; retained as warning with no fabricated probability',
               'candidate_vocabulary_sizes': [len(c.estimator.named_steps['features'].vectorizer_.vocabulary_) for c in candidate.calibrated_classifiers_],
               'legacy': {}}
    rows = []
    for split in ('val', 'test', 'external'):
        if split == 'external':
            texts, labels = [r['text'] for r in external], [r['label'] for r in external]
            ids, source_labels = [r['review_id'] for r in external], [r['source_label'] for r in external]
        else:
            part = frame[frame.split == split]
            texts, labels = (part.raw_text + ' ').tolist(), part.label.tolist()
            ids, source_labels = part.index.tolist(), part.label.tolist()
        measures, decisions = {}, {}
        for name, model, identity in [('baseline', None, baseline_id), ('candidate', candidate, candidate_id)]:
            predictor = predictors[name]
            if split == 'val':
                probabilities = probability_batch(texts, model)
                predictor.calibrate(probabilities, labels, pipeline_id=identity)
            measures[name], probabilities, actions = evaluate(texts, labels, predictor, identity, model)
            decisions[name] = list(zip(probabilities, actions))
            status('evaluating', split=split, model=name, rows=len(texts))
        for i, (row_id, source_label) in enumerate(zip(ids, source_labels)):
            rows.append({'split': split, 'row': row_id, 'source_label': source_label,
                         **{f'{name}_{key}': decisions[name][i][j] for name in decisions for j, key in enumerate(('probability', 'action'))}})
        if split == 'external':
            measures['per_source_class'] = {cls: {name: dict(Counter(decisions[name][i][1] for i, r in enumerate(external) if r['source_label'] == cls)) for name in decisions} for cls in ('Phishing', 'Spam', 'Valid')}
            results['external_author_label_comparison'] = measures
        else:
            results['legacy'][split] = measures
        save('evaluation_checkpoint.json', results)
    pd.DataFrame(rows).to_csv(OUT / 'decisions.csv', index=False)
    assert all(sha(ROOT / p) == h for p, h in protocol['protected_hashes'].items())
    assert security_pipeline_id() == baseline_id
    results.update(complete=True, production_artifacts_changed=False, independent_deployment_accuracy_established=False,
                   full_operational_engine_policy_evaluated=False, hyperparameters_selected_using_test=False)
    save('results.json', results)
    status('completed')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error))
        raise
