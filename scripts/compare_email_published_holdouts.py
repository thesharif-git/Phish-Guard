"""Compare frozen Email models on the identical publisher-day holdout.

No fitting, calibration, production model replacement or corpus edits.
New probabilities come from the completed frozen research experiment.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts.run_published_email_experiment import (
    POOLS, assert_frozen, digest, metrics, save, save_jsonl, timed_scores,
)
from scripts.compare_email_representations import score_baseline, score_candidate
from scripts.train_email_evidence_candidate import validate_output
from utils.jsonl import read_jsonl


def paired_metrics(rows, probabilities, baseline, threshold):
    p = np.asarray(probabilities, dtype=float)
    reference = np.asarray(baseline, dtype=float)
    mask = np.isfinite(p) & np.isfinite(reference)
    y = np.asarray([row['label'] for row in rows])
    return dict(total_rows=len(rows), common_supported_rows=int(mask.sum()),
        candidate_at_0_5=metrics(y[mask], p[mask]),
        candidate_at_frozen_validation_budget=metrics(y[mask], p[mask], threshold),
        baseline_at_0_5=metrics(y[mask], reference[mask]))


def run(experiment, destination):
    directory, destination = Path(experiment).resolve(), validate_output(destination)
    protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    result = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    if result.get('complete') is not True or result.get('protocol_sha256') != digest(directory/'protocol.json'):
        raise ValueError('Require a completed hash-bound research experiment')
    assert_frozen(directory, protocol)
    rows = [row for row in read_jsonl(directory/'partition_rows.jsonl')
            if row['pool']==POOLS[0] and row['split']=='published_day_test']
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Publisher-day holdout IDs are not unique')
    lookup = {row['id']: row for row in rows}
    candidate_names = [name for name, value in result['models'].items() if value.get('fitted_in_this_run') is not False]
    candidate_scores = {name: {} for name in candidate_names}
    for prediction in read_jsonl(directory/'predictions.jsonl'):
        if prediction['model'] not in candidate_scores or prediction['split'] != 'published_day_test':
            continue
        row = lookup.get(prediction['row_id'])
        bucket = candidate_scores[prediction['model']]
        if (row is None or row['id'] in bucket or prediction['label'] != row['label']
                or prediction['input_contract_sha256'] != row['input_contract_sha256']):
            raise ValueError('Candidate prediction row identity, label or contract differs')
        bucket[row['id']] = prediction['probability']
    for name, bucket in candidate_scores.items():
        if set(bucket) != set(lookup):
            raise ValueError('Incomplete frozen publisher-day predictions: '+name)
    destination.mkdir(parents=True, exist_ok=False)
    inputs = {str(directory/name): digest(directory/name) for name in
              ('protocol.json','results.json','partition_rows.jsonl','predictions.jsonl')}
    inputs[str(Path(__file__).resolve())] = digest(__file__)
    save(destination/'protocol.json', dict(input_hashes=inputs,
        original_pipeline_id=protocol['pipeline_id'], holdout_source='spaphish_v4',
        holdout_rows=len(rows), class_counts={str(label):sum(row['label']==label for row in rows) for label in (0,1)},
        no_fitting=True, no_calibration=True, approved_for_deployment=False,
        note='Publisher-reported 2025 day holdout within SpaPhish. English training dates are unverified; not strict chronology of the mixed pool. Every original threshold remains frozen on its own development validation.'))
    output = dict(complete=False, no_fitting=True, no_calibration=True,
                  approved_for_deployment=False, rows=len(rows), models={})
    original_scores, all_predictions = {}, []
    texts = [row['email_input']['body']+' '+row['email_input']['subject'] for row in rows]
    for name, metadata in protocol['original_English_models'].items():
        print('Scoring frozen original on publisher-day holdout: '+name, flush=True)
        if name=='deployed_baseline':
            model = joblib.load(ROOT/'models/phish_model.pkl')
            vectorizer = joblib.load(ROOT/'models/vectorizer.pkl')
            for fold in getattr(model,'calibrated_classifiers_',[]):
                fold.estimator.set_params(n_jobs=1)
            scoring = lambda values: score_baseline(model,vectorizer,values)
        else:
            model = joblib.load(metadata['model_path'])['model']
            scoring = lambda values: score_candidate(model,values)
        p, timing = timed_scores(scoring,texts)
        original_scores[name] = p
        output['models'][name] = dict(threshold=metadata['threshold'], timing=timing)
        all_predictions.extend(dict(model=name,row_id=row['id'],label=row['label'],
            input_contract_sha256=row['input_contract_sha256'],
            probability=float(value) if np.isfinite(value) else None)
            for row,value in zip(rows,p))
        del model, scoring
        if name=='deployed_baseline':del vectorizer
        gc.collect()
    reference = original_scores['deployed_baseline']
    for name,bucket in candidate_scores.items():
        original_scores[name] = np.asarray([bucket[row['id']] if bucket[row['id']] is not None else np.nan for row in rows])
        output['models'][name] = dict(threshold=result['models'][name]['threshold'])
    for name, probabilities in original_scores.items():
        item = output['models'][name]
        item['paired_with_baseline'] = paired_metrics(rows, probabilities, reference, item['threshold'])
        item['full_cohort_at_0_5'] = metrics([r['label'] for r in rows], probabilities)
    assert_frozen(directory,protocol)
    if any(digest(path)!=sha for path,sha in inputs.items()):
        raise ValueError('Comparison inputs changed during evaluation')
    output.update(complete=True, frozen_inputs_unchanged=True)
    save_jsonl(destination/'original_predictions.jsonl',all_predictions)
    save(destination/'results.json',output)
    lines = ['# Matched publisher-day Email comparison','',
        f'All models use the same {len(rows)}-record publisher-reported 2025 SpaPhish holdout. '
        'This is Spanish evidence, not English deployment accuracy. Source-author English training chronology is unverified. '
        'No model was fitted, selected, promoted or calibrated in this comparison.','',
        '| Model | Paired scored rows | Accuracy | AUC | FPR at .5 | Recall at .5 | Brier |','|---|---:|---:|---:|---:|---:|---:|']
    def num(value, percentage=False):
        return 'unavailable' if value is None else f'{value:.2%}' if percentage else f'{value:.4f}'
    for name,item in output['models'].items():
        paired = item['paired_with_baseline']
        report = paired['candidate_at_0_5']
        lines.append('| '+name+' | '+str(paired['common_supported_rows'])+' | '+
            ' | '.join(num(report.get(key),key in ('accuracy','fpr','recall'))
                       for key in ('accuracy','roc_auc','fpr','recall','brier'))+' |')
    lines += ['', 'Full-cohort coverage, unsupported classes and the paired baseline metrics remain in results.json. '
              'The historical validation budgets have not been refitted to this holdout and provide no external FPR or FDR guarantee.']
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.experiment,args.output)
