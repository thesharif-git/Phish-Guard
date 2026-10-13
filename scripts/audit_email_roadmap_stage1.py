"""Approved roadmap Stage 1: verify baseline measurements, without training.

Preserves historical dataset limitations; no promotion, calibration or UI edits.
Reproduces original inputs, not intervention/neutral-footer probe variants.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scripts.run_email_semantic_comparison import (
    REFERENCE, REFERENCE_DIRECTORY, baseline_lookup, corpus, digest, save,
)
from scripts.compare_email_representations import score_baseline
from training.email_evidence_model import predict_email_evidence_probabilities


def measured(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError('Invalid metric inputs')
    if np.isinf(p).any() or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError('Invalid probabilities')
    mask = np.isfinite(p)
    total = len(y)
    y, p = y[mask], p[mask]
    outcome = p >= .5
    tn = int(((y == 0) & ~outcome).sum())
    fp = int(((y == 0) & outcome).sum())
    fn = int(((y == 1) & ~outcome).sum())
    tp = int(((y == 1) & outcome).sum())
    if not len(y):
        return dict(total_rows=total, n=0, unavailable_rows=total)
    result = dict(total_rows=total, n=len(y), unavailable_rows=total-len(y), tn=tn, fp=fp, fn=fn, tp=tp,
        accuracy=(tn+tp)/len(y), precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        fpr=fp/(tn+fp) if tn+fp else None, fnr=fn/(tp+fn) if tp+fn else None,
        roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        pr_auc_average_precision=float(average_precision_score(y,p)) if len(set(y))==2 else None,
        brier=float(np.mean((p-y)**2)))
    ece = 0.
    for index in range(10):
        chosen = (p >= index/10) & (p < (index+1)/10 if index < 9 else p <= 1)
        if chosen.any():
            ece += chosen.mean()*abs(p[chosen].mean()-y[chosen].mean())
    result['ece_10_equal_width'] = float(ece)
    return result


def legacy_reproduction():
    path = ROOT/'benchmarks/email_retraining_20261003/decisions.csv'
    with path.open(encoding='utf-8') as stream:
        decisions = list(csv.DictReader(stream))
    historical = json.loads((ROOT/'benchmarks/email_generalization_root_20261003/results.json').read_text())
    checks = {}
    for split in ('test','external'):
        rows = [r for r in decisions if r['split']==split]
        labels = [int(r['source_label']=='Phishing') if split=='external' else int(r['source_label']) for r in rows]
        probabilities = [float(r['baseline_probability']) if r['baseline_probability'] else np.nan for r in rows]
        current = measured(labels, probabilities)
        original = historical['metrics'][split]['baseline']
        for key in ('n','tn','fp','fn','tp','accuracy','precision','recall','f1','fpr','fnr','roc_auc','brier','ece_10_equal_width'):
            if not np.isclose(current[key], original[key], rtol=0, atol=1e-10):
                raise ValueError('Historical baseline metric reproduction failed: '+split+'/'+key)
        checks[split] = dict(metrics=current, reproduced=True,
            scope='Previously inspected internal development' if split=='test' else
                  'Historical source labels: Phishing versus Spam/Valid, not independently reviewed phishing/BEC ground truth')
    return checks, {str(path):digest(path),
        str(ROOT/'benchmarks/email_generalization_root_20261003/results.json'):
            digest(ROOT/'benchmarks/email_generalization_root_20261003/results.json')}


def main():
    directory = ROOT/'benchmarks/email_roadmap_v1_20261006'
    snapshot_path = directory/'protected_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    for path, expected in snapshot.items():
        if digest(path) != expected:
            raise ValueError('Protected baseline/UI/source changed: '+path)
    legacy, input_hashes = legacy_reproduction()
    records, _ = corpus()
    reference_scores = baseline_lookup()
    for row in records:
        for lookup in reference_scores.values():
            observation = lookup.get((row['split'],row['id']))
            if observation is not None and (observation['label']!=row['label'] or
                    observation['input_contract_sha256']!=row['input_contract_sha256']):
                raise ValueError('Original reference evidence differs: '+row['id'])
    cohorts = {'external_english':[r for r in records if r['split']=='external_english'],
               'published_day_test':[r for r in records if r['split']=='published_day_test']}
    input_hashes.update({str(REFERENCE_DIRECTORY/name):digest(REFERENCE_DIRECTORY/name)
        for name in ('protocol.json','results.json','partition_rows.jsonl','predictions.jsonl',
                     'matched_temporal/original_predictions.jsonl')})
    input_hashes[str(Path(__file__))] = digest(__file__)
    input_hashes[str(ROOT/'scripts/run_email_semantic_comparison.py')] = digest(ROOT/'scripts/run_email_semantic_comparison.py')
    report = dict(schema_version=1, stage='1_baseline_and_harness', complete=False,
        new_training=False, calibration_fitted=False, promoted=False, frontend_changes=False,
        model_reliability_established=False, legacy_measurement_reproduction=legacy,
        source_labeled_comparisons={}, input_hashes=input_hashes,
        corrected_defect='Row-ID-only reference lookup overwrote original Brown records with neutral-footer probe records; now uses cohort/message identity and rejects duplicate records.',
        limitations=['Stage 1 checks measurement correctness, not model deployment acceptance.',
            'Inspected development sets are not a locked acceptance set.',
            'Brown contains 33 eligible English excerpts among 90 source records.',
            'Spanish publisher-day holdout comes from one source, with no trusted receiver dates/authentication.',
            'Historical external source-label benchmark mapped Spam to negative; final task-valid acceptance requires separate review.'])
    output_records=[]
    save(directory/'stage1_run_status.json',dict(stage='baseline_reproduction',complete=False))
    for name in ('deployed_baseline', REFERENCE):
        print('Reproducing original reference: '+name,flush=True)
        if name=='deployed_baseline':
            model = joblib.load(ROOT/'models/phish_model.pkl')
            vectorizer = joblib.load(ROOT/'models/vectorizer.pkl')
            for fold in getattr(model,'calibrated_classifiers_',[]):
                fold.estimator.set_params(n_jobs=1)
            def predict(rows):
                return score_baseline(model,vectorizer,[r['email_input']['body']+' '+r['email_input']['subject'] for r in rows])
        else:
            model_path = REFERENCE_DIRECTORY/(REFERENCE+'.pkl')
            input_hashes[str(model_path)] = digest(model_path)
            model = joblib.load(model_path)['model']
            def predict(rows):
                return predict_email_evidence_probabilities(model,[r['email_input'] for r in rows])
        for split, rows in cohorts.items():
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                probabilities = predict(rows)
            elapsed = time.perf_counter()-started
            maximum_difference = 0.
            for row, probability in zip(rows,probabilities):
                stored = reference_scores[name].get((split,row['id']))
                if stored is None:
                    raise ValueError('Missing original baseline prediction')
                expected = stored['probability']
                if np.isfinite(probability) != (expected is not None):
                    raise ValueError('Baseline support differs: '+row['id'])
                if expected is not None:
                    difference = abs(float(probability)-expected)
                    maximum_difference = max(maximum_difference,difference)
                    if difference > 1e-8:
                        raise ValueError('Reproduced original probability differs: '+name+'/'+row['id'])
                output_records.append(dict(model=name,split=split,row_id=row['id'],label=row['label'],
                    input_contract_sha256=row['input_contract_sha256'],probability=float(probability) if np.isfinite(probability) else None,
                    eligible_for_scoring=row.get('eligible_for_scoring',True),
                    exclusion_reasons=row.get('exclusion_reasons',[])))
            eligible = np.asarray([r.get('eligible_for_scoring',True) for r in rows],dtype=bool)
            labels = np.asarray([r['label'] for r in rows])
            report['source_labeled_comparisons'].setdefault(name,{})[split] = dict(
                total_source_rows=len(rows),eligible_rows=int(eligible.sum()),quality_excluded_rows=int((~eligible).sum()),
                original_probabilities_reproduced=True,max_absolute_probability_difference=maximum_difference,
                at_0_5=measured(labels[eligible],np.asarray(probabilities)[eligible]),
                batch_inference_seconds=elapsed,timing_scope='local model/representation only, not full engine',
                captured_warnings=dict(Counter(type(w.message).__name__+': '+str(w.message) for w in caught)))
            save(directory/'stage1_run_status.json',dict(stage='baseline_reproduction',model=name,cohort=split,
                reproduced_rows=len(rows),max_absolute_probability_difference=maximum_difference,complete=False))
            print(json.dumps(dict(model=name,cohort=split,reproduced_rows=len(rows),elapsed_seconds=elapsed)),flush=True)
        del model
    for path, expected in {**snapshot,**input_hashes}.items():
        if digest(path)!=expected:
            raise ValueError('Input changed during Stage 1 verification: '+path)
    with (directory/'stage1_baseline_predictions.jsonl').open('w',encoding='utf-8') as stream:
        for item in output_records:
            stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    report.update(complete=True,frontend_models_datasets_and_engine_unchanged=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(directory/'stage1_baseline_verification.json',report)
    save(directory/'stage1_run_status.json',dict(stage='complete',complete=True))
    print(json.dumps(dict(complete=True,stage=report['stage'],actual_predictions_reproduced=len(output_records))),flush=True)


if __name__=='__main__':
    main()
