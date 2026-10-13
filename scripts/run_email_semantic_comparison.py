"""One frozen multilingual Email candidate versus unchanged research baselines.

Existing external examples have already been inspected: this is development
evidence, not an untouched acceptance test or a claim of 26-language accuracy.
No production models, calibration, dataset files, or frontend are modified.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM'):
    os.environ[_name] = 'false' if _name == 'TOKENIZERS_PARALLELISM' else '1'

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

from training.email_semantic import FrozenEmailEncoder, content_text, SEMANTIC_TEXT_VERSION
from utils.email_input_contract import contract_sha256
from utils.jsonl import read_jsonl
from scripts.profile_email_runtime import memory
from scripts.train_email_evidence_candidate import metrics, validation_threshold, validate_output

POOL = 'published_spanish_plus_source_author_english'
REFERENCE = POOL + '__text_only__source_class_balanced'
REFERENCE_DIRECTORY = ROOT / 'benchmarks/email_published_task_repair_20261005'
DEFAULT_ASSETS = ROOT / 'benchmarks/email_multilingual_assets_20261006/e5-small-int8'
PROTECTED = ('models/phish_model.pkl', 'models/vectorizer.pkl', 'models/url_model.pkl',
             'models/domain_model.pkl', 'models/conformal_params.json',
             'models/conformal_evaluation_results.json', 'data/clean_email_corpus.csv')
SOURCE_FILES = ('training/email_semantic.py', 'scripts/run_email_semantic_comparison.py',
                'utils/email_candidate_text.py', 'utils/email_input_contract.py',
                'utils/text_sanitizer.py', 'scripts/train_email_evidence_candidate.py',
                'scripts/profile_email_runtime.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    partial = path.with_suffix(path.suffix + '.pending')
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def status(directory, stage, **details):
    value = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **details)
    save(directory / 'run_status.json', value)
    print(json.dumps(value), flush=True)


def source_weights(rows):
    counts = Counter((r['source'], r['label']) for r in rows)
    if any((s, y) not in counts for s in {r['source'] for r in rows} for y in (0, 1)):
        raise ValueError('Every training source must contain both labels')
    return np.asarray([len(rows) / (len(counts) * counts[(r['source'], r['label'])])
                       for r in rows], dtype=float)


def corpus():
    prior = json.loads((REFERENCE_DIRECTORY / 'protocol.json').read_text(encoding='utf-8'))
    prior_result = json.loads((REFERENCE_DIRECTORY / 'results.json').read_text(encoding='utf-8'))
    if not prior_result.get('complete') or prior_result['protocol_sha256'] != digest(REFERENCE_DIRECTORY / 'protocol.json'):
        raise ValueError('Previous research experiment is not complete/hash-bound')
    partition_path = REFERENCE_DIRECTORY / 'partition_rows.jsonl'
    if digest(partition_path) != prior['prepared_output_hashes']['partition_rows.jsonl']:
        raise ValueError('Frozen research partitions changed')
    # Source files must still match the original audit, but old code hashes are
    # not reused as a claim that the current production pipeline was evaluated.
    for path, expected in prior['input_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Previous research input changed: ' + path)
    for path in ('models/phish_model.pkl', 'models/vectorizer.pkl'):
        if digest(ROOT / path) != prior['protected_hashes'][path]:
            raise ValueError('Deployed reference model changed')
    records = [r for r in read_jsonl(partition_path) if r['pool'] == POOL]
    brown = read_jsonl(prior['Brown_records_path'])
    records += [dict(r, split='external_english') for r in brown]
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Repeated message IDs')
    group_sets = {}
    for row in records:
        if row['input_contract_sha256'] != contract_sha256(row['email_input']):
            raise ValueError('Email evidence contract hash mismatch: ' + row['id'])
        if row['split'] != 'external_english':
            group_sets.setdefault(row['split'], set()).add(row['experiment_template_group'])
    for a in group_sets:
        for b in group_sets:
            if a != b and group_sets[a] & group_sets[b]:
                raise ValueError('Template overlap across frozen partitions')
    return records, prior


def protected_hashes():
    paths = [ROOT / p for p in PROTECTED if (ROOT / p).is_file()]
    paths += [p for folder in ('templates', 'static') for p in (ROOT / folder).rglob('*') if p.is_file()]
    return {str(p): digest(p) for p in paths}


def assert_frozen(protocol):
    for collection in ('input_hashes', 'code_hashes', 'protected_hashes', 'asset_hashes'):
        for path, expected in protocol[collection].items():
            if digest(path) != expected:
                raise ValueError('Frozen ' + collection + ' changed: ' + path)


def prepare(output, assets):
    output = validate_output(output)
    records, prior = corpus()
    asset_manifest_path = assets / 'manifest.json'
    manifest = json.loads(asset_manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('complete') or manifest['revision'] != '614241f622f53c4eeff9890bdc4f31cfecc418b3':
        raise ValueError('Pinned model download is incomplete or wrong revision')
    assets_hashes = {str(asset_manifest_path): digest(asset_manifest_path)}
    for name, value in manifest['files'].items():
        path = assets / name
        if path.stat().st_size != value['bytes'] or digest(path) != value['sha256']:
            raise ValueError('Local encoder assets changed: ' + name)
        assets_hashes[str(path)] = value['sha256']
    names = ('protocol.json', 'results.json', 'partition_rows.jsonl', 'predictions.jsonl',
             'matched_temporal/original_predictions.jsonl')
    input_hashes = {str(REFERENCE_DIRECTORY / n): digest(REFERENCE_DIRECTORY / n) for n in names}
    input_hashes.update(prior['input_hashes'])
    protocol = dict(schema_version=1, prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        candidate='frozen_multilingual_e5_small_int8_logistic', assets_directory=str(assets),
        repository=manifest['repository'], revision=manifest['revision'], license='MIT',
        training_pool=POOL, text_version=SEMANTIC_TEXT_VERSION,
        classifier=dict(C=1.0, solver='liblinear', max_iter=500, class_weight=None, random_state=20261006),
        encoder=dict(max_tokens=512, intra_op_threads=2, inter_op_threads=1, provider='CPUExecutionProvider',
                     pooling='masked mean per window, content-token weighted normalized document mean',
                     window_rule='all non-overlapping 510-content-token windows; no silent truncation', prefix='query: '),
        sample_weights='train-only n/(observed source/class cells * cell count), same as frozen text reference',
        validation_fpr_budget=.01, hyperparameter_search=False, calibration_fitted=False,
        approved_for_deployment=False, full_engine_evaluated=False, frontend_changes=False,
        corpus_counts={s:dict(rows=sum(r['split']==s for r in records),
            labels=dict(Counter(str(r['label']) for r in records if r['split']==s)))
            for s in sorted({r['split'] for r in records})},
        input_hashes=input_hashes, code_hashes={str(ROOT / p):digest(ROOT / p) for p in SOURCE_FILES},
        protected_hashes=protected_hashes(), asset_hashes=assets_hashes,
        runtime_versions={p:version(p) for p in ('numpy', 'scikit-learn', 'onnxruntime', 'tokenizers')},
        limitations=[
            'Brown and Spanish holdout results were previously inspected: development comparison, not blind acceptance.',
            'Brown has only 33 eligible English excerpts; all 90 rows and quality exclusions remain reported.',
            'Spanish test is source-reported 2025 within one publisher; English development dates are unverified.',
            'No independent project label review is claimed. PhishFuzzer training reuse permission remains unestablished.',
            'Representation experiment only: not full OriginTrace validation, production calibration or 26-language accuracy.',
            'Pretraining overlap with public email examples cannot be ruled out.',
            'Quantized ONNX asset is tested locally; no unquantized-equivalence claim.'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'protocol.json', protocol)
    status(output, 'prepared', corpus_counts=protocol['corpus_counts'])


def baseline_lookup():
    scores = {REFERENCE:{}, 'deployed_baseline':{}}
    for name in ('predictions.jsonl', 'matched_temporal/original_predictions.jsonl'):
        for row in read_jsonl(REFERENCE_DIRECTORY / name):
            if row['model'] not in scores:
                continue
            # A message ID can occur in both original and intervention records.
            # The neutral-footer probe must never overwrite the original email.
            split = ('published_day_test' if name.startswith('matched_temporal/')
                     else {'val':'val', 'published_day_test':'published_day_test',
                           'brown_English_source_test':'external_english'}.get(row.get('split')))
            if split is None:
                continue
            key = (split, row['row_id'])
            bucket = scores[row['model']]
            if key in bucket:
                raise ValueError('Duplicate baseline model/cohort/message record: ' + str(key))
            bucket[key] = row
    return scores


def evaluate(rows, scores, threshold):
    eligible = np.asarray([r.get('eligible_for_scoring', True) for r in rows], dtype=bool)
    y = np.asarray([r['label'] for r in rows])
    p = np.asarray(scores, dtype=float)
    return dict(total_rows=len(rows), eligible_rows=int(eligible.sum()), quality_excluded_rows=int((~eligible).sum()),
                at_0_5=metrics(y[eligible], p[eligible]),
                at_frozen_validation_budget=metrics(y[eligible], p[eligible], threshold),
                quality_exclusion_counts=dict(Counter(reason for r in rows for reason in r.get('exclusion_reasons', []))))


def run(output):
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    assert_frozen(protocol)
    records, _ = corpus()
    protocol_hash = digest(output / 'protocol.json')
    embedding_path = output / 'embeddings.npy'
    trace_path = output / 'embedding_trace.jsonl'
    if embedding_path.exists():
        embeddings = np.load(embedding_path)
        traces = read_jsonl(trace_path)
        cache = json.loads((output / 'embedding_manifest.json').read_text(encoding='utf-8'))
        if cache['protocol_sha256'] != protocol_hash or cache['embeddings_sha256'] != digest(embedding_path) or cache['trace_sha256'] != digest(trace_path):
            raise ValueError('Embedding cache mismatch')
        if embeddings.shape != (len(records), 384) or [r['row_id'] for r in traces] != [r['id'] for r in records]:
            raise ValueError('Embedding row alignment mismatch')
        resource = cache['resource']
    else:
        start = time.perf_counter()
        before = memory()
        encoder = FrozenEmailEncoder(protocol['assets_directory'], max_tokens=512, threads=2)
        resource = dict(before_load=before, after_load=memory(), load_wall_seconds=time.perf_counter()-start)
        embeddings = np.full((len(records), 384), np.nan, dtype=np.float32)
        traces = []
        with trace_path.open('w', encoding='utf-8') as stream:
            for index, row in enumerate(records):
                record = dict(row_id=row['id'], input_contract_sha256=row['input_contract_sha256'], split=row['split'])
                if not row.get('eligible_for_scoring', True):
                    record.update(state='QUALITY_EXCLUDED', reasons=row['exclusion_reasons'])
                else:
                    try:
                        embeddings[index], details = encoder.encode(content_text(row['email_input']))
                        record.update(state='SCORED', **details)
                    except Exception as error:
                        record.update(state='UNAVAILABLE', error_type=type(error).__name__, error=str(error))
                traces.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
                if (index + 1) % 50 == 0:
                    status(output, 'encoding', completed=index+1, total=len(records), memory=memory())
        resource['after_encoding'] = memory()
        resource['embedding_wall_seconds'] = time.perf_counter()-start-resource['load_wall_seconds']
        np.save(embedding_path, embeddings)
        save(output / 'embedding_manifest.json', dict(protocol_sha256=protocol_hash,
            embeddings_sha256=digest(embedding_path), trace_sha256=digest(trace_path), resource=resource))
        del encoder
    finite = np.isfinite(embeddings).all(axis=1)
    train = np.asarray([r['split']=='train' for r in records])
    val = np.asarray([r['split']=='val' for r in records])
    if not finite[train | val].all():
        raise ValueError('Training or validation inference failed; no selective removal or silent refit')
    labels = np.asarray([r['label'] for r in records])
    weights = source_weights([r for r in records if r['split']=='train'])
    classifier = LogisticRegression(**protocol['classifier'])
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        classifier.fit(embeddings[train], labels[train], sample_weight=weights)
    train_seconds = time.perf_counter()-started
    p = np.full(len(records), np.nan)
    started = time.perf_counter()
    p[finite] = classifier.predict_proba(embeddings[finite])[:, list(classifier.classes_).index(1)]
    classifier_seconds = time.perf_counter()-started
    threshold = validation_threshold(labels[val], p[val], protocol['validation_fpr_budget'])
    joblib.dump(dict(classifier=classifier, protocol_sha256=protocol_hash, approved_for_deployment=False,
                     text_version=SEMANTIC_TEXT_VERSION, encoder_revision=protocol['revision']), output / 'candidate.pkl')
    save(output / 'fitted_candidate.json', dict(protocol_sha256=protocol_hash, model_sha256=digest(output / 'candidate.pkl'),
        threshold=threshold, threshold_source='validation only', trained_rows=int(train.sum()), fitted_at_utc=datetime.now(timezone.utc).isoformat()))
    # Candidate/threshold are saved before held-out metric calculation.
    reference = baseline_lookup()
    previous = json.loads((REFERENCE_DIRECTORY / 'results.json').read_text(encoding='utf-8'))
    result = dict(complete=False, approved_for_deployment=False, calibration_fitted=False,
        protocol_sha256=protocol_hash, threshold=threshold, training_seconds=train_seconds,
        classifier_all_rows_seconds=classifier_seconds, resources=resource, splits={}, baseline_comparison={},
        limitations=protocol['limitations'])
    for split in ('val', 'published_day_test', 'external_english'):
        mask = np.asarray([r['split']==split for r in records])
        cohort = [r for r in records if r['split']==split]
        result['splits'][split] = evaluate(cohort, p[mask], threshold)
        for name, lookup in reference.items():
            baseline = []
            for row in cohort:
                observation = lookup.get((split, row['id']))
                if observation is None:
                    baseline.append(np.nan)
                else:
                    if observation['label'] != row['label'] or observation['input_contract_sha256'] != row['input_contract_sha256']:
                        raise ValueError('Reference label/contract mismatch')
                    baseline.append(observation['probability'] if observation['probability'] is not None else np.nan)
            baseline = np.asarray(baseline, dtype=float)
            baseline_threshold = previous['models'][name]['threshold']
            common = np.isfinite(p[mask]) & np.isfinite(baseline) & np.asarray([r.get('eligible_for_scoring',True) for r in cohort])
            comparison = dict(baseline_full=evaluate(cohort, baseline, baseline_threshold), common_scored_rows=int(common.sum()),
                candidate_paired_at_0_5=metrics(labels[mask][common],p[mask][common]),
                baseline_paired_at_0_5=metrics(labels[mask][common],baseline[common]),
                candidate_paired_at_budget=metrics(labels[mask][common],p[mask][common],threshold),
                baseline_paired_at_budget=metrics(labels[mask][common],baseline[common],baseline_threshold))
            result['baseline_comparison'].setdefault(name,{})[split]=comparison
    timings = [t['wall_ms'] for t in traces if t['state']=='SCORED']
    result['encoder_timing'] = dict(measured_rows=len(timings), median_ms=float(np.median(timings)),
        mean_ms=float(np.mean(timings)), p95_ms=float(np.quantile(timings,.95)), p99_ms=float(np.quantile(timings,.99)),
        total_windows=sum(t.get('windows',0) for t in traces), dropped_content_tokens=0,
        includes='tokenization + all ONNX windows + pooling; excludes canonicalization/classifier/network/full engine')
    with (output / 'predictions.jsonl').open('w',encoding='utf-8') as stream, (output / 'error_review.jsonl').open('w',encoding='utf-8') as errors:
        for row, probability, trace in zip(records,p,traces):
            item = dict(row_id=row['id'],split=row['split'],source=row['source'],label=row['label'],
                input_contract_sha256=row['input_contract_sha256'], probability=float(probability) if np.isfinite(probability) else None,
                eligible_for_scoring=row.get('eligible_for_scoring',True), state=trace['state'],
                prediction_at_0_5=int(probability>=.5) if np.isfinite(probability) else None,
                prediction_at_budget=int(probability>=threshold) if np.isfinite(probability) else None,
                threshold=threshold, exclusion_reasons=row.get('exclusion_reasons',[]))
            stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
            if row['split'] not in ('train','val') and np.isfinite(probability):
                refs={name:lookup.get((row['split'],row['id']),{}).get('probability') for name,lookup in reference.items()}
                if item['prediction_at_0_5']!=row['label'] or item['prediction_at_budget']!=row['label'] or any(
                    value is not None and int(value>=.5)!=item['prediction_at_0_5'] for value in refs.values()):
                    errors.write(json.dumps(dict(item,reference_probabilities=refs,
                        original_reference=row.get('original_reference'), review_status='requires human evidence review'),ensure_ascii=False,allow_nan=False)+'\n')
    assert_frozen(protocol)
    result.update(complete=True, protected_models_datasets_and_frontend_unchanged=True,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'results.json',result)
    write_report(output,result)
    status(output,'complete',results_sha256=digest(output / 'results.json'))


def write_report(output,result):
    lines=['# Compact multilingual Email representation comparison','',
        '**Research candidate only. No deployment or production calibration.**',
        'Existing English/Spanish examples were previously inspected; these are development results, not independent 26-language acceptance.','',
        '| Cohort/model | Rows | Accuracy | AUC | FPR | Recall | Brier |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def cell(value,percent=False):
        return 'unavailable' if value is None else f'{value:.2%}' if percent else f'{value:.4f}'
    for split in ('val','published_day_test','external_english'):
        choices={'semantic candidate':result['splits'][split]['at_0_5']}
        choices.update({name:data[split]['baseline_full']['at_0_5'] for name,data in result['baseline_comparison'].items()})
        for name,m in choices.items():
            lines.append('| '+split+' / '+name+' | '+str(m['scored_rows'])+' | '+' | '.join(
                cell(m.get(k),k in ('accuracy','fpr','recall')) for k in ('accuracy','roc_auc','fpr','recall','brier'))+' |')
    lines+=['','## Frozen validation FPR budgets','',
        'Every threshold was fitted on its validation data only. Empirical validation FPR budgets do not guarantee external FPR or FDR.','',
        '| Cohort/model | FPR | Recall | Confusion matrix [[TN,FP],[FN,TP]] |','|---|---:|---:|---|']
    for split in ('published_day_test','external_english'):
        choices={'semantic candidate':result['splits'][split]['at_frozen_validation_budget']}
        choices.update({name:data[split]['baseline_full']['at_frozen_validation_budget'] for name,data in result['baseline_comparison'].items()})
        for name,m in choices.items():
            lines.append('| '+split+' / '+name+' | '+cell(m.get('fpr'),True)+' | '+cell(m.get('recall'),True)+' | '+str(m.get('confusion_matrix'))+' |')
    lines+=['','## CPU and memory','',json.dumps(result['encoder_timing'],indent=2),
            '',json.dumps(result['resources'],indent=2),'','## Limits','']
    lines+=['- '+item for item in result['limitations']]
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    parser.add_argument('--execute-prepared',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.execute_prepared:
        try:
            run(output)
        except Exception as error:
            if output.is_dir():
                status(output,'failed',error_type=type(error).__name__,error=str(error))
            raise
    else:
        prepare(output,args.assets.resolve())


if __name__=='__main__':
    main()
