"""Frozen Email-only development ablation, with every failure retained.

Uses already-inspected legacy/source-author labels. It cannot establish real
deployment reliability and never modifies production models/data/calibration.
There is no search, label relabeling, calibration fitting, or automatic promotion.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import gc
import hashlib
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from training.email_text_candidates import CandidateEmailFeatures
from utils.feature_extractor import extract_features
from utils.text_sanitizer import clean_and_mask_email_text
from scripts.profile_email_runtime import memory

SEED = 20261004
VALIDATION_FPR_BUDGET = .01
PROTECTED = [
    'data/clean_email_corpus.csv', 'data/email_evidence_review_20261003/review_corpus.jsonl',
    'models/phish_model.pkl', 'models/vectorizer.pkl', 'models/url_model.pkl',
    'models/domain_model.pkl', 'models/conformal_params.json',
    'models/conformal_evaluation_results.json']
AUDIT = ROOT / 'benchmarks/email_generalization_root_20261003'
OLD = ROOT / 'benchmarks/email_retraining_20261003'
SPECS = [
    dict(name='lgbm_word_structural', estimator='lgbm', structural_features=True),
    dict(name='lgbm_word_only', estimator='lgbm'),
    dict(name='linear_word_only', estimator='linear'),
    dict(name='linear_word_character', estimator='linear', character_features=True),
    dict(name='linear_repaired_word_character', estimator='linear', character_features=True, repaired_text=True),
    dict(name='linear_repaired_without_unresolved_spam', estimator='linear', character_features=True,
         repaired_text=True, omit_unresolved_spam=True),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    from utils.jsonl import read_jsonl as read_records
    return read_records(path)


def save(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def status(out, stage, **details):
    value = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **details)
    save(out / 'run_status.json', value)
    print(json.dumps(value), flush=True)


def validate_new_output(out):
    out = out.resolve()
    try:
        relative = out.relative_to((ROOT / 'benchmarks').resolve())
    except ValueError as error:
        raise ValueError('Research output must be a new folder within workspace benchmarks') from error
    if not relative.parts:
        raise ValueError('Research output must be a child folder, not benchmarks itself')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise RuntimeError('Research output is not empty; never overwrite or append to a previous run')
    return out


def metric(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Metric inputs must have matching shape and binary labels')
    y = y.astype(int)
    if np.isinf(p).any() or ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError('Invalid metric probabilities')
    valid = np.isfinite(p)
    result = {'total_rows': len(y), 'scored_rows': int(valid.sum()),
              'unavailable_rows': int((~valid).sum()), 'threshold': float(threshold)}
    y, p = y[valid], p[valid]
    if not len(y):
        return result
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel())
    ece = 0.
    for index in range(10):
        selected = (p >= index / 10) & (p < (index + 1) / 10 if index < 9 else p <= 1)
        if selected.any():
            ece += selected.mean() * abs(float(y[selected].mean() - p[selected].mean()))
    result.update(
        confusion_matrix=[[tn, fp], [fn, tp]], accuracy=(tn + tp) / len(y),
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        fpr=fp / (tn + fp) if tn + fp else None,
        fnr=fn / (tp + fn) if tp + fn else None,
        roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        average_precision=float(average_precision_score(y, p)) if y.sum() else None,
        brier=float(brier_score_loss(y, p)), ece_10_equal_width=float(ece))
    return result


def empirical_budget_threshold(y, p, budget=VALIDATION_FPR_BUDGET):
    """Select only on validation; ties must not exceed the empirical budget."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    metric(y, p)  # Validate intentional missing versus malformed probabilities.
    if not 0 <= budget < 1:
        raise ValueError('Validation FPR budget must be in [0, 1)')
    benign = p[(y == 0) & np.isfinite(p)]
    if not len(benign):
        raise ValueError('No scored validation negatives')
    allowed = int(np.floor(budget * len(benign)))
    ordered = np.sort(benign)[::-1]
    threshold = float(np.nextafter(ordered[allowed], np.inf))
    if int((benign >= threshold).sum()) > allowed:
        raise ValueError('Validation threshold exceeds frozen budget')
    return threshold


def score_candidate(model, texts):
    probabilities = np.full(len(texts), np.nan)
    for offset in range(0, len(texts), 64):
        batch = texts[offset:offset + 64]
        support = model.named_steps['features'].supported(batch)
        if support.any():
            values = model.predict_proba([text for text, ok in zip(batch, support) if ok])[:, 1]
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('Invalid candidate probabilities')
            probabilities[offset + np.flatnonzero(support)] = values
    return probabilities


def score_baseline(model, vectorizer, texts):
    probabilities = np.full(len(texts), np.nan)
    for offset in range(0, len(texts), 64):
        batch = texts[offset:offset + 64]
        word = vectorizer.transform([clean_and_mask_email_text(text) for text in batch])
        support = word.getnnz(axis=1) > 0
        features = hstack([word, csr_matrix(extract_features(batch))], format='csr')
        if support.any():
            values = np.asarray(model.predict_proba(features[support])[:, 1], dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('Invalid baseline probabilities')
            probabilities[offset + np.flatnonzero(support)] = values
    return probabilities


def omission_manifest(frame):
    reasons = defaultdict(list)
    for row in read_jsonl(AUDIT / 'full_message_overlap_review.jsonl'):
        if row['scope'] == 'internal_test' and row.get('full_unique_trigram_jaccard', 0) >= .90:
            reasons[int(row['train_row'])].append({'reason': 'observed_full_template_test_overlap',
                                                   'evaluation_row': row['row']})
    for row in json.loads((AUDIT / 'verified_original_message_overlap.json').read_text()):
        if any(split in ('val', 'test') for _, split in row['rows']):
            for index, split in row['rows']:
                if split == 'train':
                    reasons[index].append({'reason': 'observed_original_raw_message_overlap',
                                           'raw_sha256': row['raw_sha']})
    # Omit all same declared train-group members; this does not certify all groups.
    groups = set(frame.loc[list(reasons), 'campaign_cluster_id'])
    for index, row in frame[(frame.split == 'train') & frame.campaign_cluster_id.isin(groups)].iterrows():
        if index not in reasons:
            reasons[index].append({'reason': 'same_declared_group_as_observed_overlap'})
    uncertain = defaultdict(list)
    for row in read_jsonl(AUDIT / 'upstream_label_matches.jsonl'):
        if (row['split'] == 'train' and row['legacy_label'] == 1
                and row['upstream_label'].startswith('spam (not a phishing-specific annotation)')):
            uncertain[int(row['legacy_row'])].append({'reason': 'upstream_spam_not_phishing_adjudicated',
                                                     'raw_sha256': row['upstream_raw_sha256']})
    return reasons, uncertain


def bootstrap_auc(y, base, candidate):
    rng = np.random.default_rng(SEED)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = []
    for _ in range(1000):
        indices = np.r_[rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        deltas.append(roc_auc_score(y[indices], candidate[indices]) - roc_auc_score(y[indices], base[indices]))
    return {'paired_auc_difference': float(roc_auc_score(y, candidate) - roc_auc_score(y, base)),
            'row_stratified_bootstrap_95ci': np.quantile(deltas, [.025, .975]).tolist(),
            'resamples': 1000, 'seed': SEED,
            'scope': 'Descriptive, conditional on author labels; ignores unresolved campaign dependence'}


def append_predictions(path, name, split, part, probabilities):
    with path.open('a', encoding='utf-8') as stream:
        for row, p in zip(part, probabilities):
            stream.write(json.dumps(dict(model=name, split=split, row_id=row['id'],
                source_label=row['source_label'], label=row['label'],
                probability=float(p) if np.isfinite(p) else None)) + '\n')


def write_report(out, results):
    lines = ['# Email representation comparison', '',
        '**Deployment reliability remains unproven. No candidate was promoted.**', '',
        'This is a controlled development diagnosis on already inspected legacy and source-author labels. '
        'All deployed models, datasets and saved calibration remain unchanged. Unknown scores remain unavailable.', '',
        '| Model | Internal accuracy | Internal AUC | Paired N | External accuracy | External AUC | External FPR | External recall | Brier |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in results['models'].items():
        internal = value['splits']['test']['at_0_5']
        external = value['external_common_with_baseline']['at_0_5']
        lines.append(f"| {name} | {internal['accuracy']:.2%} | {internal['roc_auc']:.4f} | {external['scored_rows']} | "
                     f"{external['accuracy']:.2%} | {external['roc_auc']:.4f} | {external['fpr']:.2%} | "
                     f"{external['recall']:.2%} | {external['brier']:.4f} |")
    lines += ['', 'The original failing baseline is retained. Paired external denominators and any unavailable rows '
              'are recorded separately; full-cohort and Phishing-vs-Valid-only metrics are also in results.json.', '',
              '## Frozen validation budget', '',
              'Each threshold was selected only from validation negatives at a 1% empirical FPR budget, '
              'before test/external scoring. This is neither conformal calibration nor a deployment guarantee.', '',
              '| Model | Threshold | External FPR | External recall | Valid-only FPR |',
              '|---|---:|---:|---:|---:|']
    for name, value in results['models'].items():
        ext = value['external_common_with_baseline']['at_validation_budget']
        hard = value['external_phishing_vs_valid']['at_validation_budget']
        lines.append(f"| {name} | {value['validation_threshold']:.4f} | {ext['fpr']:.2%} | {ext['recall']:.2%} | {hard['fpr']:.2%} |")
    lines += ['', '## Representation sensitivity and resource scope', '',
              '| Model | Neutral-footer flips | HTML-wrapper flips | Fit seconds | Warm median ms | Model MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for name, value in results['models'].items():
        p = value['probes']
        timing = value['warm_single_message_latency']
        lines.append(f"| {name} | {p['footer']['decisions_changed_at_0_5']} / {p['footer']['paired_rows']} | "
                     f"{p['html_wrapper']['decisions_changed_at_0_5']} / {p['html_wrapper']['paired_rows']} | "
                     f"{value.get('fit_seconds', 0):.2f} | {timing['median_ms']:.2f} | {value['artifact_mib']:.2f} |")
    lines += ['', 'Timings cover local feature construction plus one content classifier only, on a shared laptop. '
              'They do not include OriginTrace, BEC, vision, network, startup or memory saved by the whole server. '
              'Serialized model size is disk size, not measured RAM.', '',
              f"Shared word vocabulary fitting took {results['shared_word_fit_seconds']:.2f}s; "
              'this common cost is excluded from each of the first three candidate fit times.', '',
              '## What the contrasts establish', '',
              '- The two new LightGBM arms share the exact word vocabulary/IDF, rows and classifier settings; '
              'their only feature difference is the 16 raw structural counters.',
              '- The deployed baseline is a calibrated five-fold ensemble; the new LightGBM arms are single estimators. '
              'Baseline-to-candidate differences combine several changes; only the new LightGBM pair isolates the counters.',
              '- Word and word/character linear arms test regularization/representation with fixed settings. '
              'The repaired-text contrast tests parser normalization, URL-marker preservation and hidden HTML content.',
              '- The final arm omits unresolved upstream-spam training rows. It is label-noise sensitivity, '
              'not an adjudicated replacement dataset. Omitted rows were never relabeled.',
              '- Observed template/raw overlap training groups were omitted for all new arms, without changing evaluation rows. '
              'Remaining rows are not certified zero-leakage.', '',
              '## Unresolved limitations', '',
              '- Source-author external labels were not independently adjudicated, and the cohort has been inspected repeatedly.',
              '- Training labels still have unresolved task/source provenance. Source identity strongly correlates with labels.',
              '- Body-only representations cannot recover absent sender, receiver authentication, original MIME or external link metadata.',
              '- The baseline retains its historical probability calibration. Candidates are uncalibrated; '
              'no new probability/conformal calibration was fitted or approved.',
              '- Row-bootstrap intervals ignore campaign dependence; an improvement here cannot certify unseen-source/time reliability.', '',
              'Every original and candidate score, omission and probe is retained in predictions.jsonl, '
              'training_omissions.jsonl and probe_predictions.jsonl. The protocol, hashes, convergence and validation thresholds '
              'are retained in protocol.json, fitted_models.json and results.json.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(out):
    out = validate_new_output(out)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest(ROOT / name) for name in PROTECTED}
    inputs = ['scripts/compare_email_representations.py', 'training/email_text_candidates.py',
              'utils/email_candidate_text.py', 'utils/text_sanitizer.py', 'utils/feature_extractor.py',
              'benchmarks/email_retraining_20261003/external_inputs.jsonl',
              'benchmarks/email_retraining_20261003/decisions.csv',
              'benchmarks/email_generalization_root_20261003/full_message_overlap_review.jsonl',
              'benchmarks/email_generalization_root_20261003/verified_original_message_overlap.json',
              'benchmarks/email_generalization_root_20261003/upstream_label_matches.jsonl']
    input_hashes = {name: digest(ROOT / name) for name in inputs}
    frame = pd.read_csv(ROOT / 'data/clean_email_corpus.csv', keep_default_na=False)
    overlap, uncertain = omission_manifest(frame)
    train = frame[(frame.split == 'train') & ~frame.index.isin(overlap)]
    reduced = train[~train.index.isin(uncertain)]
    for part in (train, reduced):
        if set(part.label) != {0, 1}:
            raise ValueError('Training arm lacks both classes')
    external = read_jsonl(OLD / 'external_inputs.jsonl')
    parts = {split: [dict(id=str(index), text=row.raw_text + ' ', label=int(row.label), source_label=str(row.label))
                   for index, row in frame[frame.split == split].iterrows()] for split in ('val', 'test')}
    parts['external'] = [dict(id=r['review_id'], text=r['text'], label=int(r['label']), source_label=r['source_label']) for r in external]
    omissions = [dict(row_id=str(index), scope='all_new_models', reasons=why) for index, why in overlap.items()]
    omissions += [dict(row_id=str(index), scope='unresolved_spam_sensitivity_arm_only', reasons=why) for index, why in uncertain.items()]
    (out / 'training_omissions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in omissions), encoding='utf-8')
    baseline = joblib.load(ROOT / 'models/phish_model.pkl')
    vectorizer = joblib.load(ROOT / 'models/vectorizer.pkl')
    if not np.array_equal(baseline.classes_, [0, 1]):
        raise ValueError('Baseline does not implement expected binary class order')
    for fold in baseline.calibrated_classifiers_:
        fold.estimator.set_params(n_jobs=1)
    base = clone(baseline.calibrated_classifiers_[0].estimator).set_params(n_jobs=1, verbosity=-1)
    protocol = dict(schema_version=1, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Previously inspected development/source-author comparison, not an untouched validation',
        protected_hashes=hashes, input_hashes=input_hashes, candidate_specs=SPECS,
        lgbm_params=base.get_params(), linear_params=dict(C=1.0, solver='liblinear', penalty='l2',
            class_weight='balanced', max_iter=500, random_state=SEED),
        representation=dict(word_limit=12000, character_limit=16000, min_df=3, dtype='float32'),
        train_rows=len(train), train_label_counts=dict(Counter(map(str, train.label))),
        spam_sensitivity_rows=len(reduced), spam_sensitivity_label_counts=dict(Counter(map(str, reduced.label))),
        validation_fpr_budget=VALIDATION_FPR_BUDGET,
        threshold_rule='next float above benign descending score at floor(budget*N); ties retained conservatively',
        test_based_search=False, recalibration=False, production_promotion=False,
        independent_label_review_completed=False, deployment_generalization_established=False,
        omissions='Remove observed overlap train groups; sensitivity removes unresolved spam only, never relabels',
        external_mapping={'Phishing': 1, 'Spam': 0, 'Valid': 0},
        confidence_intervals='Descriptive row-stratified; unresolved campaign dependence not modeled')
    save(out / 'protocol.json', protocol)
    status(out, 'protocol_frozen', train_rows=len(train), spam_sensitivity_rows=len(reduced), external_rows=len(external))
    results = dict(complete=False, protocol_sha256=digest(out / 'protocol.json'), models={},
                   approved_for_deployment=False, deployment_generalization_established=False)
    # Fit all candidates before any new external evaluation. No C/model search.
    shared_started = time.perf_counter()
    shared_word = CandidateEmailFeatures().fit((train.raw_text + ' ').tolist())
    results['shared_word_fit_seconds'] = time.perf_counter() - shared_started
    fitted = {}
    for spec in SPECS:
        name = spec['name']
        status(out, 'fitting', model=name)
        training = reduced if spec.get('omit_unresolved_spam') else train
        features = CandidateEmailFeatures(**{key: spec[key] for key in
            ('structural_features', 'character_features', 'repaired_text') if key in spec})
        started = time.perf_counter()
        before_memory = memory()
        training_texts = (training.raw_text + ' ').tolist()
        if not features.character_features and not features.repaired_text:
            # Exactly the same fitted word vocabulary/IDF for the structural pair.
            features.word_vectorizer_ = shared_word.word_vectorizer_
        else:
            features.fit(training_texts)
        matrix = features.transform(training_texts)
        estimator = clone(base) if spec['estimator'] == 'lgbm' else LogisticRegression(**protocol['linear_params'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            estimator.fit(matrix, training.label.to_numpy())
        convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        if convergence:
            raise RuntimeError('Candidate did not converge: ' + '; '.join(convergence))
        model = Pipeline([('features', features), ('classifier', estimator)])
        artifact = dict(schema_version=1, model=model, input_kind='development_raw_email_text',
                        approved_for_deployment=False, independently_reviewed=False,
                        protocol_sha256=results['protocol_sha256'])
        target = out / (name + '.pkl')
        joblib.dump(artifact, target)
        fitted[name] = dict(fit_seconds=time.perf_counter() - started, training_rows=len(training),
            feature_count=matrix.shape[1], training_nnz=int(matrix.nnz), dtype=str(matrix.dtype),
            shared_word_fit_cost_excluded=not features.character_features and not features.repaired_text,
            convergence_warnings=convergence, n_iter=np.asarray(getattr(estimator, 'n_iter_', [])).tolist(),
            artifact_sha256=digest(target), artifact_mib=target.stat().st_size / 2**20,
            process_memory_before_fit=before_memory, process_memory_after_fit=memory())
        status(out, 'fitted', model=name, **fitted[name])
        del model, artifact, estimator, features, matrix
        gc.collect()
    save(out / 'fitted_models.json', fitted)
    saved = pd.read_csv(OLD / 'decisions.csv', keep_default_na=False)
    baseline_values = {}
    baseline_threshold = None
    for split, part in parts.items():
        texts, y = [r['text'] for r in part], [r['label'] for r in part]
        p = score_baseline(baseline, vectorizer, texts)
        previous = saved[saved.split == split].set_index(saved[saved.split == split].row.astype(str))
        for row, value in zip(part, p):
            old = previous.loc[row['id'], 'baseline_probability']
            if old == '':
                if np.isfinite(value):
                    raise ValueError('Previously unavailable baseline row silently gained a score')
            elif not np.isfinite(value) or abs(value - float(old)) > 1e-8:
                raise ValueError('Recomputed baseline differs from retained original decisions')
        if split == 'val':
            baseline_threshold = empirical_budget_threshold(y, p)
        baseline_values[split] = p
        append_predictions(out / 'predictions.jsonl', 'deployed_baseline', split, part, p)
    results['original_baseline_external'] = metric(
        [row['label'] for row in parts['external']], baseline_values['external'])
    all_names = ['deployed_baseline'] + [s['name'] for s in SPECS]
    probes = {'footer': [r['text'] + '\nArchive reference: Enron 2001.' for r in parts['external']],
              'html_wrapper': ['<html><body>' + r['text'] + '</body></html>' for r in parts['external']]}
    for name in all_names:
        status(out, 'evaluating', model=name)
        if name == 'deployed_baseline':
            model = None
            score = lambda texts: score_baseline(baseline, vectorizer, texts)
            metadata = dict(fit_seconds=0, artifact_mib=(ROOT / 'models/phish_model.pkl').stat().st_size / 2**20
                            + (ROOT / 'models/vectorizer.pkl').stat().st_size / 2**20)
        else:
            target = out / (name + '.pkl')
            if digest(target) != fitted[name]['artifact_sha256']:
                raise ValueError('Candidate changed after fit freeze')
            model = joblib.load(target)['model']
            score = lambda texts: score_candidate(model, texts)
            metadata = fitted[name].copy()
        value = dict(**metadata, splits={}, probes={})
        probabilities = {}
        threshold = None
        for split, part in parts.items():
            p = baseline_values[split] if model is None else score([r['text'] for r in part])
            y = np.asarray([r['label'] for r in part])
            if split == 'val':
                threshold = empirical_budget_threshold(y, p)
                # Threshold is persisted before scoring external/test with this candidate.
                value['validation_threshold'] = threshold
                save(out / (name + '_threshold.json'), dict(threshold=threshold, budget=VALIDATION_FPR_BUDGET,
                     selected_split='val', candidate=name, fitted_model_sha256=metadata.get('artifact_sha256'),
                     protocol_sha256=results['protocol_sha256']))
            probabilities[split] = p
            value['splits'][split] = dict(at_0_5=metric(y, p), at_validation_budget=metric(y, p, threshold))
            if model is not None:
                append_predictions(out / 'predictions.jsonl', name, split, part, p)
        y = np.asarray([r['label'] for r in parts['external']])
        p = probabilities['external']
        common = np.isfinite(p) & np.isfinite(baseline_values['external'])
        valid_only = np.asarray([r['source_label'] in ('Phishing', 'Valid') for r in parts['external']])
        value['external_common_with_baseline'] = dict(at_0_5=metric(y[common], p[common]),
            at_validation_budget=metric(y[common], p[common], threshold),
            baseline_at_0_5_same_rows=metric(y[common], baseline_values['external'][common]),
            baseline_at_validation_budget_same_rows=metric(y[common], baseline_values['external'][common], baseline_threshold),
            paired_change=bootstrap_auc(y[common], baseline_values['external'][common], p[common]))
        value['external_phishing_vs_valid'] = dict(at_0_5=metric(y[valid_only], p[valid_only]),
            at_validation_budget=metric(y[valid_only], p[valid_only], threshold))
        for probe_name, texts in probes.items():
            after = score(texts)
            paired = np.isfinite(p) & np.isfinite(after)
            if probe_name == 'html_wrapper':
                canonical = model.named_steps['features'].canonicalize if model is not None else lambda texts: list(map(clean_and_mask_email_text, texts))
                invariant = np.asarray([a == b for a, b in zip(canonical([r['text'] for r in parts['external']]), canonical(texts))])
                paired &= invariant
            delta = after[paired] - p[paired]
            value['probes'][probe_name] = dict(paired_rows=int(paired.sum()),
                decisions_changed_at_0_5=int(((p[paired] >= .5) != (after[paired] >= .5)).sum()),
                mean_absolute_probability_change=float(np.abs(delta).mean()),
                max_absolute_probability_change=float(np.abs(delta).max()),
                scope='Representation sensitivity only; not new security labels or accuracy')
            with (out / 'probe_predictions.jsonl').open('a', encoding='utf-8') as stream:
                for row, before, q, included in zip(parts['external'], p, after, paired):
                    stream.write(json.dumps(dict(model=name, probe=probe_name, row_id=row['id'],
                        before=float(before) if np.isfinite(before) else None,
                        after=float(q) if np.isfinite(q) else None, included_in_pair=bool(included))) + '\n')
        eligible = [r['text'] for r, q in zip(parts['external'], p) if np.isfinite(q)][:10]
        for text in eligible:
            score([text])
        timing = []
        for index in range(30):
            started = time.perf_counter()
            score([eligible[index % len(eligible)]])
            timing.append((time.perf_counter() - started) * 1000)
        value['warm_single_message_latency'] = dict(runs=30, median_ms=float(np.median(timing)),
            p95_ms=float(np.quantile(timing, .95)), scope='Features + content classifier, excluding engine/network/startup')
        results['models'][name] = value
        save(out / 'evaluation_checkpoint.json', results)
        del model
        gc.collect()
    if any(digest(ROOT / name) != h for name, h in {**hashes, **input_hashes}.items()):
        raise RuntimeError('Protected/source artifact changed during comparison')
    results.update(complete=True, protected_artifacts_unchanged=True, source_artifacts_unchanged=True,
                   no_candidate_promoted=True, no_calibration_fitted=True,
                   final_experiment_process_memory=memory())
    write_report(out, results)
    save(out / 'results.json', results)
    status(out, 'completed', models=len(results['models']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    owns_output = False
    try:
        output = validate_new_output(arguments.output)
        owns_output = True
        main(output)
    except Exception as error:
        if owns_output and output.exists():
            status(output, 'failed', error=repr(error))
        raise
