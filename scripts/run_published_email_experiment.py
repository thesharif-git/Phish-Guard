"""Frozen, research-only published-label Email experiment.

Prepare a protocol first; fitting requires a separate --execute-prepared call.
Publisher expert labels and source-author labels retain their actual authority.
This never approves a model, manufactures independent review or fits calibration.
"""
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
import csv
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from training.email_evidence import group_email_templates
from training.email_evidence_model import (
    EMAIL_EVIDENCE_MODEL_INPUT_KIND, EMAIL_INPUT_CONTRACT_VERSION, MODEL_PARAMS,
    contract_source_hashes, email_evidence_estimator, library_versions,
    predict_email_evidence_probabilities,
)
from training.email_source_annotations import load_spaphish_publisher_annotations
from utils.email_candidate_text import canonical_email_candidate_text
from utils.email_input_contract import build_email_input, contract_sha256, evidence_sha256, validate_email_input
from utils.jsonl import read_jsonl
from utils.email_parser import visible_html_text
from scripts.prepare_email_evidence_contract_corpus import declared_url_list
from services.security_pipeline import security_pipeline_id
from scripts.train_email_evidence_candidate import metrics, validation_threshold, validate_output
from scripts.compare_email_representations import score_baseline, score_candidate
from scripts.profile_email_runtime import memory

SEED = 20261005
FPR_BUDGET = .01
FF_RELEASE_HASH = '394d542e27dcef57321bc69f0f93b03e5965531f616eb3d3c79416cc60a30765'
OLD_CANDIDATES = ('linear_word_only', 'linear_repaired_without_unresolved_spam')
ARMS = (
    dict(name='word_only', character_features=False, evidence_features=False),
    dict(name='text_only', character_features=True, evidence_features=False),
    dict(name='structured', character_features=True, evidence_features=True),
)
POOLS = ('published_spanish_only', 'published_spanish_plus_source_author_english')
FIT_JOBS = tuple(dict(pool=pool, arm=arm['name'], balance_source_classes=False)
                 for pool in POOLS for arm in ARMS) + tuple(
    dict(pool=POOLS[1], arm=arm, balance_source_classes=True)
    for arm in ('text_only', 'structured'))
PROTECTED = (
    'models/phish_model.pkl', 'models/vectorizer.pkl', 'models/url_model.pkl',
    'models/domain_model.pkl', 'models/conformal_params.json',
    'models/conformal_evaluation_results.json', 'data/clean_email_corpus.csv',
    'data/email_evidence_review_20261003/review_corpus.jsonl',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def save_jsonl(path, records):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def status(out, stage, **details):
    value = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **details)
    save(out / 'run_status.json', value)
    print(json.dumps(value), flush=True)


def source_class_weights(rows):
    """Train-only weights equalize observed source/class cells, not scores.

    Labels and source IDs are used only for training weights, never features.
    This comparison is frozen before any held-out model score is computed.
    """
    counts = Counter((row['source'], row['label']) for row in rows)
    if not counts or any((source, label) not in counts
                         for source in {row['source'] for row in rows} for label in (0, 1)):
        raise ValueError('Source/class balance requires both labels in every training source')
    return np.asarray([len(rows) / (len(counts) * counts[(row['source'], row['label'])])
                       for row in rows], dtype=float)


def code_hashes():
    values = contract_source_hashes(ROOT)
    for name in ('utils/jsonl.py', 'training/email_source_annotations.py',
                 'scripts/run_published_email_experiment.py',
                 'scripts/prepare_email_evidence_contract_corpus.py', 'utils/email_parser.py',
                 'scripts/train_email_evidence_candidate.py',
                 'scripts/compare_email_representations.py',
                 'training/email_text_candidates.py', 'utils/feature_extractor.py',
                 'utils/email_parser.py', 'scripts/prepare_email_evidence_contract_corpus.py'):
        values[name] = digest(ROOT / name)
    return values


def research_record(row, annotation):
    contract = validate_email_input(row['email_input'])
    if row.get('input_contract_sha256') != contract_sha256(contract):
        raise ValueError('Curated input hash differs: ' + row['id'])
    return dict(id=row['id'], source=row['source'], task_label=annotation['task_label'],
        label=1 if annotation['task_label'] == 'phishing' else 0,
        email_input=contract, input_contract_sha256=row['input_contract_sha256'],
        original_reference=row['original_reference'], quality_flags=row.get('quality_flags', []),
        annotation=annotation, project_independent_review_completed=False,
        training_reuse_permitted=row.get('training_reuse_permitted'),
        training_reuse_authority=row.get('training_reuse_authority'),
        approved_for_deployment=False)


def published_records(review_path, spa_directory, ff_path):
    """Bind usable source annotations; never copy model predictions into labels."""
    reviewed = read_jsonl(review_path)
    lookup = {row['id']: row for row in reviewed}
    if len(lookup) != len(reviewed):
        raise ValueError('Duplicate curated IDs')
    annotations = load_spaphish_publisher_annotations(spa_directory)
    with (Path(spa_directory) / 'corpus.csv').open(encoding='utf-8-sig',newline='') as handle:
        source_spa=list(csv.DictReader(handle,delimiter=';'))
    spa = []
    for annotation in annotations['annotations']:
        row = lookup.get(annotation['id'])
        if row is None or row.get('source') != 'spaphish_v4':
            raise ValueError('Released SpaPhish row missing from shared-contract corpus')
        if row['original_reference'].get('source_sha256') != annotations['source_corpus_sha256']:
            raise ValueError('SpaPhish source provenance differs')
        index=annotation['source_row_index']
        if row['original_reference'].get('row_index') != index:
            raise ValueError('SpaPhish source row index differs')
        original=source_spa[index]
        urls,url_flags=declared_url_list(original['urls'])
        expected=build_email_input(original['body'],subject=original['subject'],supplementary_urls=urls,
            parsing_info={'source_kind':'publisher_csv','errors':['incomplete_source_url_metadata']}
                if {'incomplete_source_url_metadata','unparsed_source_url_metadata'} & set(url_flags) else None)
        if (row.get('source_claimed_label') != ('publisher_phishing' if original['Label']=='1' else 'publisher_legitimate')
                or contract_sha256(expected) != row.get('input_contract_sha256')):
            raise ValueError('SpaPhish shared-contract raw fields differ from released source CSV')
        spa.append(research_record(row, annotation))
    if digest(ff_path) != FF_RELEASE_HASH:
        raise ValueError('Original PhishFuzzer release hash differs')
    originals = json.loads(Path(ff_path).read_text(encoding='utf-8'))
    expected_manual = {index for index, original in enumerate(originals)
                       if original.get('Created by') == 'Human' and original.get('Source') == 'Manual'}
    seen_manual = set()
    english, audit = [], []
    for row in reviewed:
        if row.get('source') != 'phishfuzzer_manual':
            continue
        reference = row['original_reference']
        index = reference.get('row_index')
        if type(index) is not int or not 0 <= index < len(originals) or reference.get('source_sha256') != FF_RELEASE_HASH:
            raise ValueError('Unbound PhishFuzzer source index/hash')
        original = originals[index]
        if index in seen_manual:
            raise ValueError('Duplicate PhishFuzzer source row index')
        seen_manual.add(index)
        if (original.get('Created by') != 'Human' or original.get('Source') != 'Manual'
                or original.get('Type') != row.get('source_claimed_label')):
            raise ValueError('PhishFuzzer row is not the declared original author sample')
        body=original.get('Body') or ''
        html=body if any(tag in body.lower() for tag in ('<html','<body','<p>','<a ')) else ''
        urls=original.get('URL') or []
        urls=urls if isinstance(urls,list) else [urls]
        if not all(isinstance(url,str) for url in urls):
            raise ValueError('Original PhishFuzzer URL metadata contains non-string evidence')
        expected=build_email_input(visible_html_text(html) if html else body,
            subject=original.get('Subject') or '',sender=original.get('Sender') or '',
            html_body=html,supplementary_urls=urls)
        if contract_sha256(expected) != row.get('input_contract_sha256'):
            raise ValueError('PhishFuzzer shared-contract raw fields differ from the original source message')
        if original['Type'] == 'Spam':
            audit.append(dict(id=row['id'], source='phishfuzzer_manual', reason='ordinary_spam_excluded_from_phishing_task',
                              original_reference=reference, source_claimed_label='Spam'))
            continue
        if original['Type'] not in ('Phishing', 'Valid'):
            raise ValueError('Unrecognized source-author task label')
        if original.get('Language') != 'en':
            audit.append(dict(id=row['id'], source='phishfuzzer_manual',
                reason='non_English_author_language_outside_English_development_scope',
                original_reference=reference, source_claimed_label=original['Type'],
                source_claimed_language=original.get('Language')))
            continue
        annotation = dict(task_label='phishing' if original['Type'] == 'Phishing' else 'legitimate',
            label_authority_kind='source_author_manual_label', source_claimed_label=original['Type'],
            label_authority_reference='https://github.com/DataPhish/PhishFuzzer',
            project_independent_review_completed=False, project_reviewer_ids=[],
            observed_day=None, time_authority='unverified_author_year', language_scope='English source-author assertion',
            training_reuse_permission_established=False,
            note='Previously inspected 292-row external comparison becomes development data; source-author labels are not independently reviewed gold')
        english.append(research_record(row, annotation))
    if not seen_manual or not seen_manual.issubset(expected_manual):
        raise ValueError('Source-author review rows fall outside the original manual release subset')
    for index in sorted(expected_manual - seen_manual):
        original = originals[index]
        audit.append(dict(id='original-manual-seed-'+str(index), source='phishfuzzer_manual',
            reason='original_manual_seed_outside_frozen_previous_comparison_subset',
            original_reference=dict(source_sha256=FF_RELEASE_HASH, row_index=index),
            source_claimed_label=original.get('Type'), source_claimed_language=original.get('Language'),
            note='Source remains in the hash-bound original release; exclusion is inherited from the preexisting comparison curation, not model-score based'))
    if {row['task_label'] for row in english} != {'phishing', 'legitimate'}:
        raise ValueError('English source-author development data lacks both task classes')
    return spa, english, audit, annotations


def partition_sources(spa, english, brown):
    """Hold later/source-test evidence fixed; remove overlapping earlier groups.

    This performs no model-based selection. Dates/labels remain out of features.
    The English group split is deterministic and its chronology is unverified.
    """
    records = spa + english + brown
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('Research source IDs must be distinct')
    working, audit = [], []
    for row in spa + english:
        item = dict(row)
        evidence = validate_email_input(item['email_input'])
        reason = None
        if evidence['parsing']['completeness'] == 'INCOMPLETE':
            reason = 'documented_incomplete_extraction'
        elif not canonical_email_candidate_text(evidence['body'] or evidence['html_body']):
            reason = 'no_observed_message_body_url_or_attachment_only_retained_in_audit'
        if item['source'] == 'spaphish_v4':
            day = item['annotation']['observed_day']
            if day is None:
                reason = reason or 'missing_publisher_reception_day'
            item['split'] = ('train' if day and day < '2024-01-01' else
                             'val' if day and day < '2025-01-01' else 'published_day_test')
        else:
            item['split'] = 'english_group_pending'
        if reason:
            audit.append(dict(id=item['id'], source=item['source'], reason=reason,
                              original_reference=item['original_reference']))
        else:
            working.append(item)
    # English groups are computed on their source before introducing Brown.
    ff = [row for row in working if row['source'] == 'phishfuzzer_manual']
    ff_groups, _ = group_email_templates([
        dict(id=r['id'], body=r['email_input']['body'] or r['email_input']['html_body'],
             subject=r['email_input']['subject']) for r in ff], canonicalizer=canonical_email_candidate_text)
    for row, group in zip(ff, ff_groups):
        row['split'] = 'val' if int(hashlib.sha256(group.encode()).hexdigest(), 16) % 5 == 0 else 'train'
        row['english_development_group'] = group
    # Exact duplicate complete evidence: quarantine conflicts, keep later
    # chronology first and one stable record per class/complete input.
    exact = defaultdict(list)
    for row in working:
        exact[evidence_sha256(row['email_input'])].append(row)
    rejected = set()
    priority = {'train': 0, 'val': 1, 'published_day_test': 2}
    for fingerprint, duplicates in exact.items():
        if len(duplicates) < 2:
            continue
        if len({row['label'] for row in duplicates}) > 1:
            chosen = None
            reason = 'conflicting_source_labels_for_identical_complete_evidence'
        else:
            chosen = sorted(duplicates, key=lambda r: (-priority[r['split']], r['id']))[0]['id']
            reason = 'identical_complete_evidence_duplicate'
        for row in duplicates:
            if row['id'] != chosen:
                rejected.add(row['id'])
                audit.append(dict(id=row['id'], source=row['source'], reason=reason,
                    complete_evidence_sha256=fingerprint, retained_id=chosen,
                    source_labels=[r['task_label'] for r in duplicates]))
    working = [row for row in working if row['id'] not in rejected]
    all_group_rows = working + brown
    group_ids, pairs = group_email_templates([
        dict(id=r['id'], body=r['email_input']['body'] or r['email_input']['html_body'],
             subject=r['email_input']['subject']) for r in all_group_rows], canonicalizer=canonical_email_candidate_text)
    groups = defaultdict(list)
    for row, group in zip(all_group_rows, group_ids):
        row['experiment_template_group'] = group
        groups[group].append(row)
    for group, members in groups.items():
        held_brown = [row for row in members if row['source'] == 'brown_phish_bowl']
        latest = max([priority[row['split']] for row in members if 'split' in row], default=-1)
        for row in members:
            if row['source'] == 'brown_phish_bowl':
                continue
            if held_brown or priority[row['split']] < latest:
                rejected.add(row['id'])
                audit.append(dict(id=row['id'], source=row['source'], reason='earlier_template_overlaps_frozen_later_or_Brown_evidence',
                    experiment_template_group=group, original_split=row['split'],
                    held_out_ids=[r['id'] for r in held_brown] if held_brown else
                        [r['id'] for r in members if r.get('split') and priority[r['split']] == latest]))
    usable = [row for row in working if row['id'] not in rejected]
    pools = {}
    for pool in POOLS:
        selected = [row for row in usable if pool == POOLS[1] or row['source'] == 'spaphish_v4']
        parts = {split: [r for r in selected if r['split'] == split] for split in ('train', 'val', 'published_day_test')}
        for split, part in parts.items():
            if set(row['label'] for row in part) != {0, 1}:
                raise ValueError(f'{pool}/{split} lacks both task classes; no fitting allowed')
        pools[pool] = parts
    return pools, audit, pairs


def legacy_metadata(directory):
    directory = Path(directory)
    results = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    if results.get('complete') is not True:
        raise ValueError('Previously frozen English comparison is incomplete')
    protocol_hash = digest(directory / 'protocol.json')
    values, paths = {}, [directory / 'protocol.json', directory / 'results.json']
    for name in ('deployed_baseline',) + OLD_CANDIDATES:
        threshold_path = directory / (name + '_threshold.json')
        threshold = json.loads(threshold_path.read_text(encoding='utf-8'))
        if (threshold.get('selected_split') != 'val' or threshold.get('budget') != FPR_BUDGET
                or threshold.get('candidate') != name or threshold.get('protocol_sha256') != protocol_hash
                or threshold.get('threshold') != results['models'][name]['validation_threshold']):
            raise ValueError('Old English validation threshold binding differs')
        values[name] = dict(threshold=threshold['threshold'], threshold_source=str(threshold_path.resolve()),
            retained_internal_development_test=results['models'][name]['splits']['test'],
            retained_internal_scope='Previously computed legacy internal test, unchanged model/data; not recomputed on SpaPhish or treated as comparable dataset difficulty')
        paths.append(threshold_path)
        if name == 'deployed_baseline':
            for model_name in ('models/phish_model.pkl', 'models/vectorizer.pkl'):
                if digest(ROOT / model_name) != protocol['protected_hashes'][model_name]:
                    raise ValueError('Deployed baseline differs from the frozen comparison')
        else:
            model_path = directory / (name + '.pkl')
            if digest(model_path) != threshold.get('fitted_model_sha256'):
                raise ValueError('Old candidate changed after threshold freeze')
            values[name]['model_path'] = str(model_path.resolve())
            paths.append(model_path)
    return values, paths


def prepare_experiment(review, spa, brown_directory, ff, old, output):
    output = validate_output(output)
    review = Path(review).resolve()
    spa, brown_directory, ff, old = map(lambda p: Path(p).resolve(), (spa, brown_directory, ff, old))
    manifest_path = review.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('review_corpus_sha256') != digest(review):
        raise ValueError('Prepared shared-contract review corpus hash mismatch')
    spa_rows, english_rows, excluded_spam, annotations = published_records(review, spa, ff)
    brown_manifest = json.loads((brown_directory / 'manifest.json').read_text(encoding='utf-8'))
    brown = read_jsonl(brown_directory / 'all_records.jsonl')
    if (brown_manifest.get('complete') is not True or
            brown_manifest.get('all_records_sha256') != digest(brown_directory / 'all_records.jsonl') or
            brown_manifest.get('evaluation_records_sha256') != digest(brown_directory / 'evaluation_records.jsonl') or
            brown_manifest.get('protocol_sha256') != digest(brown_directory / 'protocol.json')):
        raise ValueError('Brown preparation manifest is incomplete or its byte hashes differ')
    if not brown or len({r['id'] for r in brown}) != len(brown):
        raise ValueError('Brown benchmark has no unique frozen source rows')
    for row in brown:
        if (row.get('source') != 'brown_phish_bowl' or row.get('task_label') not in ('phishing', 'legitimate')
                or type(row.get('eligible_for_scoring')) is not bool or not isinstance(row.get('exclusion_reasons'), list)
                or row['eligible_for_scoring'] == bool(row['exclusion_reasons'])
                or row.get('input_contract_sha256') != contract_sha256(row['email_input'])
                or row.get('label') != (1 if row['task_label'] == 'phishing' else 0)):
            raise ValueError('Brown frozen record contract/eligibility/label differs')
    eligible_rows = read_jsonl(brown_directory / 'evaluation_records.jsonl')
    if ({row['id'] for row in eligible_rows} != {row['id'] for row in brown if row['eligible_for_scoring']} or
            len(brown) != brown_manifest.get('all_records') or
            len(eligible_rows) != brown_manifest.get('eligible_text_records')):
        raise ValueError('Brown frozen evaluation subset or denominator differs')
    # The Brown preparation manifest owns source/extraction eligibility. Capture
    # every file byte; source labels never enter the feature transformer.
    pools, audit, pairs = partition_sources(spa_rows, english_rows, brown)
    legacy, legacy_paths = legacy_metadata(old)
    paths = [review, manifest_path, ff, ff.parent / 'manifest.json', ff.parent / 'ReadME.md']
    paths += [spa / name for name in ('corpus.csv','schema.json','README.txt','acquisition.json')]
    paths += [p for p in brown_directory.iterdir() if p.is_file()] + legacy_paths
    input_hashes = {str(path.resolve()): digest(path) for path in paths}
    protected = {name: digest(ROOT / name) for name in PROTECTED}
    partition_rows = [dict(row, pool=pool) for pool, parts in pools.items() for split, rows in parts.items() for row in rows]
    summary = {pool: {split: dict(rows=len(rows), labels=dict(Counter(r['task_label'] for r in rows)),
                        sources=dict(Counter(r['source'] for r in rows))) for split, rows in parts.items()} for pool, parts in pools.items()}
    protocol = dict(schema_version=1, experiment='published_email_task_input_repair',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), classifier_params=MODEL_PARAMS,
        arms=ARMS, pools=POOLS, fit_jobs=FIT_JOBS,
        source_class_balance_rule='Train-only n/(number_of_observed_source_class_cells*cell_count); classifier class_weight=None in balanced contrasts',
        feature_limits=dict(word=12000, character=16000, identity=8000, min_df=3),
        input_kind=EMAIL_EVIDENCE_MODEL_INPUT_KIND, input_contract_version=EMAIL_INPUT_CONTRACT_VERSION,
        validation_fpr_budget=FPR_BUDGET, English_development_group_rule='sha256(source-template-group) modulo 5 == 0 -> validation',
        Spanish_day_boundaries=dict(train='before2024', val='2024', published_day_test='2025'),
        later_evidence_rule='Keep held-out rows fixed, remove overlapping earlier complete templates; no model-based exclusions',
        rows=summary, source_epoch_summary=annotations['epoch_summary'],
        Brown_total_rows=len(brown), Brown_eligible_rows=sum(r['eligible_for_scoring'] for r in brown),
        Brown_label_counts=dict(Counter(r['task_label'] for r in brown)),
        Brown_manifest=brown_manifest, Brown_records_path=str((brown_directory / 'all_records.jsonl').resolve()),
        original_English_models=legacy, input_hashes=input_hashes, protected_hashes=protected,
        code_hashes=code_hashes(), library_versions=library_versions(), pipeline_id=security_pipeline_id(),
        approved_for_deployment=False, deployment_generalization_established=False, calibration_fitted=False,
        independent_project_label_review_completed=False, model_selection_on_Brown=False,
        limitations=[
            'SpaPhish publisher expert labels are legitimate research annotations, not new project review',
            'PhishFuzzer labels are previously inspected source-author development labels, not independent gold',
            'PhishFuzzer training reuse permission remains unestablished; no permission flag is manufactured',
            'Spanish publisher dates have day precision and no trusted receiver time',
            'The mixed pool is not chronologically separated overall because English dates are unverified',
            'The SpaPhish2025 cohort tests publisher-reported chronology within SpaPhish, not all training data',
            'Brown is an English publisher excerpt/source test; original MIME, receiver authentication and attachment evidence are missing',
            'Brown extraction quality and overlap exclusions are frozen before fitting; all rows remain in coverage and diagnostic outputs',
            'This experiment is uncalibrated research; raw-probability calibration metrics do not confer valid calibration',
            'Neither empirical validation FPR thresholds nor row-based intervals establish a deployment FPR/FDR guarantee',
        ])
    output.mkdir(parents=True, exist_ok=False)
    save_jsonl(output / 'partition_rows.jsonl', partition_rows)
    save_jsonl(output / 'row_audit.jsonl', excluded_spam + audit)
    save_jsonl(output / 'template_pairs.jsonl', pairs)
    protocol['prepared_output_hashes'] = {name: digest(output / name) for name in ('partition_rows.jsonl','row_audit.jsonl','template_pairs.jsonl')}
    save(output / 'protocol.json', protocol)
    status(output, 'prepared', protocol_sha256=digest(output / 'protocol.json'), rows=summary,
           Brown_eligible_rows=protocol['Brown_eligible_rows'], models_fitted=0)
    return protocol


def assert_frozen(out, protocol):
    if protocol['code_hashes'] != code_hashes() or protocol['library_versions'] != library_versions():
        raise ValueError('Experiment code/runtime changed after protocol freeze; prepare a new run')
    if protocol['pipeline_id'] != security_pipeline_id():
        raise ValueError('Production pipeline changed after protocol freeze')
    for path, expected in protocol['input_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Frozen source/model input changed: ' + path)
    for path, expected in protocol['protected_hashes'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Protected production input changed: ' + path)
    for path, expected in protocol['prepared_output_hashes'].items():
        if digest(out / path) != expected:
            raise ValueError('Prepared experiment input changed: ' + path)


def auc_interval(y, p, *, resamples=500):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    supported = np.isfinite(p)
    y, p = y[supported], p[supported]
    if set(y) != {0, 1}:
        return None
    indices = [np.flatnonzero(y == label) for label in (0, 1)]
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(resamples):
        draw = np.concatenate([rng.choice(part, len(part), replace=True) for part in indices])
        values.append(roc_auc_score(y[draw], p[draw]))
    return dict(row_stratified_bootstrap_95ci=np.quantile(values, [.025,.975]).tolist(),
        resamples=resamples, seed=SEED, assumption='Descriptive row independence; template grouping does not prove campaign independence')


def report_metrics(rows, p, threshold, *, eligible=None):
    y = np.asarray([row['label'] for row in rows])
    selected = np.ones(len(rows), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    result = dict(total_rows=len(rows), eligibility_selected_rows=int(selected.sum()),
                  quality_excluded_rows=int((~selected).sum()),
                  scored_rows=int(np.isfinite(p).sum()), unavailable_rows=int((~np.isfinite(p)).sum()))
    result['eligible_cohort'] = dict(at_0_5=metrics(y[selected], p[selected]),
        at_frozen_validation_budget=metrics(y[selected], p[selected], threshold),
        auc_interval=auc_interval(y[selected], p[selected]))
    result['all_source_labeled_rows_diagnostic'] = dict(at_0_5=metrics(y,p),
        at_frozen_validation_budget=metrics(y,p,threshold),
        warning='Includes frozen quality exclusions; cannot be treated as clean task-valid headline performance')
    result['quality_exclusion_reasons'] = dict(Counter(reason for row in rows for reason in row.get('exclusion_reasons', [])))
    result['per_source']={}
    for source in sorted({row['source'] for row in rows}):
        source_mask=np.asarray([row['source']==source for row in rows])
        mask=source_mask & selected
        result['per_source'][source]=dict(total_rows=int(source_mask.sum()),eligible_rows=int(mask.sum()),
            at_0_5=metrics(y[mask],p[mask]),at_frozen_validation_budget=metrics(y[mask],p[mask],threshold))
    return result


def append_predictions(out, name, split, rows, p, threshold):
    with (out / 'predictions.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        for row, value in zip(rows, p):
            supported = bool(np.isfinite(value))
            stream.write(json.dumps(dict(model=name, split=split, row_id=row['id'], source=row['source'],
                task_label=row['task_label'], source_label_authority=row.get('annotation', {}).get('label_authority_kind', 'publisher_security_team_verdict'),
                label=row['label'], input_contract_sha256=row['input_contract_sha256'],
                probability=float(value) if supported else None,
                state='SCORED' if supported else 'UNAVAILABLE_REVIEW_REQUIRED',
                eligible_for_headline_scoring=row.get('eligible_for_scoring',True),
                exclusion_reasons=row.get('exclusion_reasons', []),
                prediction_at_0_5=int(value >= .5) if supported else None,
                prediction_at_frozen_validation_budget=int(value >= threshold) if supported else None,
                threshold=threshold, availability=row['email_input']['availability']),
                ensure_ascii=False, allow_nan=False) + '\n')


def timed_scores(fn, values):
    before = memory()
    wall, cpu = time.perf_counter(), time.process_time()
    p = np.asarray(fn(values), dtype=float)
    elapsed, cpu_elapsed = time.perf_counter() - wall, time.process_time() - cpu
    single = []
    for value in values[:32]:
        started = time.perf_counter()
        fn([value])
        single.append((time.perf_counter() - started) * 1000)
    return p, dict(batch_wall_seconds=elapsed, batch_process_cpu_seconds=cpu_elapsed,
        total_rows=len(values), batch_per_input_ms=elapsed*1000/len(values) if values else None,
        memory_before=before,memory_after=memory(),
        warm_single_message=dict(sample_size=len(single), median_ms=float(np.median(single)) if single else None,
            p95_ms=float(np.percentile(single,95)) if single else None,
            scope='Model and input transform only; excludes OriginTrace network/auxiliary sensors, startup and database'))


def paired_comparison(rows, before, after, before_threshold, after_threshold, *, baseline=None):
    selected = np.asarray([r['eligible_for_scoring'] for r in rows], dtype=bool)
    common = selected & np.isfinite(before) & np.isfinite(after)
    if baseline is not None:common &= np.isfinite(baseline)
    y = np.asarray([r['label'] for r in rows])[common]
    a, b = before[common], after[common]
    left, right = metrics(y,a),metrics(y,b)
    result = dict(total_source_rows=len(rows),eligible_rows=int(selected.sum()),
        common_supported_rows=int(common.sum()),before_at_0_5=left,after_at_0_5=right,
        before_at_frozen_budget=metrics(y,a,before_threshold),after_at_frozen_budget=metrics(y,b,after_threshold),
        decision_changes_at_0_5=dict(
            negatives_newly_false_positive=int(((y==0)&(a<.5)&(b>=.5)).sum()),
            negatives_corrected=int(((y==0)&(a>=.5)&(b<.5)).sum()),
            positives_newly_detected=int(((y==1)&(a<.5)&(b>=.5)).sum()),
            positives_newly_missed=int(((y==1)&(a>=.5)&(b<.5)).sum())))
    if len(y):
        result['accuracy_difference']=right['accuracy']-left['accuracy']
        result['brier_difference']=right['brier']-left['brier']
    if set(y)=={0,1}:
        result['roc_auc_difference']=right['roc_auc']-left['roc_auc']
    result['scope']='Same frozen Brown eligible rows; source labels and excerpt quality remain limitations'
    return result


def neutral_footer_inputs(rows):
    """A paired lexical sensitivity probe, not another training/evaluation set."""
    changed=[]
    for row in rows:
        original=row['email_input']
        changed.append(build_email_input(original['body']+'\n\nArchive reference: Enron 2001.',
            **{key:original[key] for key in ('subject','sender','headers','html_body','supplementary_urls','parsing_info')}))
    return changed


def footer_report(rows, before, after, threshold, *, baseline=None):
    mask=np.asarray([row['eligible_for_scoring'] for row in rows],dtype=bool)
    mask &= np.isfinite(before) & np.isfinite(after)
    if baseline is not None:mask &= np.isfinite(baseline)
    delta=after[mask]-before[mask]
    return dict(eligible_rows=sum(row['eligible_for_scoring'] for row in rows),paired_supported_rows=int(mask.sum()),
        decisions_changed_at_0_5=int(((before[mask]>=.5)!=(after[mask]>=.5)).sum()),
        decisions_changed_at_frozen_budget=int(((before[mask]>=threshold)!=(after[mask]>=threshold)).sum()),
        mean_signed_probability_change=float(delta.mean()) if len(delta) else None,
        mean_absolute_probability_change=float(np.abs(delta).mean()) if len(delta) else None,
        maximum_absolute_probability_change=float(np.abs(delta).max()) if len(delta) else None,
        scope='Same frozen Email with neutral archive footer; robustness sensitivity, not new accuracy or ground truth')


def write_report(out, results, protocol):
    def number(value, percentage=False):
        return 'unavailable' if value is None else f'{value:.2%}' if percentage else f'{value:.4f}'
    lines = ['# Published-label Email repair experiment','',
        '**Research only. No model or calibration was approved or deployed.**','',
        'The Spanish-only and Spanish plus English source-author pools use the same fixed word, word+character and structured models. '
        'All eight fits and validation thresholds are frozen before Brown scoring. Two mixed-pool contrasts equalize training source/class cell weights without adding features. Brown is a separate publisher English excerpt source; '
        'the previous PhishFuzzer comparison is now development data for the mixed pool.','',
        '| Model | Brown eligible/scored | Accuracy | AUC | FPR at .5 | Recall at .5 | Brier | FPR at frozen validation budget | Recall at frozen budget |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, item in results['models'].items():
        report = item['splits']['brown_English_source_test']
        paired=report['common_with_original_baseline']
        base,budget=paired['after_at_0_5'],paired['after_at_frozen_budget']
        lines.append('| '+name+' | '+f"{paired['common_supported_rows']}/{report['eligibility_selected_rows']}"+' | '+
            ' | '.join([number(base.get('accuracy'),True),number(base.get('roc_auc')),number(base.get('fpr'),True),
                        number(base.get('recall'),True),number(base.get('brier')),
                        number(budget.get('fpr'),True),number(budget.get('recall'),True)])+' |')
    lines += ['', 'All source rows, quality exclusions, unavailable predictions, frozen thresholds and class denominators remain in results.json and predictions.jsonl. '
              'An empirical 1% validation FPR budget is not a guaranteed 1% external FPR. There is no calibrated quarantine or prevention claim.', '',
              '## Scope limits',''] + ['- '+item for item in protocol['limitations']]
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def execute_prepared(output):
    out = Path(output).resolve()
    if not out.is_relative_to((ROOT / 'benchmarks').resolve()) or out == (ROOT / 'benchmarks').resolve():
        raise ValueError('Prepared research output must be a workspace benchmarks child')
    initial_status = json.loads((out / 'run_status.json').read_text(encoding='utf-8'))
    if initial_status.get('stage') != 'prepared' or (out / 'results.json').exists() or (out / 'predictions.jsonl').exists():
        raise ValueError('Only a pristine prepared experiment may execute; never overwrite or resume an old fit')
    protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    protocol_hash = digest(out / 'protocol.json')
    if protocol_hash != initial_status['protocol_sha256']:
        raise ValueError('Prepared protocol changed')
    assert_frozen(out,protocol)
    rows = read_jsonl(out / 'partition_rows.jsonl')
    brown = read_jsonl(protocol['Brown_records_path'])
    result = dict(schema_version=1, complete=False, approved_for_deployment=False,
        deployment_generalization_established=False, calibration_fitted=False,
        protocol_sha256=protocol_hash, pipeline_id=protocol['pipeline_id'], models={},
        interpretation='Research-only source-author and publisher-label evidence; no external result is hidden or adjusted')
    fitted = {}
    brown_predictions = {}
    footer_predictions = {}
    footer_inputs=neutral_footer_inputs(brown)
    try:
        for job in FIT_JOBS:
            pool = job['pool']
            train = [r for r in rows if r['pool']==pool and r['split']=='train']
            for arm in [next(arm for arm in ARMS if arm['name'] == job['arm'])]:
                name = pool+'__'+arm['name'] + ('__source_class_balanced' if job['balance_source_classes'] else '')
                status(out,'fitting',model=name,rows=len(train))
                model = email_evidence_estimator(character_features=arm['character_features'], evidence_features=arm['evidence_features'])
                fit_kwargs = {}
                if job['balance_source_classes']:
                    model.set_params(classifier__class_weight=None)
                    fit_kwargs['classifier__sample_weight'] = source_class_weights(train)
                before, started = memory(),time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit([r['email_input'] for r in train],[r['label'] for r in train], **fit_kwargs)
                convergence = [str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
                if convergence:
                    raise ValueError('Fixed candidate did not converge: '+'; '.join(convergence))
                fit_seconds = time.perf_counter()-started
                path = out/(name+'.pkl')
                joblib.dump(dict(schema_version=3, input_kind=EMAIL_EVIDENCE_MODEL_INPUT_KIND,
                    input_contract_version=EMAIL_INPUT_CONTRACT_VERSION, model=model,
                    approved_for_deployment=False, deployment_generalization_established=False,
                    probability_calibrated=False, protocol_sha256=protocol_hash,
                    contract_source_hashes=contract_source_hashes(), library_versions=protocol['library_versions'],
                    source_label_authority='research-only published expert plus optional source-author labels'),path)
                fitted[name]=dict(pool=pool,arm=arm['name'],model_sha256=digest(path),model_mib=path.stat().st_size/2**20,
                    fit_seconds=fit_seconds,training_rows=len(train),n_iter=model.named_steps['classifier'].n_iter_.tolist(),
                    source_class_balanced=job['balance_source_classes'],
                    effective_classifier_params={**MODEL_PARAMS, 'class_weight': None if job['balance_source_classes'] else MODEL_PARAMS['class_weight']},
                    training_source_class_counts={source: dict(Counter(str(r['label']) for r in train if r['source']==source))
                                                  for source in sorted({r['source'] for r in train})},
                    fit_warnings=[str(w.message) for w in caught],memory_before_fit=before,memory_after_fit=memory())
                del model
                gc.collect()
        # Freeze all validation thresholds before any Brown predictions.
        for name,item in fitted.items():
            validation=[r for r in rows if r['pool']==item['pool'] and r['split']=='val']
            model=joblib.load(out/(name+'.pkl'))['model']
            p=predict_email_evidence_probabilities(model,[r['email_input'] for r in validation])
            item['threshold']=validation_threshold([r['label'] for r in validation],p,FPR_BUDGET)
            item['validation_supported_rows']=int(np.isfinite(p).sum())
            save(out/(name+'_threshold.json'),dict(threshold=item['threshold'],selected_split='val',budget=FPR_BUDGET,
                protocol_sha256=protocol_hash,model_sha256=item['model_sha256']))
            del model
            gc.collect()
        save(out/'fitted_models.json',fitted)
        assert_frozen(out,protocol)
        status(out,'fits_and_thresholds_frozen',models=len(fitted),Brown_predictions_obtained=0)
        for name,item in fitted.items():
            if digest(out/(name+'.pkl')) != item['model_sha256']:
                raise ValueError('Candidate changed after fit freeze')
            model=joblib.load(out/(name+'.pkl'))['model']
            report=dict(**item,splits={})
            for split in ('val','published_day_test','brown_English_source_test'):
                part=brown if split=='brown_English_source_test' else [r for r in rows if r['pool']==item['pool'] and r['split']==split]
                status(out,'scoring',model=name,split=split,rows=len(part))
                p,timing=timed_scores(lambda x:predict_email_evidence_probabilities(model,x),[r['email_input'] for r in part])
                report['splits'][split]=report_metrics(part,p,item['threshold'],
                    eligible=[r['eligible_for_scoring'] for r in part] if split=='brown_English_source_test' else None)
                report['splits'][split]['timing']=timing
                append_predictions(out,name,split,part,p,item['threshold'])
                if split=='brown_English_source_test':brown_predictions[name]=p
            result['models'][name]=report
            status(out,'paired_neutral_footer_probe',model=name)
            footer_predictions[name]=predict_email_evidence_probabilities(model,footer_inputs)
            footer_rows=[dict(row,email_input=contract,input_contract_sha256=contract_sha256(contract))
                         for row,contract in zip(brown,footer_inputs)]
            append_predictions(out,name,'brown_neutral_footer_probe',footer_rows,footer_predictions[name],item['threshold'])
            save(out/'evaluation_checkpoint.json',result)
            del model
            gc.collect()
        # Existing English arms remain separately identified; no new fit or
        # threshold selection is performed on Brown or the old English corpus.
        texts=[r['email_input']['body']+' '+r['email_input']['subject'] for r in brown]
        for name,item in protocol['original_English_models'].items():
            status(out,'scoring_frozen_original_English_model',model=name,rows=len(brown))
            if name=='deployed_baseline':
                model=joblib.load(ROOT/'models/phish_model.pkl')
                vectorizer=joblib.load(ROOT/'models/vectorizer.pkl')
                for fold in getattr(model,'calibrated_classifiers_',[]):
                    fold.estimator.set_params(n_jobs=1)
                fn=lambda x:score_baseline(model,vectorizer,x)
            else:
                model=joblib.load(item['model_path'])['model']
                fn=lambda x:score_candidate(model,x)
            p,timing=timed_scores(fn,texts)
            report=report_metrics(brown,p,item['threshold'],eligible=[r['eligible_for_scoring'] for r in brown])
            report['timing']=timing
            result['models'][name]=dict(threshold=item['threshold'],fitted_in_this_run=False,
                retained_internal_development_test=item['retained_internal_development_test'],
                retained_internal_scope=item['retained_internal_scope'],
                splits={'brown_English_source_test':report})
            append_predictions(out,name,'brown_English_source_test',brown,p,item['threshold'])
            brown_predictions[name]=p
            footer_texts=[contract['body']+' '+contract['subject'] for contract in footer_inputs]
            footer_predictions[name]=fn(footer_texts)
            footer_rows=[dict(row,email_input=contract,input_contract_sha256=contract_sha256(contract))
                         for row,contract in zip(brown,footer_inputs)]
            append_predictions(out,name,'brown_neutral_footer_probe',footer_rows,footer_predictions[name],item['threshold'])
            save(out/'evaluation_checkpoint.json',result)
            del model,fn
            if name=='deployed_baseline':del vectorizer
            gc.collect()
        result['paired_Brown_comparisons']={}
        for name,item in result['models'].items():
            item['splits']['brown_English_source_test']['common_with_original_baseline']=paired_comparison(
                brown,brown_predictions['deployed_baseline'],brown_predictions[name],
                result['models']['deployed_baseline']['threshold'],item['threshold'])
            item['neutral_footer_probe']=footer_report(brown,brown_predictions[name],footer_predictions[name],item['threshold'],
                baseline=brown_predictions['deployed_baseline'])
        for arm in ARMS:
            before,after=[pool+'__'+arm['name'] for pool in POOLS]
            result['paired_Brown_comparisons'][arm['name']+'_English_data_addition']=paired_comparison(
                brown,brown_predictions[before],brown_predictions[after],
                result['models'][before]['threshold'],result['models'][after]['threshold'],
                baseline=brown_predictions['deployed_baseline'])
        for pool in POOLS:
            before,after=pool+'__text_only',pool+'__structured'
            result['paired_Brown_comparisons'][pool+'_structured_evidence_addition']=paired_comparison(
                brown,brown_predictions[before],brown_predictions[after],
                result['models'][before]['threshold'],result['models'][after]['threshold'],
                baseline=brown_predictions['deployed_baseline'])
        for pool in POOLS:
            name=pool+'__structured'
            result['paired_Brown_comparisons'][name+'_versus_original_baseline']=paired_comparison(
                brown,brown_predictions['deployed_baseline'],brown_predictions[name],
                result['models']['deployed_baseline']['threshold'],result['models'][name]['threshold'])
        for arm in ('text_only', 'structured'):
            before=POOLS[1]+'__'+arm
            after=before+'__source_class_balanced'
            result['paired_Brown_comparisons'][arm+'_source_class_weighting']=paired_comparison(
                brown,brown_predictions[before],brown_predictions[after],
                result['models'][before]['threshold'],result['models'][after]['threshold'],
                baseline=brown_predictions['deployed_baseline'])
        assert_frozen(out,protocol)
        result.update(complete=True,protected_artifacts_unchanged=True,frozen_inputs_unchanged=True,
                      no_candidate_promoted=True,completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(out/'results.json',result)
        write_report(out,result,protocol)
        status(out,'completed',models=len(result['models']))
        return result
    except Exception as error:
        save(out/'failure.json',dict(complete=False,error=repr(error),protocol_sha256=protocol_hash,
                                   completed_models=list(result['models']),no_candidate_promoted=True))
        status(out,'failed',error=repr(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--execute-prepared',type=Path)
    parser.add_argument('--review',type=Path,default=ROOT/'data/email_evidence_v1_20261004_v2/review_corpus.jsonl')
    parser.add_argument('--spaphish',type=Path,default=ROOT/'data/email_evidence_sources/spaphish_v4')
    parser.add_argument('--brown',type=Path,default=ROOT/'data/brown_email_benchmark_20261005')
    parser.add_argument('--phishfuzzer',type=Path,default=ROOT/'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json')
    parser.add_argument('--old-comparison',type=Path,default=ROOT/'benchmarks/email_representation_comparison_20261004_v2')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.execute_prepared:
        if args.prepare_only or args.output:
            parser.error('--execute-prepared cannot be combined with preparation arguments')
        execute_prepared(args.execute_prepared)
    elif args.prepare_only and args.output:
        prepare_experiment(args.review,args.spaphish,args.brown,args.phishfuzzer,args.old_comparison,args.output)
    else:
        parser.error('Use --prepare-only --output NEW_DIRECTORY, review its protocol, then --execute-prepared DIRECTORY')
