"""Snapshot publisher metadata and prepare pending source review, never admission.

Only writes a fresh roadmap output directory. No email submissions, account
registration, training, relabeling, calibration or production changes.
"""
import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.jsonl import read_jsonl

ROADMAP = ROOT / 'benchmarks/email_roadmap_v1_20261006'
SEEDS = ROOT / 'data/external_email_validation/phishfuzzer_original_seeds'
CORPUS = ROOT / 'data/email_evidence_v1_20261004_v2/review_corpus.jsonl'
METADATA = {
    'phishfuzzer_repository.json': 'https://api.github.com/repos/DataPhish/PhishFuzzer',
    'phishfuzzer_root.json': 'https://api.github.com/repos/DataPhish/PhishFuzzer/contents/',
    'sting9_dataset.html': 'https://sting9.org/dataset',
    'sting9_license.html': 'https://sting9.org/license',
    'sting9_research.html': 'https://sting9.org/research',
    'iwspa_task.html': 'https://dasavisha.github.io/IWSPA-sharedtask/',
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + '\n', encoding='utf-8')


def fetch_metadata(item, directory):
    name, url = item
    result = dict(name=name, source_url=url, fetched_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'OriginTrace-source-audit/1.0',
            'Accept': 'application/vnd.github+json' if name.endswith('.json') else 'text/html'})
        with urllib.request.urlopen(request, timeout=20) as response:
            content = response.read(2 * 1024 * 1024 + 1)
            if len(content) > 2 * 1024 * 1024:
                raise ValueError('Publisher metadata exceeds the 2 MiB bound')
            result.update(http_status=response.status, resolved_url=response.url,
                          content_type=response.headers.get('Content-Type'))
        path = directory / name
        path.write_bytes(content)
        result.update(complete=True, bytes=len(content), sha256=digest(path))
    except Exception as error:
        result.update(complete=False, error_type=type(error).__name__, error=str(error))
        if isinstance(error, urllib.error.HTTPError):
            result['http_status'] = error.code
    return result


def pending_manual_review(seeds, prepared, historical_inventory, source_sha):
    """Retain all manual rows, including eight not in the previous 292 cohort."""
    prepared_by_index = {}
    for row in prepared:
        if row['source'] == 'phishfuzzer_manual':
            index = row['original_reference']['row_index']
            if index in prepared_by_index:
                raise ValueError('Duplicate prepared source index')
            prepared_by_index[index] = row
    prior_by_index = {row['source_row_index']: row for row in historical_inventory}
    pending = []
    for index, row in enumerate(seeds):
        if row.get('Source') != 'Manual':
            continue
        evidence = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
        original_sha = hashlib.sha256(evidence).hexdigest()
        existing = prepared_by_index.get(index)
        pending.append(dict(
            id='phishfuzzer-source-review-' + str(index), source_row_index=index,
            source_original_id=row.get('Original_ID'), source_file_sha256=source_sha,
            source_record_sha256=original_sha,
            original_reference='data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json',
            source_claimed_label=row.get('Type'), source_claimed_language=row.get('Language'),
            source_claimed_authorship=row.get('Created by'), source_claimed_year=row.get('Year'),
            language_independently_verified=False, receiver_time_verified=False,
            native_real_vs_simulated_provenance=None, reviewed_task_label=None,
            reviewer_ids=[], project_independent_review_completed=False,
            training_reuse_permission_established=False, research_supervision_eligible=False,
            prepared_input_id=existing['id'] if existing else None,
            input_contract_sha256=existing['input_contract_sha256'] if existing else None,
            historical_selection_reasons=prior_by_index[index]['exclusion_reasons'],
            historical_selection_is_admission=False,
            role=None, assignment_status='pending_source_permission_task_and_extraction_review',
            existing_inspected_cohort=True, eligible_for_new_blind_acceptance=False,
            approved_for_training=False, approved_for_deployment=False))
    if set(prepared_by_index) - {r['source_row_index'] for r in pending}:
        raise ValueError('Prepared manual row absent from original manual inventory')
    return pending


def main(out):
    out = out.resolve()
    if not out.is_relative_to(ROADMAP.resolve()):
        raise ValueError('Output must stay inside the approved roadmap directory')
    if out.exists():
        raise FileExistsError('Refusing to overwrite recorded source evidence')
    out.mkdir(parents=True)
    protected = json.loads((ROADMAP / 'protected_snapshot.json').read_text(encoding='utf-8'))
    try:
        audit = json.loads((ROADMAP / 'stage2_data_audit_20261007_v2/results.json').read_text(encoding='utf-8'))
        if audit.get('complete') is not True:
            raise ValueError('Prior integrity audit is incomplete')
        bindings = {**protected, **audit['input_sha256']}
        for path, sha in bindings.items():
            if digest(path) != sha:
                raise ValueError('Recorded source/code/protected hash changed: ' + path)
        acquisition = json.loads((SEEDS / 'manifest.json').read_text(encoding='utf-8'))
        for item in acquisition['files']:
            if digest(ROOT / item['path']) != item['sha256']:
                raise ValueError('Original seed acquisition hash changed')
        seeds = json.loads((SEEDS / 'original_seeds.json').read_text(encoding='utf-8'))
        historical_path = ROOT / 'benchmarks/email_comparison_source_review/candidate_inventory.jsonl'
        pending = pending_manual_review(seeds, read_jsonl(CORPUS), read_jsonl(historical_path),
                                        digest(SEEDS / 'original_seeds.json'))
        with (out / 'pending_manual_source_review.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for row in pending:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print('All original manual rows retained in pending review inventory', flush=True)
        directory = out / 'publisher_metadata'
        directory.mkdir()
        with ThreadPoolExecutor(max_workers=3) as pool:
            metadata = list(pool.map(lambda item: fetch_metadata(item, directory), METADATA.items()))
        save(out / 'metadata_acquisition.json', metadata)
        for path, sha in bindings.items():
            if digest(path) != sha:
                raise ValueError('Protected/source input changed during intake: ' + path)
        baseline_path = ROADMAP / 'stage1_baseline_verification.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        results = dict(schema_version=1, preparation_complete=True,
            metadata_capture_complete=all(r['complete'] for r in metadata),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            source_seed_rows=len(seeds), manual_rows=len(pending),
            source_claimed_language_counts=dict(Counter(r['source_claimed_language'] for r in pending)),
            source_claimed_label_counts=dict(Counter(r['source_claimed_label'] for r in pending)),
            rows_with_previous_prepared_input=sum(r['prepared_input_id'] is not None for r in pending),
            previously_unprepared_rows=[r['source_row_index'] for r in pending if r['prepared_input_id'] is None],
            language_verified_rows=0, independently_reviewed_rows=0,
            new_source_admitted_rows=0, stage_gate_passed=False, fitting_ready=False,
            preserved_external_baseline=baseline['legacy_measurement_reproduction']['external'],
            protected_files_checked=len(protected), protected_unchanged=True,
            new_training=False, calibration_fitted=False, production_promoted=False,
            frontend_changes=False,
            next_required_actions=['Publisher reuse permission and per-row real/simulation provenance clarification',
                'Task/extraction and native-language review bound to original and input hashes',
                'Separately acquired uninspected source/time holdout with hard legitimate emails',
                'Source authority approval and five-role partition gates before fitting'],
            input_bindings={str(path.relative_to(ROOT)): digest(path) for path in
                (SEEDS / 'manifest.json', SEEDS / 'original_seeds.json', CORPUS,
                 historical_path, baseline_path, Path(__file__))},
            pending_inventory_sha256=digest(out / 'pending_manual_source_review.jsonl'),
            limits=['Source labels, authorship, languages and dates remain publisher claims, not project attestations',
                'All 300 manual rows remain present; older selection exclusions are review flags, not deletion',
                'Capturing public metadata does not grant dataset rights or approve a new source',
                'This output does not change predictions or establish new accuracy'])
        save(out / 'results.json', results)
        save(out / 'run_status.json', dict(complete=True, stage='source_intake_prepared_not_admitted',
            metadata_capture_complete=results['metadata_capture_complete'], stage_gate_passed=False))
        print(json.dumps({k: results[k] for k in ('manual_rows', 'source_claimed_language_counts',
            'rows_with_previous_prepared_input', 'new_source_admitted_rows', 'metadata_capture_complete',
            'protected_unchanged')}), flush=True)
        return 0 if results['metadata_capture_complete'] else 1
    except Exception as error:
        save(out / 'run_status.json', dict(complete=False, stage='failed',
            error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROADMAP / 'source_intake_v1_20261007')
    sys.exit(main(parser.parse_args().out))
