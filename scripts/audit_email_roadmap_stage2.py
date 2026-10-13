"""Read-only Email data audit for the approved roadmap; never fits a model.

Outputs are review metadata, not new labels or an admitted training corpus.
Source claims, project review and task-valid deployment coverage stay distinct.
"""
import argparse
import csv
import hashlib
import json
import mailbox
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.email_evidence import group_email_templates
from utils.email_candidate_text import canonical_email_candidate_text
from utils.email_input_contract import contract_sha256, evidence_sha256
from utils.jsonl import read_jsonl

OUT = ROOT / 'benchmarks/email_roadmap_v1_20261006/stage2_data_audit_20261007'
CORPUS = ROOT / 'data/email_evidence_v1_20261004_v2/review_corpus.jsonl'
MANIFEST = CORPUS.parent / 'manifest.json'
SEEDS = ROOT / 'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json'
BROWN = ROOT / 'data/brown_email_benchmark_20261005/all_records.jsonl'
LANGUAGES = ('English', 'Urdu', 'Arabic', 'Spanish', 'French', 'German',
    'Portuguese', 'Italian', 'Dutch', 'Russian', 'Ukrainian', 'Turkish', 'Persian',
    'Hindi', 'Bengali', 'Punjabi', 'Tamil', 'Telugu', 'Marathi', 'Chinese',
    'Japanese', 'Korean', 'Indonesian', 'Malay', 'Vietnamese', 'Thai')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')


def review_gaps(row):
    """Local review requirements only; not partition or deployment approval."""
    gaps = []
    if row.get('task_label') not in ('phishing', 'bec', 'legitimate'):
        gaps.append('task_label_unresolved_or_out_of_scope')
    for key in ('label_review_completed', 'independent_review_completed',
                'extraction_verified', 'training_reuse_permitted'):
        if row.get(key) is not True:
            gaps.append(key)
    for key in ('reviewer_id', 'second_reviewer_id', 'label_evidence_reference',
                'label_rationale', 'campaign_cluster_id', 'source'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            gaps.append(key)
    if row.get('reviewer_id') and row.get('reviewer_id') == row.get('second_reviewer_id'):
        gaps.append('reviewers_not_independent')
    return gaps


def duplicate_groups(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if row.get(key):
            groups[row[key]].append(row['id'])
    return [ids for ids in groups.values() if len(ids) > 1]


def grouping_view(rows):
    # Match the prepared corpus's documented input view. The outer source body
    # may be an older extraction, not the contract consumed by the candidate.
    return [dict(id=r['id'], body=r['email_input']['body'] or r['email_input']['html_body'],
                 subject=r['email_input']['subject']) for r in rows]


def main():
    if OUT.exists():
        raise FileExistsError('Refusing to overwrite a recorded audit: ' + str(OUT))
    OUT.mkdir(parents=True)
    try:
        audit()
    except Exception as error:
        write_json(OUT / 'run_status.json', dict(complete=False, stage='failed',
            error_type=type(error).__name__, error=str(error)))
        raise


def audit():
    original_manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    rows = read_jsonl(CORPUS)
    seeds = json.loads(SEEDS.read_text(encoding='utf-8'))
    brown = read_jsonl(BROWN)
    protected = json.loads((OUT.parent / 'protected_snapshot.json').read_text(encoding='utf-8'))
    baseline_changes = [p for p, sha in protected.items() if digest(p) != sha]
    tracked = {str(p): digest(p) for p in (CORPUS, MANIFEST, SEEDS, BROWN,
        ROOT / 'data/clean_email_corpus.csv', Path(__file__),
        ROOT / 'training/email_evidence.py', ROOT / 'utils/email_input_contract.py',
        ROOT / 'utils/email_candidate_text.py', ROOT / 'utils/text_sanitizer.py')}
    sources, queue, errors, boxes, represented_seeds = {}, [], [], {}, set()
    print('Verifying original file and per-message bindings', flush=True)
    try:
        for row in rows:
            source = row['source']
            ref = row['original_reference']
            path = (ROOT / ref['path']).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Evidence reference outside workspace: ' + row['id'])
            if str(path) not in tracked:
                tracked[str(path)] = digest(path)
            expected = ref.get('archive_sha256') or ref.get('source_sha256') or ref.get('snapshot_sha256')
            binding_errors = []
            if not expected or tracked[str(path)] != expected:
                binding_errors.append('original_file_hash_mismatch_or_absent')
            if source == 'nazario':
                if path not in boxes:
                    boxes[path] = mailbox.mbox(path, create=False)
                raw = boxes[path].get_bytes(ref['message_index'])
                if hashlib.sha256(raw).hexdigest() != ref['raw_message_sha256']:
                    binding_errors.append('original_message_hash_mismatch')
            if source == 'phishfuzzer_manual':
                index = ref['row_index']
                if not isinstance(index, int) or not 0 <= index < len(seeds):
                    binding_errors.append('invalid_original_seed_index')
                else:
                    represented_seeds.add(index)
            try:
                if contract_sha256(row['email_input']) != row['input_contract_sha256']:
                    binding_errors.append('input_contract_hash_mismatch')
                if evidence_sha256(row['email_input']) != row['evidence_sha256']:
                    binding_errors.append('evidence_hash_mismatch')
            except ValueError as error:
                binding_errors.append('contract_validation_failed:' + str(error))
            if binding_errors:
                errors.append(dict(id=row['id'], errors=binding_errors))
            gaps = review_gaps(row)
            item = sources.setdefault(source, dict(rows=0, source_labels=Counter(),
                input_forms=Counter(), language_claims=Counter(), fields=Counter(),
                quality_flags=Counter(), gaps=Counter(), review_complete_rows=0,
                reuse_permitted_rows=0, trusted_authentication_rows=0,
                time_authorities=Counter(), original_bindings_verified=0))
            item['rows'] += 1
            item['source_labels'][str(row.get('source_claimed_label'))] += 1
            item['input_forms'][str(row.get('input_form'))] += 1
            item['language_claims'][str(row.get('language_scope'))] += 1
            item['time_authorities'][str(row.get('time_authority'))] += 1
            item['fields'].update(k for k, v in row['email_input']['availability'].items() if v is True)
            item['quality_flags'].update(set(row.get('quality_flags', [])))
            item['gaps'].update(gaps)
            item['review_complete_rows'] += not gaps
            item['reuse_permitted_rows'] += row.get('training_reuse_permitted') is True
            item['trusted_authentication_rows'] += row['email_input'].get('trusted_authentication') is not None
            item['original_bindings_verified'] += not binding_errors
            queue.append(dict(id=row['id'], source=source,
                input_contract_sha256=row['input_contract_sha256'],
                original_reference=json.dumps(ref, ensure_ascii=False),
                source_claimed_label=row.get('source_claimed_label'),
                current_task_label=row.get('task_label'),
                language_claim=row.get('language_scope'), review_gaps=';'.join(gaps),
                binding_errors=';'.join(binding_errors),
                quality_flags=';'.join(row.get('quality_flags', []))))
    finally:
        for box in boxes.values():
            box.close()

    print('Recomputing complete-body template groups; no samples are removed', flush=True)
    groups, pairs = group_email_templates(grouping_view(rows), canonicalizer=canonical_email_candidate_text)
    old_groups = {r['id']: r.get('campaign_cluster_id') for r in rows}
    group_mismatches = [r['id'] for r, group in zip(rows, groups) if old_groups[r['id']] != group]
    saved_pairs = read_jsonl(CORPUS.parent / 'full_template_pairs.jsonl')
    pair_key = lambda p: (p['first'], p['second'], p['full_trigram_jaccard'])
    pairs_match = sorted(map(pair_key, pairs)) == sorted(map(pair_key, saved_pairs))
    manual = [(i, r) for i, r in enumerate(seeds) if r.get('Source') == 'Manual']
    inventory_path = ROOT / 'benchmarks/email_comparison_source_review/candidate_inventory.jsonl'
    tracked[str(inventory_path)] = digest(inventory_path)
    inventory = {r['source_row_index']: r for r in read_jsonl(inventory_path)}
    missing_manual = [dict(source_row_index=i, original_id=r.get('Original_ID'),
        source_claimed_label=r.get('Type'), language_claim=r.get('Language'),
        evidence_path=str(SEEDS.relative_to(ROOT)), source_sha256=tracked[str(SEEDS)],
        prior_exclusion_reasons=inventory[i]['exclusion_reasons'],
        existing_corpus_overlap_rows=inventory[i].get('existing_corpus_overlap_rows'),
        status='not_in_prepared_review_corpus; original remains in source and inventory')
        for i, r in manual if i not in represented_seeds]
    with (ROOT / 'data/clean_email_corpus.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        legacy_fields = reader.fieldnames
        legacy = list(reader)
    legacy_summary = dict(rows=len(legacy), columns=legacy_fields,
        labels=dict(Counter(r['label'] for r in legacy)),
        splits=dict(Counter(r['split'] for r in legacy)),
        source_labels=dict(Counter(r['source'] + ':' + r['label'] for r in legacy)),
        raw_text_duplicate_groups=duplicate_groups([
            dict(id=str(i), text_hash=hashlib.sha256(r['raw_text'].encode()).hexdigest())
            for i, r in enumerate(legacy)], 'text_hash'),
        missing_evidence_columns=[k for k in ('sender', 'headers', 'urls', 'observed_at_utc',
            'label_evidence_reference', 'language') if k not in legacy_fields])
    latest_brown = dict(rows=len(brown), publisher_label_counts=dict(Counter(r['task_label'] for r in brown)),
        text_eligible_rows=sum(r.get('eligible_for_text_metrics') is True for r in brown),
        project_review_complete_rows=sum(r.get('project_independent_review_completed') is True for r in brown),
        reuse_permission_rows=sum(r.get('training_reuse_permission_established') is True for r in brown),
        extraction_difference_rows=sum(r['input_contract_sha256'] != next(
            old['input_contract_sha256'] for old in rows if old['id'] == r['id']) for r in brown),
        status='Same original 90 messages as prepared corpus; newer extraction, never count as additional independent data')
    unchanged = all(digest(p) == sha for p, sha in tracked.items())
    protected_unchanged = all(digest(p) == sha for p, sha in protected.items())
    manifest_matches = digest(CORPUS) == original_manifest['review_corpus_sha256']
    integrity = not errors and not group_mismatches and pairs_match and unchanged and manifest_matches
    result = dict(schema_version=1, complete=True, audit_completed_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='2_task_input_data_audit', stage_gate_passed=False,
        new_training=False, calibration_fitted=False, production_promoted=False,
        sources=sources, original_evidence_rows=len(rows),
        review_manifest_hash_matches=manifest_matches,
        training_file_bytes=(CORPUS.parent / 'training_corpus.jsonl').stat().st_size,
        reviewed_records_meeting_local_requirements=sum(not review_gaps(r) for r in rows),
        integrity_verified=integrity, binding_errors=errors,
        exact_complete_evidence_duplicate_groups=duplicate_groups(rows, 'evidence_sha256'),
        complete_template_pairs=len(pairs), template_groups=len(set(groups)),
        template_groups_match_saved=not group_mismatches, group_mismatch_ids=group_mismatches,
        template_pairs_match_saved=pairs_match, all_inputs_unchanged=unchanged,
        protected_unchanged=protected_unchanged, protected_changes_since_stage1=baseline_changes,
        input_sha256=tracked, legacy_corpus=legacy_summary, latest_brown=latest_brown,
        full_seed_language_claims=dict(Counter(r.get('Language') for r in seeds)),
        private_seed_language_claims=dict(Counter(r.get('Language') for _, r in manual)),
        private_seed_records=len(manual), private_seed_unrepresented_rows=missing_manual,
        language_acceptance=[dict(language=lang, status='not_validated',
            independent_native_acceptance_counts=None,
            target_legitimate=300, target_phishing=100) for lang in LANGUAGES],
        notes=['No automated audit can supply missing independent human review or verified native-language labels.',
            'Published class labels and language claims are not silently discarded or equated to project review.',
            'No corpus rows, difficult examples or uncertain labels were deleted, relabeled or admitted.',
            'Trigram grouping is a specific complete-text similarity check, not proof against paraphrase/translation leakage.',
            'Old review preparation and newer publisher benchmarks are inspected development artifacts, not blind acceptance.'])
    write_json(OUT / 'results.json', result)
    with (OUT / 'review_gaps.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    write_json(OUT / 'run_status.json', dict(complete=True, stage='audit_complete',
        data_admission_passed=False, integrity_verified=integrity,
        protected_unchanged=protected_unchanged))
    print(json.dumps(dict(audit_complete=True, integrity_verified=integrity,
        reviewed_records=result['reviewed_records_meeting_local_requirements'],
        template_pairs=len(pairs), protected_unchanged=protected_unchanged), allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    selected = args.out.resolve()
    if not selected.is_relative_to(ROOT / 'benchmarks/email_roadmap_v1_20261006'):
        raise ValueError('Audit output must stay under the roadmap benchmarks folder')
    OUT = selected
    main()
