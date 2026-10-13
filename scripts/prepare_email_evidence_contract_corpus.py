"""Prepare a separate, auditable Email review pool. Never invent approved labels.

Published labels, annotator rationales, source names and timestamps stay outside
the inference contract. Ordinary spam is outside the phishing/BEC binary target.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.email_evidence import group_email_templates
from utils.email_candidate_text import canonical_email_candidate_text
from utils.jsonl import read_jsonl
from utils.email_input_contract import (
    EMAIL_INPUT_CONTRACT_VERSION, build_email_input, contract_sha256, evidence_sha256,
)

TASK_LABELS = {'phishing', 'bec', 'legitimate', 'spam', 'unknown', 'non_email'}
REVIEW_COLUMNS = ['id', 'input_contract_sha256', 'source', 'source_claimed_label',
    'evidence_reference', 'task_label', 'reviewer_id', 'second_reviewer_id',
    'label_rationale', 'label_evidence_reference', 'label_review_completed',
    'independent_review_completed', 'extraction_verified', 'split',
    'observed_at_utc', 'time_authority']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path, records):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                                for r in records), encoding='utf-8')


def reference_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def declared_url_list(value):
    """Parse the publisher's [URL],[URL] representation, retaining invalid items.

    This is not JSON/Python code. Never use eval or silently pick only HTTP links.
    IPv6's inner brackets are preserved because only ],[ is a delimiter.
    """
    if value == '':
        return [], []
    if not value.startswith('['):
        return [], ['unparsed_source_url_metadata']
    parts = value[1:].split('],[')
    if not value.endswith(']'):
        # Retain complete preceding values, but never fabricate the truncated
        # last destination. The exact original field remains in source_record.
        return parts[:-1], ['incomplete_source_url_metadata']
    parts[-1] = parts[-1][:-1]
    return parts, []


def declared_count(value):
    """Publisher CSV numeric counts may be serialized as 32.00, not only 32."""
    try:
        number = Decimal(value)
        return int(number) if number.is_finite() and number >= 0 and number == number.to_integral_value() else None
    except (InvalidOperation, ValueError):
        return None


def folder_metadata(row):
    # X-IMAP alone is not grounds to reject a real message. This exact container
    # signature is metadata written by the mailbox implementation, not an email.
    return (bool(re.search(r'(?im)^X-IMAP:', row.get('headers', ''))) and
            row.get('subject', '').strip().casefold() ==
            "don't delete this message -- folder internal data" and
            'not a real message' in ' '.join(row.get('body', '').lower().split()))


def migrate_existing(path):
    records = []
    for row in read_jsonl(path):
        item = dict(row)
        item['source_record'] = row  # Original extraction and provenance retained.
        item['language_scope'] = 'not_independently_verified'
        item['training_reuse_permitted'] = row['source'] == 'nazario'
        item['training_reuse_authority'] = row.get('reuse_terms_reference')
        item['source_annotation_authority'] = 'source_claim; task/extraction review pending'
        records.append(item)
    return records


def import_spaphish(directory):
    directory = Path(directory)
    acquisition = json.loads((directory / 'acquisition.json').read_text(encoding='utf-8'))
    expected = {r['path']: r['sha256'] for r in acquisition['files']}
    for filename in ('corpus.csv', 'schema.json', 'README.txt'):
        if digest(directory / filename) != expected.get(filename):
            raise ValueError('Acquired SpaPhish file hash mismatch: ' + filename)
    if acquisition.get('version') != 4 or acquisition.get('declared_license') != 'CC BY 4.0':
        raise ValueError('Unsupported SpaPhish acquisition provenance')
    rows = []
    with (directory / 'corpus.csv').open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter=';')
        if not {'hash', 'subject', 'body', 'date', 'urls', 'url_count', 'Label'}.issubset(reader.fieldnames or []):
            raise ValueError('SpaPhish CSV lacks its declared schema')
        for index, original in enumerate(reader):
            if None in original or any(value is None for value in original.values()):
                raise ValueError('Malformed SpaPhish CSV row: ' + str(index))
            if original['Label'] not in ('0', '1'):
                raise ValueError('Unknown SpaPhish task label: ' + str(index))
            urls, flags = declared_url_list(original['urls'])
            if declared_count(original['url_count']) != len(urls):
                flags.append('source_url_count_mismatch')
            flags.extend(['original_mime_sender_and_receiver_authentication_unavailable',
                          'source_reported_date_has_no_receiver_proof_or_timezone',
                          'spanish_source_not_english_generalization_evidence'])
            rows.append({
                'id': 'spaphish-v4-' + original['hash'], 'source': 'spaphish_v4',
                'input_form': 'publisher_released_subject_body_and_url_metadata',
                'original_reference': {'path': reference_path(directory / 'corpus.csv'),
                    'source_sha256': expected['corpus.csv'], 'row_index': index,
                    'source_message_identifier': original['hash'],
                    'publisher': acquisition['source_url']},
                'source_claimed_label': 'publisher_phishing' if original['Label'] == '1' else 'publisher_legitimate',
                'source_annotation_authority': 'Publisher expert annotation; individual phishing reviewer identities unavailable in released CSV',
                'source_record': original,
                'sender': '', 'subject': original['subject'], 'body': original['body'],
                'html_body': '', 'headers': '', 'source_url_metadata': urls,
                'parsing_info': ({'source_kind': 'publisher_csv',
                    'errors': ['incomplete_source_url_metadata']} if
                    'incomplete_source_url_metadata' in flags or 'unparsed_source_url_metadata' in flags else None),
                'quality_flags': flags,
                'language_scope': 'Spanish', 'observed_at_utc': None,
                'time_authority': 'unverified', 'source_reported_reception_date': original['date'],
                'training_reuse_permitted': True,
                'training_reuse_authority': acquisition['source_url'] + ' (CC BY 4.0)',
                'evaluation_role': 'new_spanish_source_review_supplement',
            })
    return rows


def attach_contract(row):
    item = dict(row)
    contract = build_email_input(row.get('body', ''), subject=row.get('subject', ''),
        sender=row.get('sender', ''), headers=row.get('headers', ''),
        html_body=row.get('html_body', ''), supplementary_urls=row.get('source_url_metadata'),
        parsing_info=row.get('parsing_info'))
    item['email_input'] = contract
    item['input_contract_sha256'] = contract_sha256(contract)
    item['evidence_sha256'] = evidence_sha256(contract)
    item.update(task_label='unknown', label_review_completed=False,
                independent_review_completed=False, extraction_verified=False,
                reviewer_id=None, second_reviewer_id=None, label_rationale=None,
                label_evidence_reference=None, split=None, approved_for_training=False)
    flags = list(row.get('quality_flags', [])) + list(contract['parsing']['flags'])
    if contract['unsupported_urls']:
        flags.append('unsupported_or_malformed_source_destinations_preserved')
    if not contract['body'].strip() and not contract['html_body'].strip():
        flags.append('no_visible_body_evidence')
    if len(contract['body']) > 100_000:
        flags.append('long_body_requires_review_not_automatic_deletion')
    item['disposition'] = 'awaiting_review'
    if folder_metadata(item):
        item.update(task_label='non_email', disposition='quarantined_non_email')
        flags.append('imap_folder_internal_metadata_not_real_email')
    item['quality_flags'] = sorted(set(flags))
    return item


def load_annotations(path, rows):
    if path is None:
        return {}
    known = {row['id']: row for row in rows}
    annotations, seen_ids = {}, set()
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            identity = row.get('id')
            if identity not in known or identity in seen_ids:
                raise ValueError('Unknown or duplicate annotation ID: ' + str(identity))
            seen_ids.add(identity)
            # Blank forms are not completed annotations.
            if not row.get('task_label', '').strip():
                continue
            if row['input_contract_sha256'] != known[identity]['input_contract_sha256']:
                raise ValueError('Annotation is bound to different Email evidence: ' + identity)
            if row['task_label'] not in TASK_LABELS:
                raise ValueError('Unknown reviewed task label: ' + identity)
            annotations[identity] = row
    return annotations


def apply_annotation(item, annotation):
    if not annotation:
        return
    if item['disposition'] == 'quarantined_non_email' and annotation['task_label'] != 'non_email':
        raise ValueError('IMAP folder metadata cannot be admitted as a real email')
    for key in ('task_label', 'reviewer_id', 'second_reviewer_id', 'label_rationale',
                'label_evidence_reference', 'split', 'observed_at_utc', 'time_authority'):
        if annotation.get(key):
            item[key] = annotation[key].strip()
    for key in ('label_review_completed', 'independent_review_completed', 'extraction_verified'):
        value = annotation.get(key, '').lower().strip()
        if value not in ('', 'true', 'false'):
            raise ValueError('Review completion must be explicit true/false: ' + key)
        item[key] = value == 'true'


def admission_blockers(item):
    issues = []
    if item['disposition'].startswith(('quarantined', 'duplicate')):
        issues.append(item['disposition'])
    if item['task_label'] not in ('phishing', 'bec', 'legitimate'):
        issues.append('no_in_scope_reviewed_task_label')
    if not all(item[k] is True for k in ('label_review_completed', 'independent_review_completed', 'extraction_verified')):
        issues.append('task_and_extraction_review_incomplete')
    reviewers = (item.get('reviewer_id'), item.get('second_reviewer_id'))
    if not all(isinstance(r, str) and r.strip() for r in reviewers) or reviewers[0] == reviewers[1]:
        issues.append('independent_identifiable_reviewers_missing')
    if not item.get('label_rationale') or not item.get('label_evidence_reference'):
        issues.append('label_evidence_missing')
    if item.get('training_reuse_permitted') is not True:
        issues.append('training_reuse_permission_not_established')
    if item.get('split') not in ('train', 'val', 'source_test', 'time_test'):
        issues.append('source_time_group_partition_not_assigned')
    if item.get('time_authority') not in ('trusted_receiver', 'publisher_observation') or not item.get('observed_at_utc'):
        issues.append('trusted_observation_time_missing')
    if item['email_input']['parsing']['completeness'] == 'INCOMPLETE':
        issues.append('message_extraction_incomplete')
    return issues


def prepare(existing, spaphish, output, annotation_path=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; review evidence must not be overwritten')
    originals = migrate_existing(existing) + import_spaphish(spaphish)
    print('Building shared contracts for ' + str(len(originals)) + ' preserved records', flush=True)
    rows = []
    for index, original in enumerate(originals):
        rows.append(attach_contract(original))
        if (index + 1) % 500 == 0:
            print('Contract records completed: ' + str(index + 1), flush=True)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Input corpus contains duplicate record IDs')
    annotations = load_annotations(annotation_path, rows)
    for row in rows:
        apply_annotation(row, annotations.get(row['id']))
    duplicates, seen, conflicts = [], {}, []
    # Exact supplied evidence (including headers and destinations) is collapsed
    # only in the training export; all source references remain in the review pool.
    for row in rows:
        signature = row['evidence_sha256']
        if signature in seen:
            first = seen[signature]
            duplicates.append({'id': row['id'], 'duplicate_of': first['id'],
                               'criterion': 'identical shared input contract', 'sha256': signature})
            row['duplicate_of'] = first['id']
            row['disposition'] = 'duplicate_exact_evidence'
            if (row['task_label'] not in ('unknown', 'non_email') and
                    first['task_label'] not in ('unknown', 'non_email') and
                    (row['task_label'] in ('phishing', 'bec')) != (first['task_label'] in ('phishing', 'bec'))):
                conflicts.append({'first': first['id'], 'second': row['id'], 'criterion': 'identical input with contradictory reviewed task labels'})
                row['disposition'] = first['disposition'] = 'quarantined_conflicting_labels'
        else:
            seen[signature] = row
    grouping_records = [{'id': r['id'], 'body': r['email_input']['body'] or r['email_input']['html_body'],
                         'subject': r['email_input']['subject']} for r in rows]
    print('Checking full-message template overlap', flush=True)
    groups, pairs = group_email_templates(grouping_records, canonicalizer=canonical_email_candidate_text)
    for row, group in zip(rows, groups):
        row['campaign_cluster_id'] = group
        row['admission_blockers'] = admission_blockers(row)
        row['approved_for_training'] = not row['admission_blockers']
        if row['approved_for_training']:
            row['disposition'] = 'eligible_pending_global_partition_validation'
    ready = [r for r in rows if r['approved_for_training']]
    global_valid = False
    global_problem = 'No reviewed, eligible Email records. Training intentionally refused.'
    if ready:
        from training.email_evidence_model import validate_evidence_records
        try:
            validate_evidence_records(ready)
            global_valid, global_problem = True, None
        except ValueError as error:
            global_problem = str(error)
    output.mkdir(parents=True, exist_ok=True)
    jsonl(output / 'review_corpus.jsonl', rows)
    jsonl(output / 'training_corpus.jsonl', ready)
    jsonl(output / 'duplicate_audit.jsonl', duplicates)
    jsonl(output / 'label_conflict_audit.jsonl', conflicts)
    jsonl(output / 'full_template_pairs.jsonl', pairs)
    jsonl(output / 'quarantine.jsonl', [r for r in rows if r['disposition'].startswith('quarantined')])
    with (output / 'annotation_review.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            annotation = {k: row.get(k) for k in REVIEW_COLUMNS}
            annotation['evidence_reference'] = json.dumps(row['original_reference'], ensure_ascii=False)
            # Do not prefill an unknown source claim as a reviewed task label.
            annotation['task_label'] = '' if row['task_label'] == 'unknown' else row['task_label']
            writer.writerow(annotation)
    manifest = {
        'schema_version': 3, 'input_kind': 'structured_email_evidence_v1',
        'input_contract_version': EMAIL_INPUT_CONTRACT_VERSION,
        'label_scope': 'phishing_bec_vs_nonmalicious',
        'corpus_sha256': digest(output / 'training_corpus.jsonl'),
        'review_corpus_sha256': digest(output / 'review_corpus.jsonl'),
        'sources': dict(Counter(r['source'] for r in rows)), 'records': len(rows),
        'eligible_records': len(ready), 'reviewed_task_labels': dict(Counter(r['task_label'] for r in rows)),
        'source_claimed_labels': dict(Counter(r['source_claimed_label'] for r in rows)),
        'input_availability': dict(Counter(k for r in rows for k,v in r['email_input']['availability'].items() if v)),
        'quality_flags': dict(Counter(f for r in rows for f in r['quality_flags'])),
        'admission_blockers': dict(Counter(f for r in rows for f in r['admission_blockers'])),
        'exact_evidence_duplicates': len(duplicates), 'conflicting_reviewed_labels': len(conflicts),
        'quarantined_non_email': sum(r['disposition'] == 'quarantined_non_email' for r in rows),
        'full_template_pairs': len(pairs), 'template_groups': len(set(groups)),
        'approved_for_training': global_valid,
        'independent_label_review_completed': bool(ready) and all(r['independent_review_completed'] for r in ready),
        'extraction_review_completed': bool(ready) and all(r['extraction_verified'] for r in ready),
        'global_partition_problem': global_problem,
        'source_inputs': {'prior_review_corpus_sha256': digest(existing),
                          'spaphish_acquisition_sha256': digest(Path(spaphish) / 'acquisition.json')},
        'preparation_code_sha256': digest(__file__),
        'contract_source_hashes': {name: digest(ROOT / name) for name in
            ('utils/email_input_contract.py','utils/email_candidate_text.py','utils/email_authentication.py','training/email_evidence.py')},
        'unchanged_production_models_and_calibration': True,
        'english_generalization_proven': False,
        'review_note': 'Published labels are provenance, not fabricated independent project review. Spanish data cannot certify English performance. Sender dates are not trusted receiver observations.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing', type=Path, default=ROOT / 'data/email_evidence_review_20261003/review_corpus.jsonl')
    parser.add_argument('--spaphish', type=Path, default=ROOT / 'data/email_evidence_sources/spaphish_v4')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/email_evidence_v1_20261004')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.existing, args.spaphish, args.out, args.annotations), indent=2), flush=True)
