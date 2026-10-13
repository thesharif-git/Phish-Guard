"""Approved publisher supervision for research, separate from fitting/release gates.

Only the specifically approved SpaPhish v4 release is recognized. A supplied
manifest, URL, source name or `publisher_reviewed` flag cannot confer authority.
The existing project-reviewed training loader and deployment gates stay strict.
"""
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.prepare_email_evidence_contract_corpus import attach_contract, import_spaphish
from training.email_evidence import group_email_templates
from training.email_source_annotations import (
    SPAPHISH_V4_CORPUS_SHA256, SPAPHISH_V4_URL, load_spaphish_publisher_annotations,
)
from utils.email_candidate_text import canonical_email_candidate_text
from utils.email_input_contract import contract_sha256, evidence_sha256, validate_email_input
from utils.jsonl import read_jsonl

POLICY_VERSION = 'email-research-supervision-v1'
TASK_SCOPE = 'phishing_bec_vs_nonmalicious'
PINNED_FILES = {
    'corpus.csv': SPAPHISH_V4_CORPUS_SHA256,
    'schema.json': '51d501037e05f2dd117f8d46c4131d08971fa1f21b220bcc401670741605cdba',
    'README.txt': 'ce44b6c86de0873f8b5d8baf4aaeb1a1710916e81966b97bbc549cfd294ab7ca',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source(directory):
    directory = Path(directory)
    acquisition = json.loads((directory / 'acquisition.json').read_text(encoding='utf-8'))
    requirements = dict(source='SpaPhish', version=4, source_url=SPAPHISH_V4_URL,
                        declared_license='CC BY 4.0', language='Spanish')
    for key, expected in requirements.items():
        if type(acquisition.get(key)) is not type(expected) or acquisition[key] != expected:
            raise ValueError('Approved source provenance mismatch: ' + key)
    declared = acquisition.get('files')
    if not isinstance(declared, list) or len(declared) != len(PINNED_FILES):
        raise ValueError('Approved source file inventory mismatch')
    declared_by_name = {item.get('path'): item for item in declared}
    if set(declared_by_name) != set(PINNED_FILES):
        raise ValueError('Approved source file inventory mismatch')
    for filename, expected in PINNED_FILES.items():
        if declared_by_name[filename].get('sha256') != expected or digest(directory / filename) != expected:
            raise ValueError('Approved source release hash mismatch: ' + filename)
    return acquisition


def source_records(directory):
    """Reconstruct supervision from the verified release, preserving all rows."""
    verify_source(directory)
    annotations = load_spaphish_publisher_annotations(directory)['annotations']
    originals = import_spaphish(directory)
    records = []
    for original, annotation in zip(originals, annotations):
        record = attach_contract(original)
        if record['id'] != annotation['id']:
            raise ValueError('Publisher row/annotation identity mismatch')
        record.update(task_label=annotation['task_label'],
            supervision_authority='publisher_reviewed',
            supervision_policy_version=POLICY_VERSION,
            label_scope=TASK_SCOPE,
            publisher_annotation=annotation,
            project_independent_review_completed=False,
            project_reviewer_ids=[],
            approved_for_training=False,
            approved_for_deployment=False,
            split=None,
            extraction_validation_kind='deterministic binding to released subject/body/URL fields; not independent human review')
        blockers = []
        if record['email_input']['parsing']['completeness'] == 'INCOMPLETE':
            blockers.append('incomplete_extraction_requires_review')
        if any(flag in record['quality_flags'] for flag in
               ('source_url_count_mismatch', 'incomplete_source_url_metadata', 'unparsed_source_url_metadata')):
            blockers.append('source_destination_metadata_requires_review')
        record['research_supervision_blockers'] = blockers
        record['research_supervision_eligible'] = not blockers
        records.append(record)
    if len(records) != len(annotations):
        raise ValueError('Incomplete publisher release import')
    # Identical complete evidence with conflicting labels is an adjudication
    # problem. Similar wording with different identities/targets is not.
    duplicates = defaultdict(list)
    for record in records:
        duplicates[record['evidence_sha256']].append(record)
    for members in duplicates.values():
        if len(members) > 1:
            reason = ('conflicting_complete_evidence_labels_require_review' if
                len({r['task_label'] for r in members}) > 1 else 'exact_complete_evidence_repeat_requires_curation')
            for record in members:
                record['research_supervision_blockers'].append(reason)
                record['research_supervision_eligible'] = False
    return records


def validate_research_record(record, expected):
    """Validate against reconstructed release evidence, not user assertions."""
    if not isinstance(record, dict) or record.get('id') != expected['id']:
        raise ValueError('Unknown publisher record identity')
    for key in ('source', 'task_label', 'label_scope', 'supervision_authority',
                'supervision_policy_version', 'publisher_annotation',
                'original_reference', 'input_contract_sha256', 'evidence_sha256',
                'source_record', 'body', 'subject', 'sender', 'headers', 'html_body',
                'quality_flags', 'language_scope', 'extraction_validation_kind',
                'training_reuse_permitted', 'training_reuse_authority',
                'research_supervision_blockers', 'research_supervision_eligible'):
        # JSON equality distinguishes bools from integers and nested metadata.
        if json.dumps(record.get(key), sort_keys=True, ensure_ascii=False, allow_nan=False) != \
           json.dumps(expected[key], sort_keys=True, ensure_ascii=False, allow_nan=False):
            raise ValueError('Publisher supervision mismatch: ' + key)
    for key in ('project_independent_review_completed', 'independent_review_completed',
                'label_review_completed', 'extraction_verified',
                'approved_for_training', 'approved_for_deployment'):
        if record.get(key) is not False:
            raise ValueError('Research supervision cannot fabricate approval/review: ' + key)
    if record.get('project_reviewer_ids') != [] or record.get('reviewer_id') is not None or record.get('second_reviewer_id') is not None:
        raise ValueError('Publisher supervision cannot fabricate project reviewers')
    evidence = validate_email_input(record.get('email_input'))
    if contract_sha256(evidence) != expected['input_contract_sha256'] or evidence_sha256(evidence) != expected['evidence_sha256']:
        raise ValueError('Publisher input differs from released evidence')
    return copy.deepcopy(evidence)


def research_model_inputs(records, expected_by_id):
    """Shared specialist input projection; outer annotation fields never enter ML."""
    inputs = []
    seen = set()
    for row in records:
        identity = row.get('id')
        if identity in seen or identity not in expected_by_id:
            raise ValueError('Duplicate or unsupported research identity')
        seen.add(identity)
        inputs.append(validate_research_record(row, expected_by_id[identity]))
    return inputs


def prepare_supervision(directory):
    records = source_records(directory)
    view = [dict(id=r['id'], body=r['email_input']['body'] or r['email_input']['html_body'],
                 subject=r['email_input']['subject']) for r in records]
    groups, pairs = group_email_templates(view, canonicalizer=canonical_email_candidate_text)
    for record, group in zip(records, groups):
        record['campaign_cluster_id'] = group
    return records, pairs


def load_research_supervision(corpus_path, manifest_path, *, source_directory):
    """Load verified research supervision; never asserts partition/fitting approval."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    requirements = dict(schema_version=1, policy_version=POLICY_VERSION,
        label_scope=TASK_SCOPE, purpose='research_supervision_preparation',
        partitions_validated=False, approved_for_training=False,
        approved_for_deployment=False, project_independent_review_completed=False)
    for key, expected in requirements.items():
        if type(manifest.get(key)) is not type(expected) or manifest[key] != expected:
            raise ValueError('Research manifest mismatch: ' + key)
    authority = manifest.get('source_authority', {})
    if authority != dict(source_url=SPAPHISH_V4_URL, source_corpus_sha256=SPAPHISH_V4_CORPUS_SHA256,
                         license='CC BY 4.0', kind='publisher_reviewed'):
        raise ValueError('Unsupported research source authority')
    if digest(corpus_path) != manifest.get('corpus_sha256'):
        raise ValueError('Research corpus hash mismatch')
    expected, expected_pairs = prepare_supervision(source_directory)
    expected_by_id = {r['id']: r for r in expected}
    records = read_jsonl(corpus_path)
    if len(records) != len(expected) or set(r.get('id') for r in records) != set(expected_by_id):
        raise ValueError('Research corpus must retain all original publisher records')
    inputs = research_model_inputs(records, expected_by_id)
    for row in records:
        if row.get('campaign_cluster_id') != expected_by_id[row['id']]['campaign_cluster_id']:
            raise ValueError('Research template group mismatch')
        if row.get('split') is not None:
            raise ValueError('Prepared supervision is unpartitioned; use a separately validated partition protocol')
    actual_counts = dict(records=len(records), eligible=sum(r['research_supervision_eligible'] for r in records),
        pending_extraction_review=sum(not r['research_supervision_eligible'] for r in records),
        template_groups=len({r['campaign_cluster_id'] for r in expected}), template_pairs=len(expected_pairs))
    if manifest.get('counts') != actual_counts:
        raise ValueError('Research manifest counts differ from verified records')
    eligible = [r for r in records if r['research_supervision_eligible']]
    return records, inputs, dict(actual_counts,
        eligible_task_labels=dict(Counter(r['task_label'] for r in eligible)),
        fitting_ready=False, deployment_approved=False)
