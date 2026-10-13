"""Prepare approved publisher supervision and reconcile Brown review bindings.

No fitting, calibration, production writes, invented human review or UI edits.
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.email_research_admission import (
    POLICY_VERSION, TASK_SCOPE, digest, load_research_supervision, prepare_supervision,
)
from training.email_source_annotations import SPAPHISH_V4_CORPUS_SHA256, SPAPHISH_V4_URL
from utils.email_input_contract import contract_sha256, evidence_sha256
from utils.jsonl import read_jsonl

SOURCE = ROOT / 'data/email_evidence_sources/spaphish_v4'
DEFAULT_OUT = ROOT / 'benchmarks/email_roadmap_v1_20261006/publisher_admission_v1_20261007'
CODE_FILES = ('training/email_research_admission.py', 'training/email_source_annotations.py',
    'scripts/prepare_email_research_admission.py', 'scripts/prepare_email_evidence_contract_corpus.py',
    'training/email_evidence.py', 'utils/email_input_contract.py', 'utils/email_authentication.py',
    'utils/email_candidate_text.py', 'utils/text_sanitizer.py', 'utils/jsonl.py')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def save_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def reconcile_brown(old_rows, latest_rows):
    """Rebind review metadata to latest extraction; transfer no review attestation."""
    old = {r['id']: r for r in old_rows if r['source'] == 'brown_phish_bowl'}
    if len(old) != len(latest_rows) or set(old) != {r['id'] for r in latest_rows}:
        raise ValueError('Brown reconciliation must retain the same original records')
    result = []
    for row in latest_rows:
        original = old[row['id']]
        current_hash = contract_sha256(row['email_input'])
        if current_hash != row['input_contract_sha256']:
            raise ValueError('Current Brown extraction contract hash mismatch')
        reference = row['original_reference']
        # Latest extraction paths are relative to the acquisition folder, while
        # the older corpus stores workspace-relative paths. Bind both forms to
        # the same archived page; never infer a new message from a path prefix.
        path = (ROOT / 'data/email_root_audit_sources_20261003' / reference['snapshot_path']).resolve()
        prior_path = (ROOT / original['original_reference']['path']).resolve()
        if path != prior_path or not path.is_relative_to(ROOT) or digest(path) != reference['snapshot_sha256']:
            raise ValueError('Brown original snapshot binding mismatch')
        if original['original_reference']['url'] != reference['url']:
            raise ValueError('Brown publisher message identity changed')
        result.append(dict(id=row['id'], source=row['source'],
            previous_input_contract_sha256=original['input_contract_sha256'],
            input_contract_sha256=current_hash,
            input_changed=current_hash != original['input_contract_sha256'],
            original_reference=reference, email_input=row['email_input'],
            evidence_sha256=evidence_sha256(row['email_input']),
            publisher_task_label=row['task_label'],
            label_authority_kind=row['label_authority_kind'],
            label_authority_reference=row['label_authority_reference'],
            eligible_for_text_metrics=row['eligible_for_text_metrics'],
            extraction_exclusion_reasons=row['exclusion_reasons'],
            extraction_coverage_reasons=row['coverage_reasons'],
            project_independent_review_completed=False, project_reviewer_ids=[],
            previous_review_transferred=False,
            training_reuse_permission_established=False,
            research_supervision_eligible=False, approved_for_training=False,
            approved_for_deployment=False,
            status='latest_extraction_bound; project review and reuse unresolved'))
    return result


def prepare(out):
    out = Path(out).resolve()
    roadmap = ROOT / 'benchmarks/email_roadmap_v1_20261006'
    if not out.is_relative_to(roadmap) or out == roadmap or out.exists():
        raise ValueError('Use a fresh child folder under the approved roadmap benchmarks')
    out.mkdir(parents=True)
    try:
        protected = json.loads((roadmap / 'protected_snapshot.json').read_text(encoding='utf-8'))
        if any(digest(path) != sha for path, sha in protected.items()):
            raise ValueError('Protected baseline changed; inspect before proceeding')
        prior = ROOT / 'data/email_evidence_v1_20261004_v2/review_corpus.jsonl'
        latest = ROOT / 'data/brown_email_benchmark_20261005/all_records.jsonl'
        tracked = {str(path): digest(path) for path in
            (prior, latest, SOURCE / 'acquisition.json', SOURCE / 'corpus.csv',
             SOURCE / 'schema.json', SOURCE / 'README.txt')}
        print('Preparing hash-bound publisher supervision', flush=True)
        records, pairs = prepare_supervision(SOURCE)
        save_rows(out / 'research_candidates.jsonl', records)
        save_rows(out / 'template_pairs.jsonl', pairs)
        save_rows(out / 'pending_extraction_review.jsonl', [dict(id=r['id'],
            input_contract_sha256=r['input_contract_sha256'], original_reference=r['original_reference'],
            blockers=r['research_supervision_blockers']) for r in records if not r['research_supervision_eligible']])
        reconciled = reconcile_brown(read_jsonl(prior), read_jsonl(latest))
        save_rows(out / 'brown_review_reconciliation.jsonl', reconciled)
        counts = dict(records=len(records), eligible=sum(r['research_supervision_eligible'] for r in records),
            pending_extraction_review=sum(not r['research_supervision_eligible'] for r in records),
            template_groups=len({r['campaign_cluster_id'] for r in records}), template_pairs=len(pairs))
        manifest = dict(schema_version=1, policy_version=POLICY_VERSION, label_scope=TASK_SCOPE,
            purpose='research_supervision_preparation', corpus_sha256=digest(out / 'research_candidates.jsonl'),
            source_authority=dict(source_url=SPAPHISH_V4_URL,
                source_corpus_sha256=SPAPHISH_V4_CORPUS_SHA256, license='CC BY 4.0', kind='publisher_reviewed'),
            counts=counts, original_inputs_sha256=tracked,
            preparation_code_sha256={name: digest(ROOT / name) for name in CODE_FILES},
            partitions_validated=False, approved_for_training=False,
            approved_for_deployment=False, project_independent_review_completed=False,
            expected_model_input='email_input only; annotations/source/label/date/reviewer rationale excluded',
            brown_reconciliation_rows=len(reconciled),
            brown_changed_input_rows=sum(r['input_changed'] for r in reconciled),
            brown_review_transferred_rows=0,
            prepared_at_utc=datetime.now(timezone.utc).isoformat())
        save(out / 'manifest.json', manifest)
        print('Reloading output against release authority and extraction', flush=True)
        _, _, checked = load_research_supervision(out / 'research_candidates.jsonl',
            out / 'manifest.json', source_directory=SOURCE)
        if any(digest(path) != sha for path, sha in tracked.items()) or any(digest(path) != sha for path, sha in protected.items()):
            raise ValueError('Original/protected files changed during preparation')
        save(out / 'verification.json', dict(complete=True, supervision=checked,
            protected_unchanged=True, original_sources_unchanged=True,
            frontend_changes=False, new_training=False, calibration_fitted=False))
        save(out / 'run_status.json', dict(complete=True, stage='publisher_supervision_verified',
            counts=counts, training_started=False, fitting_ready=False, deployment_approved=False))
        print(json.dumps(checked, allow_nan=False), flush=True)
        return manifest, checked
    except Exception as error:
        save(out / 'run_status.json', dict(complete=False, stage='failed',
            error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    prepare(parser.parse_args().out)
