"""Prepare the five-role inventory; do not invent data or allocate holdouts."""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.email_partition_roles import ROLES, check_role_boundaries
from training.email_research_admission import digest, load_research_supervision

ROADMAP = ROOT / 'benchmarks/email_roadmap_v1_20261006'
ADMISSION = ROADMAP / 'publisher_admission_v1_20261007_v2'
SOURCE = ROOT / 'data/email_evidence_sources/spaphish_v4'
OUT = ROADMAP / 'partition_readiness_v1_20261007'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main():
    if OUT.exists():
        raise FileExistsError('Refusing to overwrite recorded readiness evidence')
    OUT.mkdir(parents=True)
    try:
        protected = json.loads((ROADMAP / 'protected_snapshot.json').read_text(encoding='utf-8'))
        if any(digest(path) != sha for path, sha in protected.items()):
            raise ValueError('Protected baseline changed; investigate before proceeding')
        records, _, supervision = load_research_supervision(ADMISSION / 'research_candidates.jsonl',
            ADMISSION / 'manifest.json', source_directory=SOURCE)
        assignments = [dict(id=row['id'], input_contract_sha256=row['input_contract_sha256'],
            role=None, assignment_status='pending_reviewed_partition_protocol',
            research_supervision_eligible=row['research_supervision_eligible']) for row in records]
        print('Checking all retained records and five-role boundaries', flush=True)
        boundaries = check_role_boundaries(records, assignments)
        eligible = [row for row in records if row['research_supervision_eligible']]
        source_labels = Counter((row['source'], row['task_label']) for row in eligible)
        day_count = sum(row['publisher_annotation']['observed_day'] is not None for row in eligible)
        report = dict(schema_version=1, complete=True, stage_gate_passed=False,
            generated_at_utc=datetime.now(timezone.utc).isoformat(), approved_roles=list(ROLES),
            supervision=supervision, partition_boundaries=boundaries,
            eligible_source_labels=[dict(source=source, task_label=label, rows=count)
                for (source, label), count in sorted(source_labels.items())],
            publisher_reported_day_rows=day_count,
            independently_verified_reception_time_rows=sum(
                row['publisher_annotation']['receiver_time_verified'] is True for row in eligible),
            project_independently_reviewed_rows=sum(
                row['project_independent_review_completed'] is True for row in eligible),
            blockers=['Only one admitted publisher/source; source-separated generalization cannot be established from it alone',
                'No admitted English, Urdu or Arabic supervision in this verified inventory',
                'Inspected Spanish source is development evidence, not a newly locked independent acceptance collection',
                'Publisher reception days do not establish receiver-verified timestamps; no UTC times invented',
                'Six extraction-pending records remain present and unassigned',
                'Five distinct roles have not been allocated or approved; no training/calibration allowed by this report'],
            proposed_collection=['Permissioned native original emails from multiple independent inbox/reporting sources',
                'Preserve MIME, subject/body, actual destinations and authentication origin; missing evidence remains unavailable',
                'Independently adjudicate phishing/BEC versus legitimate with language-competent reviewers',
                'Include legitimate password resets, security alerts, invoices and urgent business requests',
                'Retain spam as distinct task provenance, not an automatic phishing label',
                'Bind reviews to original/input hashes; keep threads/campaigns/translations within one role',
                'Reserve uninspected future/source cohorts before fitting; existing failed external sets stay development'],
            new_training=False, calibration_fitted=False, production_promoted=False,
            frontend_changes=False)
        with (OUT / 'pending_assignments.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for row in assignments:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if any(digest(path) != sha for path, sha in protected.items()):
            raise ValueError('Protected baseline changed during readiness verification')
        report['protected_unchanged'] = True
        report['input_bindings'] = {str(path.relative_to(ROOT)): digest(path) for path in
            (ADMISSION / 'research_candidates.jsonl', ADMISSION / 'manifest.json',
             ROOT / 'training/email_partition_roles.py', Path(__file__))}
        save(OUT / 'results.json', report)
        save(OUT / 'run_status.json', dict(complete=True, stage='readiness_checked_not_partitioned',
            stage_gate_passed=False, fitting_ready=False))
        print(json.dumps(dict(rows=len(records), eligible=len(eligible),
            roles_complete=boundaries['roles_complete'], fitting_ready=False,
            protected_unchanged=True)), flush=True)
    except Exception as error:
        save(OUT / 'run_status.json', dict(complete=False, stage='failed',
            error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
