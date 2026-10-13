"""Authority/leakage boundaries; these tests do not measure phishing accuracy."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from training.email_research_admission import (
    load_research_supervision, research_model_inputs, source_records, validate_research_record, verify_source,
)
from utils.email_input_contract import build_email_input, contract_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/email_evidence_sources/spaphish_v4'


@pytest.fixture(scope='module')
def admitted_source():
    return source_records(SOURCE)


@pytest.mark.parametrize('field,value', [('declared_license', 'unknown'),
    ('source_url', 'https://unreviewed.example/dataset'), ('version', 5), ('source', 'Nazario')])
def test_unapproved_source_license_or_release_fails(tmp_path, field, value):
    acquisition = json.loads((SOURCE / 'acquisition.json').read_text(encoding='utf-8'))
    acquisition[field] = value
    (tmp_path / 'acquisition.json').write_text(json.dumps(acquisition), encoding='utf-8')
    with pytest.raises(ValueError, match='provenance mismatch'):
        verify_source(tmp_path)


def test_self_consistent_rehashed_release_cannot_acquire_authority(tmp_path):
    for filename in ('acquisition.json', 'corpus.csv', 'schema.json', 'README.txt'):
        shutil.copyfile(SOURCE / filename, tmp_path / filename)
    with (tmp_path / 'corpus.csv').open('ab') as stream:
        stream.write(b'\n')
    import hashlib
    acquisition = json.loads((tmp_path / 'acquisition.json').read_text(encoding='utf-8'))
    acquisition['files'][0]['sha256'] = hashlib.sha256((tmp_path / 'corpus.csv').read_bytes()).hexdigest()
    (tmp_path / 'acquisition.json').write_text(json.dumps(acquisition), encoding='utf-8')
    with pytest.raises(ValueError, match='release hash mismatch'):
        verify_source(tmp_path)


@pytest.mark.parametrize('key,value', [('task_label', 'bec'), ('label_scope', 'spam_vs_ham'),
    ('supervision_authority', 'collector_claim'), ('training_reuse_permitted', False),
    ('project_independent_review_completed', True), ('approved_for_training', True),
    ('approved_for_deployment', True), ('independent_review_completed', True)])
def test_authority_does_not_allow_label_or_approval_rewrites(admitted_source, key, value):
    original = admitted_source[0]
    forged = copy.deepcopy(original)
    forged[key] = value
    with pytest.raises(ValueError):
        validate_research_record(forged, original)


def test_recomputed_input_hash_does_not_rebind_a_publisher_label(admitted_source):
    original = admitted_source[0]
    forged = copy.deepcopy(original)
    forged['email_input'] = build_email_input('A completely different email')
    forged['input_contract_sha256'] = contract_sha256(forged['email_input'])
    with pytest.raises(ValueError):
        validate_research_record(forged, original)


def test_publisher_labels_are_usable_supervision_without_fake_project_review(admitted_source):
    assert len(admitted_source) == 1395
    assert {r['task_label'] for r in admitted_source} == {'legitimate', 'phishing'}
    assert any(r['research_supervision_eligible'] for r in admitted_source)
    for row in admitted_source:
        assert not row['project_independent_review_completed']
        assert row['reviewer_id'] is None and row['second_reviewer_id'] is None
        assert not row['approved_for_training'] and not row['approved_for_deployment']


def test_annotation_date_source_and_rationale_never_enter_model_input(admitted_source):
    original = admitted_source[0]
    annotated = copy.deepcopy(original)
    annotated.update(model_score=0.99, annotation_rationale='phishing definitely', source_selection_notes='dataset hint')
    projected = research_model_inputs([annotated], {original['id']: original})[0]
    assert projected == original['email_input']
    assert not {'source', 'task_label', 'publisher_annotation', 'source_record', 'model_score'} & set(projected)
    forged = copy.deepcopy(original)
    forged['email_input']['task_label'] = 'phishing'
    with pytest.raises(ValueError):
        research_model_inputs([forged], {original['id']: original})


def test_incomplete_destinations_remain_pending_not_deleted(admitted_source):
    incomplete = [r for r in admitted_source if r['email_input']['parsing']['completeness'] == 'INCOMPLETE']
    assert incomplete
    assert all(not r['research_supervision_eligible'] and r['research_supervision_blockers'] for r in incomplete)
    assert len(admitted_source) == 1395


def test_duplicate_or_unknown_identity_cannot_inflate_research_rows(admitted_source):
    row = admitted_source[0]
    for records in ([row, row], [dict(row, id='unapproved-source-row')]):
        with pytest.raises(ValueError):
            research_model_inputs(records, {row['id']: row})


def test_original_annotation_provenance_cannot_be_rewritten(admitted_source):
    original = admitted_source[0]
    forged = copy.deepcopy(original)
    forged['source_record']['Label'] = '0' if forged['source_record']['Label'] == '1' else '1'
    with pytest.raises(ValueError, match='source_record'):
        validate_research_record(forged, original)


@pytest.mark.parametrize('field,value', [('label_scope', 'spam_vs_ham'),
    ('approved_for_training', True), ('partitions_validated', True),
    ('approved_for_deployment', True), ('project_independent_review_completed', True)])
def test_manifest_cannot_turn_supervision_into_fitting_or_release(tmp_path, field, value):
    from training.email_research_admission import POLICY_VERSION, TASK_SCOPE
    manifest = dict(schema_version=1, policy_version=POLICY_VERSION, label_scope=TASK_SCOPE,
        purpose='research_supervision_preparation', partitions_validated=False,
        approved_for_training=False, approved_for_deployment=False,
        project_independent_review_completed=False)
    manifest[field] = value
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest mismatch'):
        load_research_supervision(tmp_path / 'unused.jsonl', path, source_directory=SOURCE)


def test_brown_path_conventions_bind_same_snapshot_and_invalidate_old_review(tmp_path, monkeypatch):
    from scripts import prepare_email_research_admission as preparation
    import hashlib
    monkeypatch.setattr(preparation, 'ROOT', tmp_path)
    path = tmp_path / 'data/email_root_audit_sources_20261003/brown_pages/message.html'
    path.parent.mkdir(parents=True)
    path.write_text('Fictional archived page for path-binding test', encoding='utf-8')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    url = 'https://it.brown.edu/phish-bowl-alerts/fixture'
    evidence = build_email_input('Actual extracted fixture email body')
    old = dict(id='fixture', source='brown_phish_bowl', input_contract_sha256='previous-view',
        original_reference=dict(url=url, path=str(path.relative_to(tmp_path))))
    latest = dict(id='fixture', source='brown_phish_bowl', email_input=evidence,
        input_contract_sha256=contract_sha256(evidence), task_label='legitimate',
        original_reference=dict(url=url, snapshot_path='brown_pages/message.html', snapshot_sha256=sha),
        label_authority_kind='publisher_security_team_verdict', label_authority_reference=url,
        eligible_for_text_metrics=True, exclusion_reasons=[], coverage_reasons=[])
    result = preparation.reconcile_brown([old], [latest])[0]
    assert result['input_changed'] and not result['previous_review_transferred']
    assert not result['research_supervision_eligible'] and not result['approved_for_deployment']
    latest['original_reference']['snapshot_path'] = '../other.html'
    with pytest.raises(ValueError, match='snapshot binding mismatch'):
        preparation.reconcile_brown([old], [latest])
