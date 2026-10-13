"""Boundary regression tests; constructed fixtures are not accuracy evidence."""
import copy
import pytest
from training.email_partition_roles import ROLES, check_role_boundaries
from utils.email_input_contract import build_email_input, contract_sha256


def fixture_rows():
    rows, assignments = [], []
    for index, role in enumerate(ROLES):
        for label in ('legitimate', 'phishing'):
            identity = str(index) + label
            # Whole words differ, so fixture messages are not near duplicates.
            email = build_email_input(' '.join(identity + chr(97 + n) for n in range(20)))
            rows.append(dict(id=identity, email_input=email, input_contract_sha256=contract_sha256(email),
                task_label=label, campaign_cluster_id=identity, research_supervision_eligible=True))
            assignments.append(dict(id=identity, input_contract_sha256=contract_sha256(email), role=role))
    return rows, assignments


def test_five_distinct_roles_never_imply_training_or_deployment_authority():
    rows, assignments = fixture_rows()
    result = check_role_boundaries(rows, assignments)
    assert result['roles_complete']
    assert not result['fitting_ready'] and not result['deployment_approved']
    assert not result['source_time_and_acceptance_review_checked']


def test_unassigned_rows_remain_unassigned_and_missing_roles_visible():
    rows, assignments = fixture_rows()
    for item in assignments:
        item['role'] = None
    result = check_role_boundaries(rows, assignments)
    assert not result['roles_complete'] and result['unassigned_rows'] == len(rows)
    assert set(result['missing_role_classes']) == set(ROLES)


@pytest.mark.parametrize('fault', ['missing_row', 'duplicate_row', 'changed_input', 'legacy_role', 'pending_extraction'])
def test_role_manifest_cannot_drop_rebind_or_admit_pending_rows(fault):
    rows, assignments = fixture_rows()
    if fault == 'missing_row':
        assignments.pop()
    elif fault == 'duplicate_row':
        assignments.append(copy.deepcopy(assignments[0]))
    elif fault == 'changed_input':
        rows[0]['email_input'] = build_email_input('Replacement input cannot reuse review binding')
    elif fault == 'legacy_role':
        assignments[0]['role'] = 'val'
    else:
        rows[0]['research_supervision_eligible'] = False
    with pytest.raises(ValueError):
        check_role_boundaries(rows, assignments)


@pytest.mark.parametrize('field', ['campaign_cluster_id', 'thread_id', 'translation_family_id'])
def test_related_families_cannot_cross_roles(field):
    rows, assignments = fixture_rows()
    rows[0][field] = rows[2][field] = 'one-family'
    with pytest.raises(ValueError, match='crosses roles'):
        check_role_boundaries(rows, assignments)


def test_renaming_campaign_does_not_hide_near_template_leakage():
    rows, assignments = fixture_rows()
    text = ' '.join('distinctword' + str(i) for i in range(150))
    for index, prefix in ((0, 'First introduction'), (2, 'Another introduction')):
        email = build_email_input(prefix + ' ' + text)
        rows[index]['email_input'] = email
        rows[index]['input_contract_sha256'] = assignments[index]['input_contract_sha256'] = contract_sha256(email)
    with pytest.raises(ValueError, match='crosses roles'):
        check_role_boundaries(rows, assignments)
