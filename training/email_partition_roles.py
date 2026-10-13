"""Five-role partition boundary checks for the approved Email roadmap.

Call after source-authority verification. Boundary validity does not certify
labels, native languages, unseen acceptance, calibration, or deployment.
No roles are inferred or assigned here; missing assignments remain missing.
"""
from collections import Counter, defaultdict

from training.email_evidence import group_email_templates
from utils.email_candidate_text import canonical_email_candidate_text
from utils.email_input_contract import contract_sha256, evidence_sha256

ROLES = ('train', 'development_validation', 'calibration',
         'operating_point_validation', 'locked_acceptance')


def check_role_boundaries(records, assignments):
    """Reject altered input bindings, duplicate IDs and cross-role relatives.

An assignment has id, input_contract_sha256 and role. All records (including
pending extraction) must be retained in the inventory. None is an explicit
unassigned role, never an implicit training or test allocation.
"""
    expected = {row['id']: row for row in records}
    supplied = {row['id']: row for row in assignments}
    if len(expected) != len(records) or len(supplied) != len(assignments):
        raise ValueError('Duplicate partition identity')
    if set(expected) != set(supplied):
        raise ValueError('Partition inventory must retain every verified source record')
    families = defaultdict(set)
    counts = {role: Counter() for role in ROLES}
    view, roles, unassigned = [], [], []
    for row in records:
        item = supplied[row['id']]
        actual_hash = contract_sha256(row['email_input'])
        if row.get('input_contract_sha256') != actual_hash or item.get('input_contract_sha256') != actual_hash:
            raise ValueError('Partition input binding mismatch')
        role = item.get('role')
        if role is not None and role not in ROLES:
            raise ValueError('Unknown role; the approved roadmap requires five separate roles')
        if role is not None:
            if row.get('research_supervision_eligible') is not True:
                raise ValueError('Pending evidence cannot receive a fitting/evaluation role')
            label = row.get('task_label')
            if label not in ('phishing', 'bec', 'legitimate'):
                raise ValueError('Unsupported phishing/BEC task label')
            counts[role]['legitimate' if label == 'legitimate' else 'phishing_or_bec'] += 1
            families[('complete_evidence', evidence_sha256(row['email_input']))].add(role)
            for field in ('campaign_cluster_id', 'thread_id', 'translation_family_id'):
                family = row.get(field)
                if family:
                    families[(field, str(family))].add(role)
        else:
            unassigned.append(row['id'])
        roles.append(role)
        view.append(dict(id=row['id'], body=row['email_input']['body'] or row['email_input']['html_body'],
                         subject=row['email_input']['subject']))
    # Do not trust renamed groups to conceal near-template leakage.
    computed, pairs = group_email_templates(view, canonicalizer=canonical_email_candidate_text)
    for group, role in zip(computed, roles):
        if role is not None:
            families[('recomputed_template', group)].add(role)
    overlap = [(kind, identity) for (kind, identity), values in families.items() if len(values) > 1]
    if overlap:
        raise ValueError('Related email evidence crosses roles: ' + str(overlap[:5]))
    missing_classes = {role: sorted({'legitimate', 'phishing_or_bec'} - set(counts[role]))
                       for role in ROLES if set(counts[role]) != {'legitimate', 'phishing_or_bec'}}
    return dict(role_counts={role: dict(counts[role]) for role in ROLES},
        unassigned_rows=len(unassigned), roles_complete=not unassigned and not missing_classes,
        missing_role_classes=missing_classes, cross_role_families=0,
        recomputed_template_groups=len(set(computed)), recomputed_near_pairs=len(pairs),
        source_authority_checked_by_this_function=False,
        source_time_and_acceptance_review_checked=False, fitting_ready=False,
        deployment_approved=False)
