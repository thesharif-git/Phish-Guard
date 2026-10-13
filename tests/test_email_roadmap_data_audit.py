"""Audit metadata must not fabricate data admission or delete evidence repeats."""
from scripts.audit_email_roadmap_stage2 import duplicate_groups, grouping_view, review_gaps


def reviewed_record():
    return dict(task_label='legitimate', label_review_completed=True,
        independent_review_completed=True, extraction_verified=True,
        training_reuse_permitted=True, reviewer_id='first', second_reviewer_id='second',
        label_evidence_reference='archived-evidence', label_rationale='documented verification',
        campaign_cluster_id='family', source='origin')


def test_source_label_and_string_true_cannot_supply_completed_review():
    row = reviewed_record()
    row.update(task_label='unknown', source_claimed_label='publisher_legitimate',
               independent_review_completed='True')
    assert 'task_label_unresolved_or_out_of_scope' in review_gaps(row)
    assert 'independent_review_completed' in review_gaps(row)


def test_same_reviewer_or_missing_evidence_is_not_independent_review():
    row = reviewed_record()
    row.update(second_reviewer_id='first', label_evidence_reference='')
    assert 'reviewers_not_independent' in review_gaps(row)
    assert 'label_evidence_reference' in review_gaps(row)


def test_complete_evidence_repeats_remain_identifiable_with_distinct_labels():
    rows = [dict(id='a', evidence='same', label=0),
            dict(id='b', evidence='same', label=1), dict(id='c', evidence='different')]
    assert duplicate_groups(rows, 'evidence') == [['a', 'b']]
    assert len(rows) == 3 and rows[1]['label'] == 1


def test_template_check_uses_candidate_input_not_old_source_extraction():
    rows = [dict(id='a', body='old publisher extraction', subject='publisher title',
        email_input=dict(body='', html_body='<p>Original message</p>', subject='Actual subject'))]
    assert grouping_view(rows) == [dict(id='a', body='<p>Original message</p>', subject='Actual subject')]
