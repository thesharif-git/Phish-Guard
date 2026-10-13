"""Review intake must preserve difficult originals and never invent approval."""
import hashlib
import json

import pytest

from scripts.prepare_email_public_source_intake import pending_manual_review


def test_uncertain_spam_and_noncanonical_manual_rows_are_retained():
    seeds = [
        {'Source': 'Manual', 'Body': 'مرحبا\u2028世界', 'Type': 'Spam',
         'Language': 'en', 'Created by': 'AI/Human', 'Year': 1900},
        {'Source': 'Public', 'Type': 'Phishing'},
        {'Source': 'Manual', 'Body': '', 'Type': 'unknown', 'Language': None},
    ]
    inventory = [dict(source_row_index=i, exclusion_reasons=['historically_excluded'])
                 for i in range(3)]
    prepared = [dict(source='phishfuzzer_manual', id='existing',
                     original_reference={'row_index': 0}, input_contract_sha256='bound-input')]
    rows = pending_manual_review(seeds, prepared, inventory, 'bound-source')
    assert [r['source_row_index'] for r in rows] == [0, 2]
    assert rows[0]['source_claimed_label'] == 'Spam'
    assert rows[0]['reviewed_task_label'] is None
    assert rows[1]['input_contract_sha256'] is None
    assert rows[1]['prepared_input_id'] is None
    expected = hashlib.sha256(json.dumps(seeds[0], sort_keys=True,
                             ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    assert rows[0]['source_record_sha256'] == expected
    for row in rows:
        assert row['role'] is None
        assert row['reviewer_ids'] == []
        assert row['language_independently_verified'] is False
        assert row['eligible_for_new_blind_acceptance'] is False
        assert row['approved_for_training'] is False
        assert row['research_supervision_eligible'] is False
        assert row['project_independent_review_completed'] is False


def test_prepared_source_identity_mismatch_fails_instead_of_dropping_row():
    seeds = [{'Source': 'Public'}]
    prepared = [dict(source='phishfuzzer_manual', id='existing',
                     original_reference={'row_index': 0}, input_contract_sha256='bound-input')]
    with pytest.raises(ValueError, match='absent from original manual inventory'):
        pending_manual_review(seeds, prepared, [], 'bound-source')


def test_duplicate_prepared_source_index_is_rejected():
    prepared = [dict(source='phishfuzzer_manual', original_reference={'row_index': 0})] * 2
    with pytest.raises(ValueError, match='Duplicate prepared source index'):
        pending_manual_review([], prepared, [], 'bound-source')
