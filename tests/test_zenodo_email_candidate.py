"""Candidate inspection cannot turn source claims into certified labels."""
import pytest
from scripts.audit_zenodo_email_candidate import inspect_csv


def test_repeated_body_and_label_conflict_are_preserved():
    result = inspect_csv(b'Email Text,Email Type\nHello  world,Safe Email\nHELLO world,Phishing Email\nDifferent mail,Safe Email\n')
    assert result['rows'] == 3
    assert result['normalized_unique_nonempty_bodies'] == 2
    assert result['repeat_excess_rows'] == result['conflicting_label_body_groups'] == 1
    assert len(result['repeated_bodies'][0]['rows']) == 2
    assert not result['approved_for_training'] and not result['approved_independent_benchmark']


def test_malformed_empty_and_unknown_labels_are_explicit():
    result = inspect_csv(b'Email Text,Email Type\n,Safe Email\nSome message,Spam,extra\nMissing label\n')
    assert result['empty_body_row_indices'] == [0]
    assert result['malformed_row_indices'] == [1, 2]
    assert result['unknown_label_counts'] == {'Spam': 1, '': 1}


def test_different_destinations_are_not_collapsed():
    result = inspect_csv(b'Email Text,Email Type\nVisit https://bank.example,Safe Email\nVisit https://attacker.example,Phishing Email\n')
    assert result['normalized_unique_nonempty_bodies'] == 2
    assert result['repeated_body_groups'] == 0


def test_wrong_task_schema_rejected():
    with pytest.raises(ValueError, match='schema'):
        inspect_csv(b'URL,label\nhttps://example.org,0\n')
