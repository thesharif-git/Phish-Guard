"""Reject the actual provenance/template failures found in the Email audit."""
import pandas as pd
import pytest
from training.email_evidence import full_message_family, group_email_templates, validate_email_evidence_partitions


def reviewed_fixture():
    rows = []
    for split in ('train', 'val', 'source_test', 'time_test'):
        for source in (('origin_a', 'origin_b') if split == 'train' else ('origin_c',) if split == 'source_test' else ('origin_a',)):
            for label in (0, 1):
                identifier = f'{split}_{source}_class_{label}'
                rows.append(dict(raw_text=' '.join(identifier + chr(97+i) for i in range(12)), label=label, split=split,
                    source=source, campaign_cluster_id=identifier, label_review_completed=True, reviewer_id='fixture-reviewer',
                    label_evidence_reference='fixture://annotation/'+identifier,
                    observed_at_utc='2025-01-01T12:00:00Z' if split == 'time_test' else '2024-01-01T12:00:00Z',
                    time_authority='publisher_observation'))
    return pd.DataFrame(rows)


def test_punctuation_versions_of_same_entire_message_have_same_family():
    assert full_message_family('Please, review the full document today!') == full_message_family('please review the full document today')


def test_full_template_check_catches_prefix_change_that_first_words_miss():
    body = ' '.join('distinctword'+str(i) for i in range(120))
    records = [{'id': 'first', 'body': 'Notice for recipients. '+body}, {'id': 'second', 'body': 'Different archive introduction. '+body}]
    groups, matches = group_email_templates(records)
    assert groups[0] == groups[1] and matches[0]['full_trigram_jaccard'] >= .9


def test_valid_reviewed_source_and_time_partitions_pass():
    assert validate_email_evidence_partitions(reviewed_fixture())['time_test_strictly_later']


def test_legacy_source_label_export_cannot_overwrite_email_corpus(monkeypatch):
    from scripts import clean_and_curate_master_datasets as legacy
    monkeypatch.setattr(legacy.pd, 'read_csv', lambda *a, **k: pytest.fail('Must fail before reading or writing any legacy dataset'))
    with pytest.raises(ValueError, match='cannot overwrite'):
        legacy.clean_and_curate_emails()


@pytest.mark.parametrize('problem', ['source_overlap', 'unreviewed_label', 'untrusted_date', 'time_overlap', 'content_variant'])
def test_evidence_partition_failures_cannot_be_certified(problem):
    frame = reviewed_fixture()
    if problem == 'source_overlap':
        frame.loc[frame.split == 'source_test', 'source'] = 'origin_a'
    elif problem == 'unreviewed_label':
        frame.loc[0, 'label_review_completed'] = False
    elif problem == 'untrusted_date':
        frame.loc[0, 'time_authority'] = 'sender_Date_header'
    elif problem == 'time_overlap':
        frame.loc[frame.split == 'time_test', 'observed_at_utc'] = '2023-01-01T00:00:00Z'
    else:
        index = frame[frame.split == 'source_test'].index[0]
        frame.loc[index, 'raw_text'] = frame.loc[0, 'raw_text'].upper() + '!!!'
    with pytest.raises(ValueError):
        validate_email_evidence_partitions(frame)
