import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_email_evidence_contract_corpus import (
    admission_blockers, apply_annotation, attach_contract, declared_url_list, declared_count,
    folder_metadata, import_spaphish, load_annotations, prepare,
)
from utils.email_input_contract import build_email_input, contract_sha256, evidence_sha256


def sample(**changes):
    row = dict(id='a', source='nazario', body='Please review the attached report.',
               subject='Report', sender='Editor <editor@example.org>', headers='',
               html_body='', quality_flags=[], original_reference={'message_index': 3},
               source_claimed_label='collector_claimed_phishing', training_reuse_permitted=True)
    row.update(changes)
    return row


def spaphish_fixture(tmp_path):
    directory = tmp_path / 'spa'
    directory.mkdir()
    with (directory / 'corpus.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter=';', fieldnames=['hash','subject','body','date','url_count','urls','Label','justif_authority_A'])
        writer.writeheader()
        writer.writerow(dict(hash='source-id', subject='Mensaje', body='Contenido original', date='01/02/2025',
                             url_count='2', urls='[https://example.org/confirm],[#]', Label='1',
                             justif_authority_A='PHISHING LABEL RATIONALE MUST NOT BE MODEL INPUT'))
    (directory / 'schema.json').write_text('{}', encoding='utf-8')
    (directory / 'README.txt').write_text('Declared fixture schema', encoding='utf-8')
    import hashlib
    files = [{'path': name, 'sha256': hashlib.sha256((directory / name).read_bytes()).hexdigest()}
             for name in ('corpus.csv','schema.json','README.txt')]
    (directory / 'acquisition.json').write_text(json.dumps(dict(version=4,declared_license='CC BY 4.0',
        source_url='https://data.mendeley.com/datasets/hz2d6gz7pc/4',files=files)), encoding='utf-8')
    return directory


def test_publisher_label_does_not_become_training_truth():
    row = attach_contract(sample())
    assert row['task_label'] == 'unknown'
    assert row['label_review_completed'] is False
    assert row['approved_for_training'] is False
    assert 'collector_claimed_phishing' not in json.dumps(row['email_input'])


def test_real_folder_metadata_is_quarantined_and_not_merely_x_imap():
    row = sample(headers='X-IMAP: 123 456', subject="DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA",
                 body='This text is part of the mail folder, and is not\na real message.')
    assert folder_metadata(row)
    result = attach_contract(row)
    assert result['task_label'] == 'non_email'
    assert result['disposition'] == 'quarantined_non_email'
    assert not folder_metadata(sample(headers='X-IMAP: 123 456'))
    with pytest.raises(ValueError, match='cannot be admitted'):
        apply_annotation(result, {'task_label':'legitimate'})


def test_unusual_or_long_valid_message_is_retained():
    original = sample(body='x' * 100_001)
    result = attach_contract(original)
    assert result['body'] == original['body']
    assert result['disposition'] == 'awaiting_review'
    assert 'long_body_requires_review_not_automatic_deletion' in result['quality_flags']


def test_declaration_parser_keeps_non_http_and_ipv6():
    values, flags = declared_url_list('[http://[::1]/a],[mailto:sender@example.org],[#]')
    assert values == ['http://[::1]/a','mailto:sender@example.org','#']
    assert flags == []
    assert declared_url_list('malformed') == ([], ['unparsed_source_url_metadata'])


def test_source_truncation_preserves_only_complete_values():
    assert declared_url_list('[https://example.org/],[http://[::1]/a],[https://truncated') == (
        ['https://example.org/', 'http://[::1]/a'], ['incomplete_source_url_metadata'])
    assert declared_count('32.00') == 32
    assert declared_count('1.50') is None
    assert declared_count('NaN') is None


def test_source_missing_fields_dates_and_rationales_are_not_invented(tmp_path):
    rows = import_spaphish(spaphish_fixture(tmp_path))
    item = attach_contract(rows[0])
    assert item['sender'] == item['headers'] == ''
    assert item['observed_at_utc'] is None
    assert item['time_authority'] == 'unverified'
    assert item['email_input']['parsing']['completeness'] == 'UNKNOWN'
    assert item['email_input']['trusted_authentication'] is None
    assert item['email_input']['link_destinations'] == ['https://example.org/confirm']
    assert item['email_input']['unsupported_urls'][0]['raw_value'] == '#'
    assert 'Rationale' not in json.dumps(item['email_input'])
    assert 'PHISHING LABEL RATIONALE' not in json.dumps(item['email_input'])
    assert item['source_record']['justif_authority_A'].startswith('PHISHING')
    assert item['task_label'] == 'unknown'


def test_source_tamper_is_rejected(tmp_path):
    directory = spaphish_fixture(tmp_path)
    (directory / 'corpus.csv').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        import_spaphish(directory)


def test_annotation_requires_matching_exact_evidence(tmp_path):
    row = attach_contract(sample())
    path = tmp_path / 'annotation.csv'
    path.write_text('id,input_contract_sha256,task_label\na,wrong,phishing\n', encoding='utf-8')
    with pytest.raises(ValueError, match='different Email evidence'):
        load_annotations(path, [row])


def test_even_blank_duplicate_annotation_ids_are_rejected(tmp_path):
    row = attach_contract(sample())
    path = tmp_path / 'annotation.csv'
    path.write_text('id,input_contract_sha256,task_label\na,,\na,,\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate annotation ID'):
        load_annotations(path, [row])


def test_non_boolean_license_permission_is_not_approval():
    row = attach_contract(sample(training_reuse_permitted='false'))
    assert 'training_reuse_permission_not_established' in admission_blockers(row)


@pytest.mark.parametrize('label', ['spam','unknown','non_email'])
def test_out_of_scope_labels_cannot_be_binary_training_labels(label):
    row = attach_contract(sample())
    row['task_label'] = label
    assert 'no_in_scope_reviewed_task_label' in admission_blockers(row)


def test_blank_or_same_reviewers_cannot_certify_data():
    row = attach_contract(sample())
    row.update(task_label='phishing',label_review_completed=True,independent_review_completed=True,
               extraction_verified=True,reviewer_id='a',second_reviewer_id='a')
    assert 'independent_identifiable_reviewers_missing' in admission_blockers(row)


def test_dedup_ignores_parser_notes_but_not_identity():
    first = build_email_input('Message', sender='a@example.org', parsing_info={'notes': ['first parser']})
    second = build_email_input('Message', sender='a@example.org', parsing_info={'notes': ['second parser']})
    different_sender = build_email_input('Message', sender='b@example.org', parsing_info={'notes': ['first parser']})
    assert contract_sha256(first) != contract_sha256(second)
    assert evidence_sha256(first) == evidence_sha256(second)
    assert evidence_sha256(first) != evidence_sha256(different_sender)


def test_displayed_url_is_not_invented_as_clickable_destination():
    record = build_email_input('', html_body='<a href="https://target.example/verify"><span>https://bank.example/</span></a>')
    assert record['link_destinations'] == ['https://target.example/verify']
    assert record['links'][0]['anchor_text'] == 'https://bank.example/'


def test_builder_never_trains_source_claims_and_preserves_audit(tmp_path):
    path = tmp_path / 'existing.jsonl'
    first = sample()
    second = sample(id='b', parsing_info={'notes':['alternative parser']})
    path.write_text('\n'.join(json.dumps(r) for r in [first, second]), encoding='utf-8')
    output = tmp_path / 'output'
    manifest = prepare(path, spaphish_fixture(tmp_path), output)
    assert manifest['records'] == 3
    assert manifest['exact_evidence_duplicates'] == 1
    assert manifest['eligible_records'] == 0
    assert manifest['approved_for_training'] is False
    assert (output / 'training_corpus.jsonl').read_bytes() == b''
    rows = [json.loads(s) for s in (output / 'review_corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 3 and rows[1]['duplicate_of'] == 'a'
    with pytest.raises(ValueError, match='must not be overwritten'):
        prepare(path, tmp_path / 'spa', output)
