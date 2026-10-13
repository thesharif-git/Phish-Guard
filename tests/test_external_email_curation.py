"""Candidate cleaning excludes exact repeats but preserves flagged rare messages."""
import hashlib
import json
import mailbox
from email.message import EmailMessage
from pathlib import Path
import pandas as pd
from scripts import curate_external_email_candidates as curation
from utils.text_sanitizer import clean_and_mask_email_text


def test_transfer_decoding_defects_are_not_missed_before_payload_decoding():
    from email import message_from_bytes
    message = message_from_bytes(b'Content-Type: text/plain\nContent-Transfer-Encoding: base64\n\nVGVzdA==!!!')
    assert not message.defects
    assert 'InvalidBase64CharactersDefect' in curation.decoded_mime_defects(message)


def test_candidate_inventory_preserves_raw_sources_and_does_not_drop_outliers(tmp_path, monkeypatch):
    original_root = curation.ROOT
    source_dir=tmp_path/'data/external_email_validation/nazario_2024_2025'
    source_dir.mkdir(parents=True)
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/clean_and_curate_master_datasets.py').write_bytes(
        (original_root/'scripts/clean_and_curate_master_datasets.py').read_bytes())
    old='Existing corpus text for contract checking.'
    validation_text='Unrelated validation content not used for quality thresholds. ' * 1000
    pd.DataFrame([{'raw_text':old,'masked_text':clean_and_mask_email_text(old),'split':'train'},
                  {'raw_text':validation_text,'masked_text':clean_and_mask_email_text(validation_text),'split':'val'}]).to_csv(
        tmp_path/'data/clean_email_corpus.csv',index=False)
    unusual='A rare full message with unusually long content. ' * 150
    for year, bodies in [(2024,[old,unusual,unusual]),(2025,[unusual])]:
        box=mailbox.mbox(source_dir/f'phishing-{year}.mbox',create=True)
        for body in bodies:
            message=EmailMessage()
            message['From']='fixture@example.com'
            message['Date']=f'01 Jan {year} 12:00:00 +0000'
            message.set_content(body)
            box.add(message)
        box.flush();box.close()
    snapshots=[{'path':p.relative_to(tmp_path).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in source_dir.glob('*.mbox')]
    (source_dir/'manifest.json').write_text(json.dumps({'complete':True,'files':snapshots}))
    monkeypatch.setattr(curation,'ROOT',tmp_path)
    monkeypatch.setattr(curation,'OUT',source_dir)
    curation.main()
    records=[json.loads(line) for line in (source_dir/'candidate_inventory.jsonl').read_text().splitlines()]
    assert len(records)==4
    assert 'exact_body_or_model_input_overlap_with_existing_corpus' in records[0]['exclusion_reasons']
    assert records[1]['text_candidate_eligible']
    assert 'size_outlier_vs_legacy_training_distribution' in records[1]['review_flags']
    assert not records[2]['text_candidate_eligible']
    assert 'exact_message_duplicate' in records[2]['exclusion_reasons']
    assert 'exact_model_input_contract_duplicate' in records[3]['exclusion_reasons']
    assert all(r['independent_security_label'] is None and not r['approved_for_training'] for r in records)
    selected=[json.loads(line) for line in (source_dir/'deduplicated_text_candidates.jsonl').read_text().splitlines()]
    assert len(selected)==1 and selected[0]['text_candidate_eligible']
    assert 'size_outlier_vs_legacy_training_distribution' in selected[0]['review_flags']
    assert all(hashlib.sha256((tmp_path/item['path']).read_bytes()).hexdigest()==item['sha256'] for item in snapshots)
    assert not json.loads((source_dir/'curation_results.json').read_text())['model_predictions_computed']
