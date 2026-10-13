"""Research protocol tests. Synthetic fixtures do not measure security accuracy."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scripts import run_published_email_experiment as experiment
from utils.email_input_contract import build_email_input, contract_sha256


def record(identifier, source, label, day=None, body=None):
    letters=''.join(chr(97+ord(value)%26) for value in identifier)
    body=body or ' '.join(letters+suffix for suffix in ('alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel'))
    contract=build_email_input(body,sender='editor@example.org',subject='Message')
    return dict(id=identifier,source=source,label=label,task_label='phishing' if label else 'legitimate',
        email_input=contract,input_contract_sha256=contract_sha256(contract),original_reference={'fixture_id':identifier},
        annotation={'observed_day':day,'label_authority_kind':'synthetic_test_fixture'},
        eligible_for_scoring=True,exclusion_reasons=[])


def cohorts():
    spa=[record('s'+str(index)+str(label),'spaphish_v4',label,day)
        for index,day in enumerate(('2023-05-01','2024-05-01','2025-05-01')) for label in (0,1)]
    english=[record('e'+str(index),'phishfuzzer_manual',index%2) for index in range(40)]
    brown=[record('b'+str(label),'brown_phish_bowl',label) for label in (0,1)]
    return spa,english,brown


def test_group_split_is_frozen_and_never_claims_English_time_separation():
    spa,english,brown=cohorts()
    first,audit,_=experiment.partition_sources(copy.deepcopy(spa),copy.deepcopy(english),copy.deepcopy(brown))
    second,_,_=experiment.partition_sources(copy.deepcopy(spa),copy.deepcopy(english),copy.deepcopy(brown))
    assert first==second
    assert not audit
    for pool,parts in first.items():
        groups={split:{r['experiment_template_group'] for r in rows} for split,rows in parts.items()}
        assert not groups['train'] & groups['val']
        assert not groups['train'] & groups['published_day_test']
        assert not groups['val'] & groups['published_day_test']
        if pool==experiment.POOLS[0]:assert all(r['source']=='spaphish_v4' for rows in parts.values() for r in rows)
        assert all(r['source']=='spaphish_v4' for r in parts['published_day_test'])


def test_later_template_and_Brown_evidence_are_retained_and_earlier_matches_excluded():
    spa,english,brown=cohorts()
    long_body=' '.join('word'+chr(97+i//26)+chr(97+i%26) for i in range(60))
    spa += [record('earlier','spaphish_v4',1,'2022-04-01',long_body),
            record('later','spaphish_v4',1,'2025-04-01',long_body+' newword')]
    held=record('brown-match','brown_phish_bowl',0,body=spa[0]['email_input']['body'])
    brown.append(held)
    spa.append(record('second-training-negative','spaphish_v4',0,'2022-01-01'))
    before=copy.deepcopy(held['email_input'])
    pools,audit,pairs=experiment.partition_sources(spa,english,brown)
    ids={r['id'] for rows in pools[experiment.POOLS[0]].values() for r in rows}
    assert 'later' in ids and 'earlier' not in ids and spa[0]['id'] not in ids
    assert held['email_input']==before
    assert any(r['id']=='earlier' and 'later' in r['held_out_ids'] for r in audit)
    assert any(r['id']==spa[0]['id'] and r['held_out_ids']==['brown-match'] for r in audit)
    assert pairs


def test_missing_date_incomplete_and_url_only_rows_are_preserved_in_audit():
    spa,english,brown=cohorts()
    missing=record('missing-day','spaphish_v4',1)
    bad=record('incomplete','spaphish_v4',1,'2022-01-01')
    bad['email_input']=build_email_input('Original partial body',parsing_info={'errors':['documented source truncation']})
    bad['input_contract_sha256']=contract_sha256(bad['email_input'])
    empty=record('only-link','phishfuzzer_manual',1)
    empty['email_input']=build_email_input('',supplementary_urls=['https://example.org/verify'])
    empty['input_contract_sha256']=contract_sha256(empty['email_input'])
    _,audit,_=experiment.partition_sources(spa+[missing,bad],english+[empty],brown)
    assert {r['id'] for r in audit}=={'missing-day','incomplete','only-link'}
    assert {r['reason'] for r in audit}=={'missing_publisher_reception_day','documented_incomplete_extraction',
        'no_observed_message_body_url_or_attachment_only_retained_in_audit'}


def test_complete_evidence_label_conflicts_are_quarantined_without_relabeling():
    spa,english,brown=cohorts()
    a=record('same-evidence-a','spaphish_v4',1,'2022-01-01',body='Conflicting source email evidence body.')
    b=record('same-evidence-b','spaphish_v4',0,'2022-01-01',body='Conflicting source email evidence body.')
    pools,audit,_=experiment.partition_sources(spa+[a,b],english,brown)
    assert {r['id'] for r in audit}=={a['id'],b['id']}
    assert all(r['reason']=='conflicting_source_labels_for_identical_complete_evidence' and r['retained_id'] is None for r in audit)
    assert a['label']==1 and b['label']==0


def test_coverage_denominators_and_unavailable_classes_are_not_hidden():
    rows=[record('one','brown_phish_bowl',0),record('two','brown_phish_bowl',1),record('three','brown_phish_bowl',0)]
    rows[2].update(eligible_for_scoring=False,exclusion_reasons=['unscanned_image'])
    report=experiment.report_metrics(rows,np.array([.9,np.nan,.1]),.95,eligible=[True,True,False])
    assert report['total_rows']==3 and report['quality_excluded_rows']==1
    assert report['eligible_cohort']['at_0_5']['total_rows']==2
    assert report['eligible_cohort']['at_0_5']['unsupported_by_class']=={'0':0,'1':1}
    assert report['all_source_labeled_rows_diagnostic']['at_0_5']['total_rows']==3
    assert report['quality_exclusion_reasons']=={'unscanned_image':1}


def test_paired_comparisons_use_same_rows_and_baseline_support():
    rows=[record(str(i),'brown_phish_bowl',i%2) for i in range(4)]
    rows[3].update(eligible_for_scoring=False,exclusion_reasons=['partial'])
    result=experiment.paired_comparison(rows,np.array([.1,.1,.9,.1]),np.array([np.nan,.9,.1,.9]),.8,.8,
        baseline=np.array([.1,.1,np.nan,.1]))
    assert result['total_source_rows']==4 and result['eligible_rows']==3
    assert result['common_supported_rows']==1
    assert result['before_at_0_5']['total_rows']==result['after_at_0_5']['total_rows']==1
    assert result['decision_changes_at_0_5']['positives_newly_detected']==1
    assert 'roc_auc_difference' not in result


def source_fixture(tmp_path,monkeypatch):
    review=tmp_path/'review.jsonl'
    spa=tmp_path/'spa'
    spa.mkdir()
    spa_original={'hash':'a'*64,'subject':'Original subject','body':'Original Spanish body','urls':'[https://example.org/report]','Label':'1'}
    with (spa/'corpus.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,delimiter=';',fieldnames=list(spa_original))
        writer.writeheader();writer.writerow(spa_original)
    sha=experiment.digest(spa/'corpus.csv')
    annotation=dict(id='spaphish-v4-'+'a'*64,source_row_index=0,task_label='phishing',observed_day='2023-01-01',
                    label_authority_kind='synthetic_source_fixture')
    monkeypatch.setattr(experiment,'load_spaphish_publisher_annotations',lambda _:dict(annotations=[annotation],source_corpus_sha256=sha))
    expected=build_email_input(spa_original['body'],subject=spa_original['subject'],supplementary_urls=['https://example.org/report'])
    spa_row=dict(id=annotation['id'],source='spaphish_v4',source_claimed_label='publisher_phishing',
        original_reference={'source_sha256':sha,'row_index':0},email_input=expected,input_contract_sha256=contract_sha256(expected))
    originals=[]
    for index in range(292):
        originals.append(dict(Body='Original English body '+str(index),Subject='Original subject',Sender='editor@example.org',
            URL=['https://example.org/help'],Type='Spam' if index>=200 else 'Phishing' if index%2 else 'Valid',
            Source='Manual',Language='en',**{'Created by':'Human'}))
    ff=tmp_path/'original.json'
    ff.write_text(json.dumps(originals),encoding='utf-8')
    ff_sha=experiment.digest(ff)
    monkeypatch.setattr(experiment,'FF_RELEASE_HASH',ff_sha)
    rows=[spa_row]
    for index,original in enumerate(originals):
        contract=build_email_input(original['Body'],subject=original['Subject'],sender=original['Sender'],supplementary_urls=original['URL'])
        rows.append(dict(id='ff-'+str(index),source='phishfuzzer_manual',source_claimed_label=original['Type'],
            original_reference={'source_sha256':ff_sha,'row_index':index},email_input=contract,input_contract_sha256=contract_sha256(contract)))
    experiment.save_jsonl(review,rows)
    return review,spa,ff,rows


def test_bound_publisher_and_author_labels_keep_real_authority_without_project_review(tmp_path,monkeypatch):
    review,spa,ff,_=source_fixture(tmp_path,monkeypatch)
    spanish,english,audit,_=experiment.published_records(review,spa,ff)
    assert len(spanish)==1 and len(english)==200 and len(audit)==92
    assert all(row['annotation']['label_authority_kind']=='source_author_manual_label' for row in english)
    assert all(row['project_independent_review_completed'] is False for row in spanish+english)
    assert all(row['annotation']['training_reuse_permission_established'] is False for row in english)


def test_non_English_source_author_rows_are_retained_outside_English_training(tmp_path,monkeypatch):
    review,spa,ff,_=source_fixture(tmp_path,monkeypatch)
    originals=json.loads(ff.read_text(encoding='utf-8'))
    originals[1]['Language']='no'
    ff.write_text(json.dumps(originals),encoding='utf-8')
    changed_hash=experiment.digest(ff)
    monkeypatch.setattr(experiment,'FF_RELEASE_HASH',changed_hash)
    rows=experiment.read_jsonl(review)
    for row in rows[1:]:
        row['original_reference']['source_sha256']=changed_hash
    experiment.save_jsonl(review,rows)
    _,english,audit,_=experiment.published_records(review,spa,ff)
    assert len(english)==199
    assert len(audit)==93
    excluded=[row for row in audit if row['reason'].startswith('non_English')]
    assert excluded[0]['source_claimed_language']=='no'
    assert excluded[0]['source_claimed_label']=='Phishing'
    assert excluded[0]['id']=='ff-1'


@pytest.mark.parametrize('source,field',[('spa','body'),('spa','sender'),('english','body'),('english','supplementary_urls')])
def test_self_consistent_tampered_contract_is_rejected_against_original_source(tmp_path,monkeypatch,source,field):
    review,spa,ff,rows=source_fixture(tmp_path,monkeypatch)
    row=rows[0 if source=='spa' else 1]
    raw={key:row['email_input'][key] for key in ('subject','sender','headers','html_body','supplementary_urls','parsing_info')}
    body=row['email_input']['body']
    if field=='body':body='Invented email content'
    elif field=='sender':raw['sender']='invented@different.example'
    else:raw['supplementary_urls']=['https://invented.example/']
    row['email_input']=build_email_input(body,**raw)
    row['input_contract_sha256']=contract_sha256(row['email_input'])
    experiment.save_jsonl(review,rows)
    with pytest.raises(ValueError,match='raw fields differ'):
        experiment.published_records(review,spa,ff)


def test_existing_or_external_output_and_stale_prepared_run_cannot_fit(tmp_path,monkeypatch):
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    monkeypatch.setattr(experiment,'email_evidence_estimator',lambda **kwargs:pytest.fail('Refused run reached fitting'))
    with pytest.raises(ValueError,match='benchmarks child'):
        experiment.execute_prepared(tmp_path/'outside')
    out=tmp_path/'benchmarks'/'already_finished'
    out.mkdir(parents=True)
    experiment.save(out/'run_status.json',{'stage':'completed'})
    before=(out/'run_status.json').read_bytes()
    with pytest.raises(ValueError,match='pristine prepared'):
        experiment.execute_prepared(out)
    assert (out/'run_status.json').read_bytes()==before


def test_per_source_false_positive_rates_are_reported_at_one_frozen_threshold():
    rows=[record('spa-neg','spaphish_v4',0),record('spa-pos','spaphish_v4',1),
          record('ff-neg','phishfuzzer_manual',0),record('ff-pos','phishfuzzer_manual',1)]
    result=experiment.report_metrics(rows,np.array([.1,.9,.9,.9]),.8)
    assert result['per_source']['spaphish_v4']['at_frozen_validation_budget']['fpr']==0.
    assert result['per_source']['phishfuzzer_manual']['at_frozen_validation_budget']['fpr']==1.
    assert {r['at_frozen_validation_budget']['threshold'] for r in result['per_source'].values()}=={.8}


def test_train_only_source_class_weights_remove_source_label_prevalence_confounding():
    rows=[record('spa-'+str(i),'spaphish_v4',int(i==0)) for i in range(10)]
    rows += [record('eng-'+str(i),'phishfuzzer_manual',i%2) for i in range(4)]
    weights=experiment.source_class_weights(rows)
    assert weights.sum()==pytest.approx(len(rows))
    totals={}
    for row,weight in zip(rows,weights):
        key=(row['source'],row['label'])
        totals[key]=totals.get(key,0)+weight
    assert set(totals)=={('spaphish_v4',0),('spaphish_v4',1),('phishfuzzer_manual',0),('phishfuzzer_manual',1)}
    assert all(total==pytest.approx(len(rows)/4) for total in totals.values())
    assert all(r['email_input']['sender']=='editor@example.org' for r in rows)
    with pytest.raises(ValueError,match='both labels'):
        experiment.source_class_weights([record('unmatched','spaphish_v4',0)])


def test_neutral_footer_keeps_original_identity_and_destinations_and_does_not_become_a_label():
    rows=[record('footer','brown_phish_bowl',0)]
    original=copy.deepcopy(rows[0]['email_input'])
    changed=experiment.neutral_footer_inputs(rows)[0]
    assert rows[0]['email_input']==original
    assert changed['body']==original['body']+'\n\nArchive reference: Enron 2001.'
    assert changed['sender']==original['sender']
    assert changed['link_destinations']==original['link_destinations']
    assert rows[0]['label']==0
    result=experiment.footer_report(rows,np.array([.1]),np.array([.9]),.8)
    assert result['decisions_changed_at_0_5']==result['decisions_changed_at_frozen_budget']==1
    assert result['mean_absolute_probability_change']==pytest.approx(.8)
    assert 'accuracy' not in result
