import json
from pathlib import Path
import pytest
from training.domain_evidence import review_record, write_review_queue, reject_legacy_domain_training
from scripts.export_domain_evidence_review import export

@pytest.mark.parametrize('label',[0,1,None,'bad'])
def test_source_labels_never_become_domain_labels(label):
    r=review_record('https://services.example.co.uk/Login?a=A%2FB#step',source='fixture',
                    source_label=label,source_label_scope='url_phishing')
    assert r['domain_label'] is None and r['training_eligible'] is False
    assert r['full_hostname']=='services.example.co.uk'
    assert r['registrable_domain']=='example.co.uk'
    assert r['raw_input']=='https://services.example.co.uk/Login?a=A%2FB#step'
    assert r['observation_date'] is None

def test_dates_and_metadata_remain_source_assertions():
    r=review_record('host.example.com',source='fixture',source_label_scope='unverified_reputation',
                    observation_date='1777636885.0',observation_date_kind='source_reported_last_analysis',
                    metadata={'Creation_Date':'2001','Malicious_Votes':'0'})
    assert r['observation_date']=='1777636885.0'
    assert not r['observation_date_verified']
    assert r['source_metadata']['Creation_Date']=='2001'
    assert r['domain_label'] is None

def test_invalid_input_preserved_for_review():
    r=review_record('http://[broken',source='fixture',source_label_scope='url_phishing')
    assert r['raw_input']=='http://[broken' and r['parsing_error']
    assert not r['training_eligible']

@pytest.mark.parametrize('date',['Unknown','N/A','',None])
def test_unknown_dates_are_null_without_losing_original_metadata(date):
    r=review_record('example.com',source='fixture',source_label_scope='unverified_reputation',
                    observation_date=date,metadata={'Last_Analysis_Date':date})
    assert r['observation_date'] is None and r['observation_date_kind'] is None
    assert r['source_metadata']['Last_Analysis_Date']==date

def test_atomic_queue_keeps_conflicts_and_blocks_admitted_labels(tmp_path):
    records=[review_record('example.com',source='fixture',source_label_scope='url_phishing',source_label=y)
             for y in (0,1)]
    path=tmp_path/'review.jsonl'
    result=write_review_queue(path,records)
    original=path.read_bytes()
    assert result['records']==2
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    assert {r['source_label'] for r in rows}=={0,1}
    bad=dict(records[0],domain_label=1)
    with pytest.raises(ValueError):write_review_queue(path,[records[0],bad])
    assert path.read_bytes()==original
    assert not path.with_suffix('.jsonl.pending').exists()
    with pytest.raises(ValueError):write_review_queue(tmp_path/'clean_domain_corpus.csv',records)

def test_export_keeps_all_rows_dates_and_unknowns(tmp_path):
    url=tmp_path/'urls.csv';rep=tmp_path/'reputation.csv'
    url.write_text('url,label,source\nhttps://login.example.com/Case?q=A%2FB,1,URLFixture\nhttps://login.example.com/Case?q=A%2FB,0,URLFixture\n')
    rep.write_text('Domain,Last_Analysis_Date,Creation_Date,Malicious_Votes\nwww.example.org,1777636885,2001,0\nother.example.org,,2002,9\n')
    originals=(url.read_bytes(),rep.read_bytes())
    output=tmp_path/'review.jsonl'
    report=export(output,url,rep)
    rows=[json.loads(s) for s in output.read_text().splitlines()]
    assert report['records']==4 and all(not r['training_eligible'] for r in rows)
    assert rows[0]['source_row_number']==2 and rows[1]['source_row_number']==3
    assert rows[2]['full_hostname']=='www.example.org'
    assert rows[2]['observation_date']=='1777636885'
    assert rows[3]['observation_date'] is None
    assert rows[3]['source_metadata']['Creation_Date']=='2002'
    assert (url.read_bytes(),rep.read_bytes())==originals

def test_old_trainers_refuse_unverified_domain_data(monkeypatch):
    import training.train_url_and_domain_models as legacy
    import scripts.train_production_models as production
    def unexpected(*a,**k):raise AssertionError('Training or saving occurred before admission guard')
    monkeypatch.setattr(legacy,'load_all_url_data',unexpected)
    monkeypatch.setattr(legacy.joblib,'dump',unexpected)
    monkeypatch.setattr(production.pd,'read_csv',unexpected)
    for call in (reject_legacy_domain_training,legacy.load_all_domain_data,legacy.train_models,
                 production.train_url_models):
        with pytest.raises(ValueError,match='Legacy domain training is disabled'):call()

def test_url_loader_keeps_original_source_text_and_row_links(tmp_path):
    from training.legacy_url_sources import load_url_candidates
    from scripts.clean_and_curate_master_datasets import canonicalize_url_clean
    import hashlib
    (tmp_path/'top-1m.csv').write_text(''.join(f'{i+1},rank{i}.com\n' for i in range(1002)))
    mp=tmp_path/'malicious_phish.csv'
    mp.write_text('url,type\n https://login.example.com/Case?q=A%2FB ,phishing\nhttps://other.example.org/a,benign\n')
    (tmp_path/'PhiUSIIL_Phishing_URL_Dataset.csv').write_text('URL,label\nhttps://phish.example.net/a,0\nhttps://safe.example.net/a,1\n')
    frame=load_url_candidates(canonicalize_url_clean,tmp_path)
    row=frame[frame.source=='MaliciousPhish_Deep'].query('label == 1').iloc[0]
    assert row.original_source_input==' https://login.example.com/Case?q=A%2FB '
    assert row.url=='https://login.example.com/Case?q=A%2FB'
    assert row.source_row_number==2
    assert row.source_snapshot_sha256==hashlib.sha256(mp.read_bytes()).hexdigest()
    top=frame[frame.source=='Tranco_Top50k'].iloc[0]
    assert top.original_source_input=='rank1000.com' and top.source_row_number==1001
