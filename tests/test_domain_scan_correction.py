import asyncio
import copy
import json
import textwrap
import types
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import pytest
from services.domain_scan_policy import domain_scan_input,scope_domain_result

@pytest.mark.parametrize('value,host,url',[
    ('Login.Example.CO.UK','login.example.co.uk','https://login.example.co.uk'),
    ('http://login.example.com/','login.example.com','http://login.example.com'),
    ('https://www.example.com','www.example.com','https://www.example.com')])
def test_full_host_preserved(value,host,url):
    assert domain_scan_input(value)==(host,url)

@pytest.mark.parametrize('value',['https://example.com/Login','example.com?a=1',
    'https://user:pass@example.com','example.com:443','http://[broken','example.com#x',
    'example .com','javascript://example.com','localhost'])
def test_domain_route_rejects_url_payload_or_invalid_host(value):
    with pytest.raises((ValueError,UnicodeError)):domain_scan_input(value)

def result():
    return dict(url='https://login.example.com',final_score=4,status='CLEAN',verdict_state='CLEAN',
                html_analysis={'success':True},origintrace_pillars={'model_probabilities':{'url':.02,'domain':.99}})

@pytest.mark.parametrize('missing',['page','model'])
def test_missing_evidence_is_not_clean(missing):
    r=result()
    if missing=='page':r['html_analysis']['success']=False
    else:r['origintrace_pillars']['model_probabilities']['url']=None
    old=copy.deepcopy(r)
    out=scope_domain_result(r,'login.example.com')
    assert out['status']=='UNKNOWN' and out['verdict_state']=='INSUFFICIENT_EVIDENCE'
    assert out['final_score']==4 and out['risk_level']=='unknown'
    assert out['evidence_gaps'] and r==old

@pytest.mark.parametrize('status',['PHISHING','SUSPICIOUS'])
def test_missing_evidence_preserves_threat_or_review(status):
    r=result();r.update(status=status,final_score=91 if status=='PHISHING' else 55)
    r['html_analysis']['success']=False
    out=scope_domain_result(r,'login.example.com')
    assert out['status']==status and out['final_score']==r['final_score']

def test_completed_scan_has_explicit_limited_scope():
    out=scope_domain_result(result(),'login.example.com')
    assert out['status']=='CLEAN'
    assert out['scan_scope']['whole_domain_ownership_verified'] is False
    assert out['scan_scope']['submitted_hostname']=='login.example.com'
    assert 'Other paths and subdomains were not assessed' in out['summary']

def test_legacy_domain_cannot_raise_email_url_score(monkeypatch):
    import services.scan_engine as engine
    import utils.threat_intel as intel
    import utils.url_scanner as web
    class Model:
        feature_names_in_=np.array(['a'])
        def __init__(self,p):self.p=p
        def predict_proba(self,frame):return np.array([[1-self.p,self.p]])
    monkeypatch.setattr(engine,'MODEL_LOADED',True)
    monkeypatch.setattr(engine,'URL_MODEL',Model(.01))
    monkeypatch.setattr(engine,'DOMAIN_MODEL',Model(.99))
    monkeypatch.setattr(engine,'extract_url_features',lambda u:{'a':1})
    monkeypatch.setattr(engine,'extract_domain_features',lambda u:{'a':1})
    monkeypatch.setattr(engine,'resolve_short_url',lambda u:u)
    monkeypatch.setattr(intel,'check_virustotal',lambda u:{'available':False})
    monkeypatch.setattr(web,'is_safe_authority_domain',lambda d:False)
    with_domain=engine.score_urls('See https://quiet.example.com/plain')
    monkeypatch.setattr(engine,'DOMAIN_MODEL',None)
    without=engine.score_urls('See https://quiet.example.com/plain')
    assert with_domain['score']==without['score']
    assert with_domain['flagged_urls']==without['flagged_urls']
    assert with_domain['analyzed_urls'][0]['domain_model_diagnostic']==.99
    assert not any('AI Domain' in s for s in with_domain['analyzed_urls'][0]['flags'])

def test_legacy_domain_cannot_add_web_threat_flag():
    from services.security_pipeline import numeric_url_frame,domain_model_input
    source=Path('utils/url_scanner.py').read_text(encoding='utf-8')
    block=source[source.index('    # 4. ML Predictions'):source.index('    # 5. Parallel Multi-Modal')]
    class Model:
        feature_names_in_=np.array(['a'])
        def __init__(self,p):self.p=p
        def predict_proba(self,frame):return np.array([[1-self.p,self.p]])
    scan=types.SimpleNamespace(URL_MODEL=Model(.01),DOMAIN_MODEL=Model(.99),
         extract_url_features=lambda u:{'a':1},extract_domain_features=lambda u:{'a':1},
         numeric_url_frame=numeric_url_frame,domain_model_input=domain_model_input)
    ns=dict(scan_engine=scan,url='https://login.example.com',urlparse=urlparse,
            logger=types.SimpleNamespace(error=lambda *a:None))
    exec(textwrap.dedent(block),ns)
    assert ns['ml_url_score']==0 and ns['domain_model_probability']==.99
    assert ns['ai_flags']==[]

def test_domain_api_preserves_host_and_applies_scope(monkeypatch):
    import routers.api as api
    seen=[]
    monkeypatch.setattr(api,'require_scan_allowance',lambda *a:None)
    monkeypatch.setattr(api,'run_url_scan',lambda u:seen.append(u) or dict(result(),html_analysis={'success':False}))
    monkeypatch.setattr(api.store,'add_scan',lambda *a,**k:None)
    response=asyncio.run(api.scan_domain_endpoint(api.ScanDomainRequest(domain='Login.Example.com'),session=None,user=None))
    body=json.loads(response.body)
    assert seen==['https://login.example.com']
    assert body['status']=='UNKNOWN'
    assert body['scan_scope']['submitted_hostname']=='login.example.com'

def test_domain_http_route_rejects_payload_before_scan(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routers.api as api
    app=FastAPI();app.include_router(api.router)
    app.dependency_overrides[api.get_session]=lambda:None
    app.dependency_overrides[api.get_optional_current_user]=lambda:None
    def unexpected(*a,**k):raise AssertionError('Rejected input was scanned or charged')
    monkeypatch.setattr(api,'run_url_scan',unexpected)
    monkeypatch.setattr(api,'require_scan_allowance',unexpected)
    with TestClient(app) as client:
        response=client.post('/api/scan-domain',json={'domain':'https://example.com/Login?token=secret'})
    assert response.status_code==400
    assert 'URL scanner' in response.json()['detail']

def test_uncertain_existing_verdict_does_not_get_clean_summary():
    r=result();r['verdict_state']='INSUFFICIENT_EVIDENCE'
    out=scope_domain_result(r,'login.example.com')
    assert out['status']=='UNKNOWN' and 'uncertain' in out['summary']

def test_high_risk_does_not_claim_confirmed_domain_ownership():
    r=result();r.update(status='PHISHING',verdict_state='PHISHING',final_score=90,
                       summary='Active fraudulent infrastructure confirmed.')
    out=scope_domain_result(r,'login.example.com')
    assert out['final_score']==90 and out['status']=='PHISHING'
    assert 'does not establish whole-domain ownership' in out['summary']
