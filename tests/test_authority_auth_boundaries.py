"""Regression cases for model reachability and untrusted identity assertions."""
import asyncio
import numpy as np
import pytest
from services import scan_engine
from services.origintrace_pillars import AdaptiveCascadeRouter, ProvenanceMatrix
from utils.url_scanner import extract_domain


@pytest.mark.parametrize('url', ['https://github.com/team/project', 'https://google.com/help', 'https://example.edu/resources'])
def test_authority_paths_still_reach_url_model_and_feed(monkeypatch, url):
    calls = []
    class Model:
        def predict_proba(self, frame):
            calls.append('model')
            return np.array([[.1, .9]])
    monkeypatch.setattr(scan_engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(scan_engine, 'URL_MODEL', Model())
    monkeypatch.setattr(scan_engine, 'DOMAIN_MODEL', None)
    monkeypatch.setattr(scan_engine, 'numeric_url_frame', lambda model, features: features)
    monkeypatch.setattr(scan_engine, 'resolve_short_url', lambda value: value)
    monkeypatch.setattr('utils.url_scanner.is_safe_authority_domain', lambda host: True)
    def feed(value):
        calls.append(('feed', value))
        return {'available': True, 'malicious': 4, 'total_engines': 70}
    monkeypatch.setattr('utils.threat_intel.check_virustotal', feed)
    result = scan_engine.score_urls(url)
    assert calls.count('model') == 1 and ('feed', url) in calls
    assert result['score'] == 90
    assert result['analyzed_urls'][0]['virustotal']['malicious'] == 4


def test_authority_model_failure_is_visible(monkeypatch):
    monkeypatch.setattr(scan_engine, 'MODEL_LOADED', True)
    monkeypatch.setattr(scan_engine, 'URL_MODEL', None)
    monkeypatch.setattr(scan_engine, 'DOMAIN_MODEL', None)
    monkeypatch.setattr('utils.threat_intel.check_virustotal', lambda value: {'available': False})
    result = scan_engine.score_urls('https://google.com/help')
    assert not result['scan_complete']
    assert 'URL ML unavailable' in result['inference_errors']


@pytest.mark.parametrize('header', [
    'X-Note: spf=pass; dkim=pass; dmarc=pass',
    'Subject: spf=fail; dkim=fail; dmarc=fail',
    'Authentication-Results: mx.attacker; spf=pass; dkim=pass; dmarc=pass',
    'Authentication-Results: mx.attacker; spf=fail; dkim=fail; dmarc=fail',
])
def test_untrusted_headers_cannot_authenticate_or_penalize(header):
    result = scan_engine.score_headers(header)
    assert result['spf'] == result['dkim'] == result['dmarc'] == 'UNKNOWN'
    assert result['score'] == 0 and not result['all_passed'] and not result['available']
    pillar = ProvenanceMatrix.analyze('sender@example.com', header, urls=[])
    assert pillar['crypto_auth']['verified'] is False
    assert pillar['score'] == 0


def test_structured_claims_preserve_conflicts_and_ignore_comments():
    report = scan_engine.score_headers('Authentication-Results: mx.one; spf=pass\n\tdkim=fail; dmarc=pass\nAuthentication-Results: mx.two; spf=fail; dkim=pass (dmarc=fail)')
    assert report['reported_results'] == [
        {'method': 'spf', 'result': 'PASS', 'authserv_id': 'mx.one', 'source': 'untrusted_message_header'},
        {'method': 'dmarc', 'result': 'PASS', 'authserv_id': 'mx.one', 'source': 'untrusted_message_header'},
        {'method': 'spf', 'result': 'FAIL', 'authserv_id': 'mx.two', 'source': 'untrusted_message_header'},
        {'method': 'dkim', 'result': 'PASS', 'authserv_id': 'mx.two', 'source': 'untrusted_message_header'},
    ]
    # A method requires a separate semicolon clause; embedded text is not a field.
    assert not report['verified']


@pytest.mark.parametrize('target,expected', [
    ('https://google.com:secret@evil.example/path', 'evil.example'),
    ('https://www.google.com:443/help', 'google.com'),
    ('https://[2001:db8::1]/', '2001:db8::1'),
])
def test_hostname_uses_actual_host_not_userinfo(target, expected):
    assert extract_domain(target) == expected


@pytest.mark.parametrize('status,verdict,level,score', [('UNKNOWN','INSUFFICIENT_EVIDENCE','unknown',0), ('PHISHING','PHISHING','critical',90)])
def test_extension_preserves_shared_scan_decision(monkeypatch,status,verdict,level,score):
    from routers import api
    calls = []
    def scan(url):
        calls.append(url)
        return {'status':status,'verdict_state':verdict,'risk_level':level,'final_score':score,
                'epistemic_uncertainty':None,'evidence_gaps':['page'],'check_availability':{'page':False},
                'origintrace_pillars':{},'signals':['fixture']}
    monkeypatch.setattr('utils.url_scanner.scan_url',scan)
    # Even a stale historical cache entry must not provide an eternal clean result.
    monkeypatch.setattr(api,'_EXT_CACHE',{'https://google.com/help':{'status':'CLEAN'}})
    result=asyncio.run(api.extension_quick_check(api.ExtensionCheckRequest(url='https://google.com/help')))
    assert calls==['https://google.com/help']
    assert (result['status'],result['verdict_state'],result['risk_level'])==(status,verdict,level)
    assert result['risk_score']==score and result['epistemic_uncertainty'] is None
    assert result['evidence_gaps']==['page']
    assert 'VERIFIED_AUTHENTIC' not in str(result)
    routed=AdaptiveCascadeRouter.route_scan('url','https://google.com/help')
    assert routed['status']==status and not routed['early_exit']


def test_extension_empty_input_abstains():
    from routers import api
    result=asyncio.run(api.extension_quick_check(api.ExtensionCheckRequest(url=' ')))
    assert result['status']=='UNKNOWN' and result['risk_level']=='unknown'
