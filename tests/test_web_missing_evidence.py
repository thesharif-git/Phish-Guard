import numpy as np
import pytest
from services.origintrace_pillars import ProvenanceMatrix,InfrastructureLineage,SemanticDeception,EvidentialFusion
from services.web_evidence_policy import web_evidence_decision
from utils import url_scanner as web
from services import scan_engine

def test_missing_web_modalities_are_vacuous_not_clean():
    url='https://neutral-fixture.invalid/'
    p1=ProvenanceMatrix.analyze_web('neutral-fixture.invalid',url,tls_intel={})
    p2=InfrastructureLineage.analyze_web('neutral-fixture.invalid',dns_ephemerality={},asn_intel={},geo={})
    p3=SemanticDeception.analyze_web({'success':False},url,'neutral-fixture.invalid')
    fused=EvidentialFusion.synthesize(p1,p2,p3,context='web')
    assert fused['belief_clean']==0 and fused['belief_phish']==0
    assert fused['uncertainty']==1
    empty=EvidentialFusion.synthesize({}, {}, {}, context='web')
    assert empty['uncertainty']==1 and empty['belief_clean']==0

def test_partial_check_preserves_positive_evidence_without_clean_mass():
    partial={'threat_prob':.8,'confidence':.75,'evidence_complete':False}
    fused=EvidentialFusion.synthesize(partial,{}, {},context='web')
    assert fused['belief_phish']==pytest.approx(.6)
    assert fused['belief_clean']==0
    assert fused['uncertainty']==pytest.approx(.4)

@pytest.mark.parametrize('score', [0,10,29,35,44])
def test_low_scores_with_missing_core_checks_never_claim_clean(score):
    decision=web_evidence_decision(score,url_probability=None,html={},tls={},dns={},asn={},intel={},contradiction={},ignorance=1)
    assert decision['status']=='UNKNOWN' and decision['verdict_state']=='INSUFFICIENT_EVIDENCE'
    assert decision['reputation_checked'] is False

def fixture_scanner(monkeypatch,*,model=True,page=False,intel=None,authority=False):
    monkeypatch.setattr(scan_engine,'MODEL_LOADED',True)
    monkeypatch.setattr(scan_engine,'DOMAIN_MODEL',None)
    class Model:
        feature_names_in_=np.array(['fixture'])
        def predict_proba(self,frame):return np.array([[.95,.05]])
    monkeypatch.setattr(scan_engine,'URL_MODEL',Model() if model else None)
    monkeypatch.setattr(scan_engine,'extract_url_features',lambda url:{'fixture':1})
    monkeypatch.setattr(scan_engine,'resolve_short_url',lambda url:url)
    monkeypatch.setattr('data.store.is_blocked',lambda *a:False)
    monkeypatch.setattr(web,'full_threat_check',lambda *a:intel or {})
    monkeypatch.setattr(web,'analyze_live_html',lambda *a:{'success':page,'error':None if page else 'Fixture retrieval failed'})
    monkeypatch.setattr(web,'geolocate_ip',lambda *a:{'available':False})
    monkeypatch.setattr(web,'analyze_tls_cert_lineage',lambda *a:{})
    monkeypatch.setattr(web,'calculate_dns_ephemerality',lambda *a:{})
    monkeypatch.setattr(web,'check_asn_brand_incongruity',lambda *a:{})
    monkeypatch.setattr(web,'is_safe_authority_domain',lambda *a:authority)

@pytest.mark.parametrize('model,page',[(True,False),(False,True),(False,False)])
def test_actual_url_scanner_abstains_on_unavailable_core_checks(monkeypatch,model,page):
    fixture_scanner(monkeypatch,model=model,page=page)
    result=web.scan_url('https://neutral-fixture.invalid/')
    assert result['status']=='UNKNOWN' and result['verdict_state']=='INSUFFICIENT_EVIDENCE'
    assert result['epistemic_uncertainty'] is None
    assert result['reputation_checked'] is False
    assert result['check_availability']['page']==page
    assert result['check_availability']['url_model']==model

@pytest.mark.parametrize('feed,payload',[('phishtank',{'available':True,'is_phishing':True}),('urlhaus',{'available':True,'is_malicious':True}),('virustotal',{'available':True,'malicious':1})])
def test_confirmed_feed_is_not_erased_on_authority_root_or_missing_page(monkeypatch,feed,payload):
    fixture_scanner(monkeypatch,model=False,page=False,intel={feed:payload},authority=True)
    result=web.scan_url('https://neutral-fixture.invalid/')
    assert result['status']==result['verdict_state']=='PHISHING'
    assert result['final_score']>=90
    assert result['evidence_incomplete'] is True and result['reputation_checked'] is True

def test_completed_page_and_model_report_only_completed_check_findings(monkeypatch):
    fixture_scanner(monkeypatch,model=True,page=True)
    result=web.scan_url('https://neutral-fixture.invalid/')
    assert result['status']==result['verdict_state']=='CLEAN'
    assert result['check_availability']['tls'] is False
    assert result['evidence_incomplete'] is True
    assert 'not proof of safety' in result['summary']

def test_status_and_verdict_agree_at_previous_conflicting_boundary():
    result=web_evidence_decision(65,url_probability=.65,html={'success':True},tls={},dns={},asn={},intel={},contradiction={},ignorance=.1)
    assert result['status']==result['verdict_state']=='SUSPICIOUS'
