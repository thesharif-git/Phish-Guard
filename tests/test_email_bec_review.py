"""Semantic evidence must not be diluted by missing keywords or called calibrated."""
import sys
import types
import pytest
from services import bec_detector, scan_engine


@pytest.mark.parametrize('score', [.65, .85, .95])
def test_keyword_free_semantic_signal_is_not_halved(monkeypatch, score):
    monkeypatch.setattr(bec_detector, '_get_bec_classifier', lambda: lambda *a:
        {'labels':['fraudulent wire transfer request'], 'scores':[score]})
    result = bec_detector.predict_bec_intent('Please arrange the remittance to the revised beneficiary by noon.')
    assert result['rule_score'] == 0
    assert result['bec_score'] == round(score * 100) and result['is_bec']
    assert result['transformer_score'] == score and result['transformer_available']
    assert result['confidence'] is None and result['score_kind'] == 'unvalidated_review_score'


def test_semantic_clean_result_does_not_erase_rule_signal(monkeypatch):
    monkeypatch.setattr(bec_detector, '_get_bec_classifier', lambda: lambda *a:
        {'labels':['legitimate business email'], 'scores':[.98]})
    result = bec_detector.predict_bec_intent('Urgent wire transfer: transfer funds to bank details. Do not call.')
    assert result['transformer_available'] and result['transformer_score'] is None
    assert result['bec_score'] == result['rule_score'] and result['is_bec']


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 1.1])
def test_invalid_semantic_scores_are_visible_and_not_used(monkeypatch, score):
    monkeypatch.setattr(bec_detector, '_get_bec_classifier', lambda: lambda *a:
        {'labels':['fraudulent wire transfer request'], 'scores':[score]})
    result = bec_detector.predict_bec_intent('Please arrange the remittance to the revised beneficiary by noon.')
    assert not result['transformer_available'] and not result['is_bec']
    assert result['transformer_error'] and result['transformer_score'] is None


def test_missing_transformer_is_reported_honestly(monkeypatch):
    monkeypatch.setattr(bec_detector, '_get_bec_classifier', lambda: None)
    result = bec_detector.predict_bec_intent('The schedule for the meeting is attached for your review.')
    assert not result['transformer_available'] and result['confidence'] is None


def test_real_email_evaluator_routes_semantic_signal_to_review_without_inflating_probability(monkeypatch):
    monkeypatch.setattr(bec_detector, '_get_bec_classifier', lambda: lambda *a:
        {'labels':['fraudulent wire transfer request'], 'scores':[.85]})
    monkeypatch.setattr(scan_engine, 'score_content', lambda text: {'ml_powered':True,'ml_probability':.02,'score':2})
    monkeypatch.setattr(scan_engine, 'score_urls', lambda text: {'score':0,'urls_found':0,'inference_errors':[]})
    vector=types.ModuleType('services.vector_store')
    vector.search_similar_campaigns=lambda text: {'has_match':False}
    vector.index_phishing_email=lambda *a,**kw: None
    monkeypatch.setitem(sys.modules,'services.vector_store',vector)
    class Predictor:
        is_calibrated=True
        pipeline_id=scan_engine.security_pipeline_id()
        def predict_set(self,*a,**kw): return {'CLEAN'}, 'CLEAN', 'SAFE_DELIVERY'
    result=scan_engine.evaluate_email_security('Please arrange the remittance to the revised beneficiary by noon.',predictor=Predictor())
    assert result['bec_result']['is_bec']
    assert result['security_action']=='WARNING_BANNER'
    assert result['conformal_res']['threat_probability']==.02
    assert result['conformal_res']['decision_basis']=='separate_evidence_review'


def test_transformer_loader_uses_offline_pretrained_loading(monkeypatch):
    calls=[]
    class Loader:
        @staticmethod
        def from_pretrained(name,**kwargs):
            calls.append(kwargs)
            return object()
    fake=types.ModuleType('transformers')
    fake.AutoTokenizer=fake.AutoModelForSequenceClassification=Loader
    def pipeline(task,**kwargs):
        assert 'local_files_only' not in kwargs
        assert 'tokenizer' in kwargs and 'model' in kwargs
        return 'fixture'
    fake.pipeline=pipeline
    monkeypatch.setitem(sys.modules,'transformers',fake)
    monkeypatch.setattr(bec_detector,'_PIPELINE_LOADED',False)
    monkeypatch.setattr(bec_detector,'_BEC_PIPELINE',None)
    assert bec_detector._get_bec_classifier()=='fixture'
    assert calls==[{'local_files_only':True}]*2


def test_quarantine_recommendation_does_not_claim_mailbox_action(monkeypatch):
    from services import conformal_evidential_engine as fusion
    monkeypatch.setattr(bec_detector,'_get_bec_classifier',lambda:None)
    monkeypatch.setattr(scan_engine,'score_content',lambda text:{'ml_powered':True,'ml_probability':.99,'score':99})
    monkeypatch.setattr(scan_engine,'score_urls',lambda text:{'score':0,'urls_found':0,'inference_errors':[]})
    vector=types.ModuleType('services.vector_store')
    vector.search_similar_campaigns=lambda text:{'has_match':False}
    indexed=[]
    vector.index_phishing_email=lambda *a,**kw:indexed.append(a)
    monkeypatch.setitem(sys.modules,'services.vector_store',vector)
    monkeypatch.setattr('data.store.is_blocked',lambda *a:False)
    monkeypatch.setattr('data.store.add_scan',lambda *a,**kw:None)
    class Predictor:
        is_calibrated=True
        pipeline_id=scan_engine.security_pipeline_id()
        def predict_set(self,*a,**kw):return {'PHISHING'},'PHISHING','AUTO_QUARANTINE'
    monkeypatch.setattr(fusion,'CONFORMAL_PREDICTOR',Predictor())
    result=scan_engine.scan_email('Please review the meeting agenda for tomorrow.',session=object())
    assert result['security_action']=='AUTO_QUARANTINE'
    assert result['action_execution']=={'executed':False,'scope':'scan_recommendation_only'}
    assert 'has not moved or blocked' in result['ai_explanation']['recommended_action']
    assert not indexed and not result['campaign_memory_write']['executed']


def test_manual_sender_block_does_not_fabricate_model_or_auth_evidence(monkeypatch):
    monkeypatch.setattr('data.store.is_blocked',lambda *a:True)
    saved=[]
    monkeypatch.setattr('data.store.add_scan',lambda *a,**kw:saved.append(kw))
    monkeypatch.setattr(scan_engine,'score_content',lambda *a:pytest.fail('Blocked sender must not enter statistical scoring'))
    result=scan_engine.scan_email('Ordinary email body',session=object(),sender='sender@example.com')
    assert result['decision_basis']=='manual_sender_block'
    assert result['security_action']=='BLOCK_BY_POLICY'
    assert result['statistical_verdict'] is None
    assert result['content_analysis']['ml_probability'] is None
    assert not result['content_analysis']['ml_powered']
    assert result['header_authentication']['spf']=='UNKNOWN'
    assert not result['action_execution']['executed']
    assert len(saved)==1
