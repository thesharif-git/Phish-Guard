import asyncio
import base64
import io
import sys
import types
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import pytest
from fastapi import BackgroundTasks, UploadFile
from services import scan_engine
from services import inbox_connector as inbox
from routers import api
from utils.email_parser import parse_eml_content, parse_pasted_email, extract_email_anchor_links

TARGET = "https://unrelated.example/collect?a=1&b=2"
HTML = '<p>Your document is ready.</p><a href="https://unrelated.example/collect?a=1&amp;b=2">https://paypal.com</a>'

def message(plain=True):
    msg = EmailMessage()
    msg['From'] = 'sender@example.com'
    msg['Subject'] = 'Document available'
    if plain:
        msg.set_content('Your document is ready. Open document.')
    msg.add_alternative(HTML, subtype='html')
    return msg

@pytest.mark.parametrize('plain', [True, False])
def test_eml_and_pasted_rfc_preserve_html_independently(plain):
    msg = message(plain)
    file_parsed = parse_eml_content(msg.as_bytes())
    pasted = parse_pasted_email(msg.as_string())
    for parsed in (file_parsed, pasted):
        assert parsed['html_body'].strip() == HTML
        assert extract_email_anchor_links(parsed['html_body']) == [{'text':'https://paypal.com','href':TARGET}]
        if plain:
            assert parsed['body'] == 'Your document is ready. Open document.'

def test_anchor_evidence_deduplicates_and_rejects_non_web_or_malformed_links():
    links = extract_email_anchor_links(HTML + HTML + '<a href="mailto:a@example.com">Mail</a><a href="javascript:alert(1)">Click</a><a href="/relative">Relative</a><a href="https://[">Bad</a>')
    assert links == [{'text':'https://paypal.com','href':TARGET}]
    assert extract_email_anchor_links('<a href="HTTPS://example.com/Case">Help</a>')[0]['href'] == 'https://example.com/Case'

def test_gmail_keeps_html_after_plain_alternative_and_excludes_text_attachments():
    encode=lambda text:base64.urlsafe_b64encode(text.encode()).decode()
    payload={'mimeType':'multipart/alternative','parts':[
        {'mimeType':'text/plain','body':{'data':encode('Your document is ready. Open document.')}},
        {'mimeType':'text/html','body':{'data':encode(HTML)}},
        {'mimeType':'text/html','filename':'attachment.html','body':{'data':encode('<a href="https://ignored.example">Attached</a>')}}]}
    assert inbox._extract_body_from_payload(payload) == 'Your document is ready. Open document.'
    assert inbox._extract_html_from_payload(payload) == HTML

@pytest.mark.parametrize('via_file',[True,False,'uppercase_file'])
def test_actual_email_endpoint_forwards_hidden_link_evidence(monkeypatch,via_file):
    calls=[]
    monkeypatch.setattr(api,'require_scan_allowance',lambda *a:None)
    monkeypatch.setattr(api,'run_scan',lambda **kw:calls.append(kw) or {'status':'CLEAN'})
    monkeypatch.setattr(api.store,'add_scan',lambda *a,**kw:None)
    msg=message()
    upload=UploadFile(filename='message.EML' if via_file == 'uppercase_file' else 'message.eml',file=io.BytesIO(msg.as_bytes())) if via_file else None
    asyncio.run(api.scan_email_endpoint(background_tasks=BackgroundTasks(),file=upload,content=None if via_file else msg.as_string(),sender='',subject='',headers='',session=object(),user=types.SimpleNamespace(id=1,browser_alerts=False)))
    assert len(calls)==1
    assert calls[0]['content']=='Your document is ready. Open document.'
    assert extract_email_anchor_links(calls[0]['html_body'])[0]['href']==TARGET
    assert calls[0]['parsing_info']['text_complete']
    assert calls[0]['user_id']==1


def test_endpoint_does_not_save_the_same_scan_twice(monkeypatch):
    saved=[]
    monkeypatch.setattr(api,'require_scan_allowance',lambda *a:None)
    def scanner(**kw):
        saved.append(kw['user_id'])
        return {'status':'CLEAN'}
    monkeypatch.setattr(api,'run_scan',scanner)
    monkeypatch.setattr(api.store,'add_scan',lambda *a,**kw:pytest.fail('Endpoint attempted duplicate persistence'))
    asyncio.run(api.scan_email_endpoint(background_tasks=BackgroundTasks(),file=None,content='Meeting agenda for tomorrow.',sender='',subject='',headers='',session=object(),user=types.SimpleNamespace(id=1,browser_alerts=False)))
    assert saved==[1]


def test_pasted_html_endpoint_normalizes_body_and_preserves_destinations(monkeypatch):
    calls=[]
    monkeypatch.setattr(api,'require_scan_allowance',lambda *a:None)
    monkeypatch.setattr(api,'run_scan',lambda **kw:calls.append(kw) or {'status':'CLEAN'})
    monkeypatch.setattr(api.store,'add_scan',lambda *a,**kw:None)
    asyncio.run(api.scan_email_endpoint(background_tasks=BackgroundTasks(),file=None,content=HTML,sender='',subject='',headers='',session=object(),user=types.SimpleNamespace(id=1,browser_alerts=False)))
    assert '<a' not in calls[0]['content']
    assert calls[0]['html_body']==HTML
    assert calls[0]['parsing_info']['source_kind']=='pasted_html'


def test_image_upload_filename_cannot_become_email_model_text(monkeypatch):
    calls=[]
    monkeypatch.setattr(api,'require_scan_allowance',lambda *a:None)
    monkeypatch.setattr(api,'run_scan',lambda **kw:calls.append(kw) or {'status':'CLEAN'})
    monkeypatch.setattr(api.store,'add_scan',lambda *a,**kw:None)
    upload=UploadFile(filename='urgent-bank-verification.png',file=io.BytesIO(b'fixture'))
    asyncio.run(api.scan_email_endpoint(background_tasks=BackgroundTasks(),file=upload,content=None,sender='',subject='',headers='',session=object(),user=types.SimpleNamespace(id=1,browser_alerts=False)))
    assert calls[0]['content']==''
    assert calls[0]['attachments'][0]['data']==b'fixture'

def test_gmail_sync_forwards_html_to_real_scanner_boundary(monkeypatch):
    calls=[]
    async def fetch(*a,**kw):return [{'body':'Visible text','html_body':HTML,'sender':'a@example.com','subject':'Document','headers_raw':'','gmail_id':'fixture'}]
    monkeypatch.setattr(inbox,'fetch_gmail_emails',fetch)
    monkeypatch.setattr(inbox,'decrypt_token',lambda value:value)
    monkeypatch.setattr(scan_engine,'scan_email',lambda **kw:calls.append(kw) or {'status':'CLEAN'})
    session=types.SimpleNamespace(get=lambda *a:None,exec=lambda *a:types.SimpleNamespace(first=lambda:None),commit=lambda:None,add=lambda *a:None,rollback=lambda:None)
    connection=types.SimpleNamespace(user_id=1,token_expiry=datetime.now(timezone.utc).replace(tzinfo=None)+timedelta(days=1),access_token_enc='fixture',refresh_token_enc='',last_synced=None)
    result=asyncio.run(inbox.sync_gmail_inbox(connection,session))
    assert result['new_scans']==1
    assert calls[0]['content']=='Visible text' and calls[0]['html_body']==HTML

def isolate_optional_sensors(monkeypatch):
    vector=types.ModuleType('services.vector_store')
    vector.search_similar_campaigns=lambda *a,**kw:{'has_match':False}
    vector.index_phishing_email=lambda *a,**kw:None
    monkeypatch.setitem(sys.modules,'services.vector_store',vector)
    bec=types.ModuleType('services.bec_detector')
    bec.predict_bec_intent=lambda *a,**kw:{'is_bec':False,'bec_score':0,'intent_category':'Clean'}
    monkeypatch.setitem(sys.modules,'services.bec_detector',bec)

def test_hidden_destination_reaches_url_and_contradiction_checks_without_changing_model_text(monkeypatch):
    isolate_optional_sensors(monkeypatch)
    inputs=[]; destinations=[]
    monkeypatch.setattr(scan_engine,'score_content',lambda t:inputs.append(t) or {'ml_powered':True,'ml_probability':.02,'score':2})
    def urls(text,*,additional_urls=None):
        destinations.extend(additional_urls or [])
        return {'score':0,'urls_found':len(additional_urls or []),'inference_errors':[]}
    monkeypatch.setattr(scan_engine,'score_urls',urls)
    class Predictor:
        is_calibrated=True
        pipeline_id=scan_engine.security_pipeline_id()
        def predict_set(self,*a,**kw):return {'CLEAN'},'CLEAN','SAFE_DELIVERY'
    result=scan_engine.evaluate_email_security('Your document is ready.','a@example.com','Document',predictor=Predictor(),html_body=HTML)
    assert inputs==['Your document is ready. Document']
    assert destinations==[TARGET]
    assert result['contra_result']['is_contradictory'] is True
    assert result['security_action']=='WARNING_BANNER'
    assert result['conformal_res']['threat_probability']==.02

def test_actual_url_scoring_checks_duplicate_destination_once_and_flags_truncation(monkeypatch):
    monkeypatch.setattr(scan_engine,'MODEL_LOADED',True)
    monkeypatch.setattr(scan_engine,'URL_MODEL',None)
    monkeypatch.setattr(scan_engine,'DOMAIN_MODEL',None)
    monkeypatch.setattr(scan_engine,'resolve_short_url',lambda u:u)
    monkeypatch.setattr('utils.threat_intel.check_virustotal',lambda *a:{'available':False})
    result=scan_engine.score_urls('See https://example.com/help',additional_urls=['https://example.com/help']*3)
    assert result['urls_found']==result['urls_analyzed']==1
    capped=scan_engine.score_urls('',additional_urls=[f'https://example.com/help/{i}' for i in range(11)])
    assert capped['urls_found']==11 and capped['urls_analyzed']==10
    assert capped['scan_complete'] is False
    assert any('incomplete' in e for e in capped['inference_errors'])

def test_direct_html_passes_decoded_href_once_and_benign_html_does_not_force_warning(monkeypatch):
    isolate_optional_sensors(monkeypatch)
    inputs=[];received=[]
    monkeypatch.setattr(scan_engine,'score_content',lambda t:inputs.append(t) or {'ml_powered':True,'ml_probability':.02,'score':2})
    def score(text,*,additional_urls=None):
        received.append(list(dict.fromkeys(scan_engine.extract_urls(text)+list(additional_urls or []))))
        return {'score':0,'urls_found':len(received[-1]),'inference_errors':[]}
    monkeypatch.setattr(scan_engine,'score_urls',score)
    class Predictor:
        is_calibrated=True
        pipeline_id=scan_engine.security_pipeline_id()
        def predict_set(self,*a,**kw):return {'CLEAN'},'CLEAN','SAFE_DELIVERY'
    html='<a href="https://example.com/help?a=1&amp;b=2">Open help</a>'
    result=scan_engine.evaluate_email_security(html,predictor=Predictor())
    assert inputs==[html+' ']
    assert received==[['https://example.com/help?a=1&b=2']]
    assert result['security_action']=='SAFE_DELIVERY'

def test_actual_gmail_fetch_preserves_hidden_html(monkeypatch):
    payload={'mimeType':'multipart/alternative','headers':[{'name':'From','value':'a@example.com'}],'parts':[
        {'mimeType':'text/plain','body':{'data':base64.urlsafe_b64encode(b'Visible body').decode()}},
        {'mimeType':'text/html','body':{'data':base64.urlsafe_b64encode(HTML.encode()).decode()}}]}
    class Client:
        def __init__(self,*a,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def get(self,url,**kw):
            data={'messages':[{'id':'fixture'}]} if url==inbox.GMAIL_MESSAGES_URL else {'payload':payload}
            return types.SimpleNamespace(status_code=200,json=lambda:data)
    monkeypatch.setattr(inbox.httpx,'AsyncClient',Client)
    emails=asyncio.run(inbox.fetch_gmail_emails('fixture'))
    assert emails[0]['body']=='Visible body' and emails[0]['html_body']==HTML
