"""Reproduce reachable input/policy defects without network or production edits.

Small functions and blocks are compiled directly from current source AST/text.
Fixtures establish behavior, not real-world error rates or ground-truth labels.
"""
from pathlib import Path
import sys,ast,json,re,hashlib,base64
from types import SimpleNamespace
from email.message import EmailMessage
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.email_parser import parse_eml_content
OUT=ROOT/'benchmarks/code_path_audit_20261003'

def selected_functions(path,names,scope):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)

def block(source,start,end,scope):
    import textwrap
    chunk=source[source.index(start):source.index(end,source.index(start))]
    exec(compile(ast.parse(textwrap.dedent(chunk)),'current_source_block','exec'),scope)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/p for p in ('utils/email_parser.py','services/inbox_connector.py','services/scan_engine.py','utils/url_scanner.py','routers/api.py','services/origintrace_pillars.py','services/bec_detector.py','training/train_url_and_domain_models.py')]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    scope={'re':re}
    selected_functions(ROOT/'services/scan_engine.py',{'extract_urls','score_headers'},scope)
    selected_functions(ROOT/'utils/url_scanner.py',{'extract_domain','get_status'},scope)
    findings=[]
    target='https://credential-fixture.invalid/collect'
    for plain in (False,True):
        message=EmailMessage();message['From']='sender@example.invalid';message['To']='recipient@example.invalid';message['Subject']='Document available'
        if plain:message.set_content('Your document is ready. Open document.')
        message.add_alternative(f'<html><body>Your document is ready. <a href="{target}">Open document</a></body></html>',subtype='html')
        parsed=parse_eml_content(message.as_bytes())
        assert target not in parsed['body'] and not scope['extract_urls'](parsed['body'])
        findings.append({'id':'eml_html_link_loss','plain_alternative':plain,'parsed_body':parsed['body'],'links_received_by_email_url_extractor':scope['extract_urls'](parsed['body']),'original_html_destination':target})
    inbox={'re':re,'base64':base64}
    selected_functions(ROOT/'services/inbox_connector.py',{'_decode_base64_url','_extract_body_from_payload'},inbox)
    payload={'mimeType':'text/html','body':{'data':base64.urlsafe_b64encode(f'<a href="{target}">Open document</a>'.encode()).decode()}}
    body=inbox['_extract_body_from_payload'](payload)
    assert target not in body
    findings.append({'id':'gmail_html_link_loss','parsed_body':body,'links_received_by_email_url_extractor':scope['extract_urls'](body)})
    headers=scope['score_headers']('X-Note: spf=pass dkim=pass dmarc=pass')
    assert headers['all_passed'] and headers['available']
    findings.append({'id':'untrusted_header_substring_authentication','fixture_header':'X-Note: spf=pass dkim=pass dmarc=pass','actual_result':headers,'cryptographic_checks_performed':False})
    raw='https://name:secret@actual-host.invalid/path'
    from urllib.parse import urlsplit
    extracted=scope['extract_domain'](raw)
    assert extracted!=urlsplit(raw).hostname
    findings.append({'id':'userinfo_hostname_misparse','fixture_url':raw,'actual_host':urlsplit(raw).hostname,'scanner_extracted_domain':extracted})
    web=(ROOT/'utils/url_scanner.py').read_text(encoding='utf-8')
    for score,uncertainty in ((10,1.0),(65,.1)):
        context={'p4_result':{'uncertainty':uncertainty},'contra_analysis':{},'final_score':score}
        block(web,'    uncertainty_val = p4_result.get','    return {\n        "url": raw_url,',context)
        findings.append({'id':'web_verdict_missing_evidence_or_conflict','fixture_score':score,'fixture_uncertainty':uncertainty,'actual_status':scope['get_status'](score),'actual_verdict_state':context['verdict_state']})
    assert findings[-2]['actual_verdict_state']=='CLEAN'
    assert findings[-1]['actual_status']=='SUSPICIOUS' and findings[-1]['actual_verdict_state']=='PHISHING'
    engine=(ROOT/'services/scan_engine.py').read_text(encoding='utf-8')
    loop=engine[engine.index('    url_work = []'):engine.index('    # Run VirusTotal checks for all URLs IN PARALLEL')]
    import textwrap
    class NeverCalled:
        calls=0
        def predict_proba(self,*args):
            self.calls+=1
            raise AssertionError('Authority bypass should skip model')
    model=NeverCalled()
    context={'urls':['https://popular-fixture.invalid/assets/document.html'],'extract_urls':scope['extract_urls'],'resolve_short_url':lambda u:u,'_extract_domain':scope['extract_domain'],'is_safe_authority_domain':lambda d:True,'urlparse':urlsplit,'is_ugc_hosting_platform':lambda *a:False,'URL_MODEL':model,'DOMAIN_MODEL':None,'re':re}
    exec(compile(textwrap.dedent(loop),'actual_email_url_loop','exec'),context)
    assert model.calls==0 and context['url_work'][0]['score']==0
    findings.append({'id':'authority_anchor_skips_model','fixture_assumed_authority':True,'url':context['urls'][0],'model_calls':model.calls,'actual_work_item':context['url_work'][0],'limitation':'Authority predicate supplied by fixture; frequency and classification of real popular hosts not measured'})
    extracted=scope['extract_urls'](' '.join(f'https://fixture-{i}.invalid/path' for i in range(11)))
    assert len(extracted)==11 and 'urls = all_urls[:10]' in engine
    findings.append({'id':'first_ten_url_truncation','extracted_count':len(extracted),'analyzed_slice_count':len(extracted[:10]),'omitted_destination':extracted[-1],'scope':'Source-level cap; no assertion that this fixture contains a malicious URL'})
    from services.origintrace_pillars import ProvenanceMatrix,InfrastructureLineage,SemanticDeception,EvidentialFusion
    fixture_url='https://neutral-fixture.invalid/'
    p1=ProvenanceMatrix.analyze_web(domain='neutral-fixture.invalid',url=fixture_url,tls_intel={})
    p2=InfrastructureLineage.analyze_web(domain='neutral-fixture.invalid',dns_ephemerality={},asn_intel={},geo={})
    missing=SemanticDeception.analyze_web(html_analysis={'success':False,'error':'Fixture retrieval failed'},url=fixture_url,domain='neutral-fixture.invalid')
    completed=SemanticDeception.analyze_web(html_analysis={'success':True},url=fixture_url,domain='neutral-fixture.invalid')
    assert missing['confidence']==completed['confidence']==.85
    assert missing['threat_prob']==completed['threat_prob']==0
    fused=EvidentialFusion.synthesize(p1,p2,missing,context='web')
    assert fused['belief_clean']>.9 and fused['uncertainty']<.1
    findings.append({'id':'missing_live_evidence_assigned_clean_mass','scope':'Actual web pillar/fusion functions with neutral fixture URL and unavailable live sensors; no website ground truth','provenance':p1,'infrastructure':p2,'semantic_failed_page':missing,'semantic_completed_no_findings':completed,'fusion':fused})
    bec_tree=ast.parse((ROOT/'services/bec_detector.py').read_text(encoding='utf-8'))
    patterns=[n for n in bec_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BEC_PATTERNS' for t in n.targets)]
    bec={'re':re,'Dict':dict,'Any':object,'logger':SimpleNamespace(debug=lambda *a:None)}
    exec(compile(ast.Module(body=patterns,type_ignores=[]),'actual_bec_patterns','exec'),bec)
    selected_functions(ROOT/'services/bec_detector.py',{'predict_bec_intent'},bec)
    for probability in (.85,.95):
        def classifier(*args,p=probability):return {'labels':['fraudulent wire transfer request','legitimate business email'],'scores':[p,1-p]}
        bec['_get_bec_classifier']=lambda:classifier
        actual=bec['predict_bec_intent']('Please handle the request described in the attached message.',urls_found=0)
        findings.append({'id':'bec_transformer_score_diluted_without_rule_hits','fixture_transformer_output':probability,'actual_result':actual,'scope':'Injected classifier output through actual rule/fusion function; no model accuracy claim'})
        assert actual['bec_score']==int(probability*50)
    import math
    lexical={'re':re,'math':math,'urlparse':urlsplit}
    tree=ast.parse((ROOT/'training/train_url_and_domain_models.py').read_text(encoding='utf-8'))
    required={'TLD_RISK','KNOWN_SHORTENERS','BRAND_KEYWORDS','OFFICIAL_DOMAINS'}
    assignments=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in required for t in n.targets)]
    exec(compile(ast.Module(body=assignments,type_ignores=[]),'actual_url_constants','exec'),lexical)
    selected_functions(ROOT/'training/train_url_and_domain_models.py',{'extract_url_features','_entropy','_tld_risk','_char_continuation_rate','_unique_char_ratio'},lexical)
    a='https://ordinary.invalid/paypal/login';b='https://ordinary.invalid/playap/lingo'
    af=lexical['extract_url_features'](a);bf=lexical['extract_url_features'](b)
    assert af==bf and af['has_brand_in_url']==0
    findings.append({'id':'url_semantic_feature_collision','fixture_urls':[a,b],'identical_numeric_feature_vectors':True,'brand_present_in_path_but_brand_feature':af['has_brand_in_url'],'feature_vector':af,'scope':'Representation limitation only; neither fixture has a verified phishing/benign label'})
    assert hashes=={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'complete':True,'no_network_requests':True,'production_unchanged':True,'scope':'Actual source function/block counterexamples, not full live performance benchmark','input_sha256':hashes,'counterexamples':findings}
    (OUT/'counterexamples.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
