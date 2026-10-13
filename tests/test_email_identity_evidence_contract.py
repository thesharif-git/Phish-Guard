"""Identity assertions and observed URL relationships must not invent fraud."""
import sys
import types

import numpy as np
import pytest

from services import scan_engine as engine
from services.origintrace_pillars import CrossEntityContradictionEngine as Contradiction, ProvenanceMatrix


@pytest.mark.parametrize('name,domain', [
    ('Appleton Family', 'appleton.example'),
    ('Pineapple Newsletter', 'pineapple.example'),
    ('Chaseley School', 'chaseley.example'),
])
def test_personal_name_substring_is_not_a_brand_identity_claim(name, domain):
    sender = f'{name} <news@{domain}>'
    result = Contradiction.analyze(sender_header=sender, urls=[f'https://{domain}/photos'])
    assert result['claimed_brand'] is None
    assert not result['is_contradictory']
    assert result['mismatches'] == []
    provenance = ProvenanceMatrix.analyze(sender, '', urls=[f'https://{domain}/photos'])
    assert not provenance['display_name_domain_difference']
    assert not provenance['lookalike_domain']


def test_known_brand_domain_relationship_is_consistent_between_email_pillars():
    sender = 'Microsoft <notice@microsoftonline.com>'
    provenance = ProvenanceMatrix.analyze(sender, '', urls=['https://login.microsoftonline.com/login'])
    contradiction = Contradiction.analyze(sender_header=sender, urls=['https://login.microsoftonline.com/login'])
    assert provenance['score'] == contradiction['contradiction_score'] == 0
    assert not provenance['lookalike_domain']
    assert provenance['crypto_auth']['verified'] is False
    assert 'Verified' not in contradiction['alignment_summary']


@pytest.mark.parametrize('headers', [
    'X-Comment: d=attacker.example',
    'Authentication-Results: mx.example; dkim=pass header.d=attacker.example',
    'Subject: d=attacker.example\nX-Comment: d=another.example',
])
def test_dkim_tag_outside_signature_header_cannot_be_a_signer(headers):
    result = Contradiction.analyze(sender_header='Microsoft <a@microsoft.com>', raw_headers=headers)
    assert result['reported_dkim_domains'] == []
    assert result['observations'] == []
    assert not result['is_contradictory']


def test_multiple_folded_actual_signatures_are_preserved_without_claiming_validation():
    headers = ('DKIM-Signature: v=1; a=rsa-sha256;\n\td=MAILER.EXAMPLE; s=relay; b=raw\n'
               'DKIM-Signature: v=1; d=microsoft.com; b=another\n'
               'X-Comment: d=not-a-signer.example')
    result = Contradiction.analyze(sender_header='Microsoft <a@microsoft.com>', raw_headers=headers,
                                   spf='FAIL', dkim='FAIL', dmarc='FAIL')
    assert result['reported_dkim_domains'] == ['mailer.example', 'microsoft.com']
    assert not result['authentication_verified']
    assert not result['is_contradictory']
    assert len(result['observations']) == 2
    assert all(item['verified'] is False for item in result['observations'])
    assert all('were not checked' in item['evidence'] for item in result['observations'])
    assert result['signals'] == []


@pytest.mark.parametrize('tag', ['d=sender.example/path', 'd=one.example; d=two.example', 'd=user@sender.example'])
def test_invalid_or_duplicate_signing_domain_tags_cannot_be_identity_evidence(tag):
    result = Contradiction.analyze(raw_headers='DKIM-Signature: v=1; ' + tag + '; b=fake')
    assert result['reported_dkim_domains'] == []


@pytest.mark.parametrize('anchor', ['See microsoft.com for details', 'Contact support@example.com',
                                   'Your file on https://example.com', 'javascript:https://example.com'])
def test_anchor_prose_and_unsupported_scheme_are_not_visible_host_claims(anchor):
    result = Contradiction.analyze(anchor_links=[{'text': anchor, 'href': 'https://registration.example/start'}])
    assert not result['is_contradictory']
    assert result['mismatches'] == []


def test_literal_visible_url_hostname_difference_remains_review_evidence():
    result = Contradiction.analyze(anchor_links=[{'text': 'https://paypal.com/account',
                                                  'href': 'https://different.example/start'}])
    assert result['is_contradictory']
    assert result['mismatch_count'] == 1
    finding = result['mismatches'][0]
    assert finding['type'] == 'VISIBLE_LINK_HOST_DIFFERENCE'
    assert finding['verified'] is False
    assert 'does not establish phishing' in finding['evidence']
    assert not result['threat_probability_validated']


@pytest.mark.parametrize('shown,href', [
    ('https://www.example.com/a', 'https://example.com/b'),
    ('https://EXAMPLE.COM/a', 'https://example.com/a'),
    ('https://b\u00fccher.example/a', 'https://xn--bcher-kva.example/b'),
])
def test_equivalent_hostnames_are_not_different_identity_evidence(shown, href):
    assert not Contradiction.analyze(anchor_links=[{'text': shown, 'href': href}])['is_contradictory']


def test_partner_and_hosting_differences_are_observations_not_authorization_failures():
    result = Contradiction.analyze(sender_header='Microsoft Events <events@conference.example>',
        urls=['https://registration.example/register'], destination_isp='DigitalOcean')
    assert not result['is_contradictory']
    assert result['contradiction_score'] == 0
    assert {item['type'] for item in result['observations']} == {
        'DISPLAY_NAME_DOMAIN_DIFFERENCE', 'BRAND_LINK_HOST_RELATIONSHIP', 'HOSTING_PROVIDER_LIST_MATCH'}
    assert result['signals'] == []
    assert all(item['verified'] is False for item in result['observations'])


def test_brand_text_in_destination_cannot_create_a_sender_identity():
    result = Contradiction.analyze(destination_domain='google-conference.example',
                                   urls=['https://google-conference.example/register'])
    assert result['claimed_brand'] is None
    assert not result['is_contradictory']


def isolate_pipeline(monkeypatch, *, actual_urls=False):
    content_inputs = []
    monkeypatch.setattr(engine, 'EMAIL_INPUT_KIND', 'legacy_tfidf_structural')
    monkeypatch.setattr(engine, 'score_content', lambda text, **kwargs:
        content_inputs.append(text) or {'ml_powered': True, 'ml_probability': .02, 'score': 2,
                                      'keyword_hits': [], 'suspicious_phrases': [], 'is_english': True})
    if actual_urls:
        class CleanURLModel:
            feature_names_in_ = np.array(list(engine.extract_url_features('https://example.com')))
            def predict_proba(self, frame):
                return np.tile([.99, .01], (len(frame), 1))
        monkeypatch.setattr(engine, 'MODEL_LOADED', True)
        monkeypatch.setattr(engine, 'URL_MODEL', CleanURLModel())
        monkeypatch.setattr(engine, 'DOMAIN_MODEL', None)
        monkeypatch.setattr(engine, 'resolve_short_url', lambda url: url)
        monkeypatch.setattr('utils.threat_intel.check_virustotal', lambda *args: {'available': False})
    else:
        monkeypatch.setattr(engine, 'score_urls', lambda text, **kwargs:
            {'score': 0, 'urls_found': len(kwargs.get('additional_urls', [])), 'flagged_urls': [],
             'analyzed_urls': [], 'inference_errors': []})
    monkeypatch.setattr(engine, 'security_pipeline_id', lambda: 'identity-review-fixture')
    vector = types.ModuleType('services.vector_store')
    vector.search_similar_campaigns = lambda *args: {'has_match': False, 'matches': []}
    monkeypatch.setitem(sys.modules, 'services.vector_store', vector)
    bec = types.ModuleType('services.bec_detector')
    bec.predict_bec_intent = lambda *args, **kwargs: {'is_bec': False, 'bec_score': 0, 'signals': []}
    monkeypatch.setitem(sys.modules, 'services.bec_detector', bec)
    monkeypatch.setattr(engine, 'load_ml_model', lambda: pytest.fail('No weights should be loaded'))
    monkeypatch.setattr(engine.requests.sessions.Session, 'request',
                        lambda *args, **kwargs: pytest.fail('No network should be called'))

    class Predictor:
        is_calibrated = True
        pipeline_id = 'identity-review-fixture'
        def predict_set(self, value, **kwargs):
            assert value == .02
            return {'CLEAN'}, 'CLEAN', 'SAFE_DELIVERY'
    return Predictor(), content_inputs


@pytest.mark.parametrize('sender,headers,html', [
    ('Appleton Family <news@appleton.example>', '', '<a href="https://appleton.example/photos">Family photos</a>'),
    ('Microsoft Events <events@microsoft.com>', 'X-Comment: d=mailer.example',
     '<a href="https://registration.example/register">See microsoft.com for details</a>'),
    ('Microsoft Events <events@conference.example>', 'DKIM-Signature: v=1; d=mailer.example; b=raw',
     '<a href="https://registration.example/register">Register for the conference</a>'),
])
def test_connected_identity_checks_do_not_override_model_with_fabricated_contradiction(monkeypatch, sender, headers, html):
    predictor, inputs = isolate_pipeline(monkeypatch)
    result = engine.evaluate_email_security('Your event details are ready.', sender, 'Event details',
        headers, predictor=predictor, html_body=html, parsing_info={'text_complete': True})
    assert inputs == ['Your event details are ready. Event details']
    assert not result['contra_result']['is_contradictory']
    assert result['conformal_res']['threat_probability'] == .02
    assert result['final_score'] == 2
    assert result['security_action'] == 'SAFE_DELIVERY'


def test_literal_destination_review_preserves_original_model_probability_end_to_end(monkeypatch):
    predictor, _ = isolate_pipeline(monkeypatch)
    result = engine.evaluate_email_security('Your event details are ready.', predictor=predictor,
        html_body='<a href="https://different.example/start">https://paypal.com</a>',
        parsing_info={'text_complete': True})
    assert result['contra_result']['is_contradictory']
    assert result['security_action'] == 'WARNING_BANNER'
    assert result['conformal_res']['threat_probability'] == .02
    assert result['final_score'] == 2


@pytest.mark.parametrize('sender,url', [
    ('Google <notice@google.com>', 'https://accounts.google.com/login?token=example'),
    ('Appleton Family <news@appleton.example>', 'https://appleton.example/photos'),
    ('Newsletter <news@example.com>', 'https://example.com/guide/google'),
    ('Newsletter <news@example.com>', 'https://example.com/report?previous=192.0.2.1'),
])
def test_actual_embedded_url_and_identity_components_do_not_invent_probability_floors(monkeypatch, sender, url):
    predictor, inputs = isolate_pipeline(monkeypatch, actual_urls=True)
    result = engine.evaluate_email_security('Your requested details are ready.', sender, 'Details',
        predictor=predictor, html_body=f'<a href="{url}">Open details</a>',
        parsing_info={'text_complete': True})
    assert inputs == ['Your requested details are ready. Details']
    url_result = result['url_result']
    assert url_result['score'] == 0
    assert url_result['analyzed_urls'][0]['model_probability'] == .01
    assert url_result['flagged_urls'] == []
    assert not result['contra_result']['is_contradictory']
    assert result['conformal_res']['threat_probability'] == .02
    assert result['final_score'] == 2
    assert result['security_action'] == 'SAFE_DELIVERY'


def test_missing_url_model_cannot_become_safe_delivery_after_removing_topic_floors(monkeypatch):
    predictor, _ = isolate_pipeline(monkeypatch, actual_urls=True)
    monkeypatch.setattr(engine, 'URL_MODEL', None)
    result = engine.evaluate_email_security('Your event details are ready.', predictor=predictor,
        html_body='<a href="https://example.com/help">Help</a>', parsing_info={'text_complete': True})
    assert result['url_result']['analyzed_urls'][0]['model_probability'] is None
    assert 'URL ML unavailable' in result['sensor_errors']
    assert result['security_action'] == 'WARNING_BANNER'
    assert result['conformal_res']['threat_probability'] == .02
    assert result['final_score'] == 2
