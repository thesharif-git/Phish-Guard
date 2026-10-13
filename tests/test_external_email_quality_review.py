"""Audit fidelity and review isolation, not email classifier accuracy."""
from email.message import EmailMessage
from scripts.audit_external_email_quality import inspect_message, review_document


def test_missing_charset_utf8_is_recovered_in_production_and_diagnostic():
    raw = b'From: fixture@example.com\nContent-Type: text/plain\nContent-Transfer-Encoding: 8bit\n\n' + 'Réunion jeudi — document'.encode('utf-8')
    report, card, _ = inspect_message(raw)
    assert 'missing_charset_strict_utf8_recovery' in report['flags']
    assert card['diagnostic_body'] == 'Réunion jeudi — document'
    assert card['production_body'] == card['diagnostic_body']
    assert report['independent_security_label'] is None
    assert not report['approved_for_training'] and not report['approved_for_final_benchmark']


def test_invalid_declared_utf8_is_not_silently_guessed_as_latin1():
    raw = b'Content-Type: text/plain; charset=utf-8\n\nInvalid \x92 byte payload'
    report, card, _ = inspect_message(raw)
    assert 'unresolved_encoding' in report['flags']
    assert 'replacement_characters_remain' in report['flags']
    assert card['diagnostic_body'] != 'Invalid ’ byte payload'


def test_transport_markup_defect_and_nested_attachment_are_separate_evidence():
    raw = b'Content-Type: text/html; charset=utf-8\n\n<html><head><style>hidden boilerplate</style></head><p>Review document</p><script>alert(1)</script><a href="https://fixture.example">Open</a></html>'
    report, card, _ = inspect_message(raw)
    assert 'single_part_html_markup_enters_production_body' not in report['flags']
    assert card['diagnostic_body'] == 'Review document Open'
    assert not report['structural_feature_indices_changed']
    outer = EmailMessage()
    outer.set_content('Outer visible text')
    attached = EmailMessage()
    attached.set_content('Attached message must not be selected as inline text')
    outer.add_attachment(attached)
    report, card, _ = inspect_message(outer.as_bytes())
    assert card['diagnostic_body'] == 'Outer visible text'
    assert len(report['parts']) == 1


def test_review_html_escapes_all_message_content_and_has_no_live_destinations():
    attack = '</pre><script src="https://fixture.example/x">alert(1)</script><img src="https://fixture.example/pixel">'
    card = dict(review_id='fixture', sender=attack, subject=attack, headers=attack,
                production_body=attack, diagnostic_body=attack, technical_flags=[attack],
                anchor_destinations=[dict(text=attack, href='https://fixture.example/x')])
    document = review_document([card])
    assert '<script' not in document and '<img' not in document and '<a ' not in document
    assert "connect-src 'none'" in document
    assert '&lt;script' in document
    assert 'model predictions are withheld' in document
