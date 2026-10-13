"""Transport invariance and evidence-loss regressions for the Email Scanner."""
import base64
from email.message import EmailMessage
from utils.email_parser import parse_eml_content
from services.inbox_connector import _extract_body_from_payload


def test_single_and_multipart_html_have_identical_visible_model_body():
    markup = '<html><head><style>.hidden{color:red}</style></head><p>Review your document.</p><script>noise()</script><a href="https://example.com/doc">Open document</a></html>'
    single = EmailMessage()
    single.set_content(markup, subtype='html')
    multi = EmailMessage()
    multi.add_alternative(markup, subtype='html')
    a, b = parse_eml_content(single.as_bytes()), parse_eml_content(multi.as_bytes())
    assert a['body'] == b['body'] == 'Review your document. Open document'
    assert a['html_body'].strip() == b['html_body'].strip() == markup


def test_valid_utf8_without_charset_is_not_replaced_by_ascii_decoder():
    raw = b'Content-Type: text/plain\nContent-Transfer-Encoding: 8bit\n\n' + 'Réunion jeudi — document'.encode()
    assert parse_eml_content(raw)['body'] == 'Réunion jeudi — document'


def test_eml_attached_message_cannot_replace_empty_outer_body():
    outer = EmailMessage()
    outer['Subject'] = 'See attached message'
    outer.set_content('')
    nested = EmailMessage()
    nested.set_content('Confidential wire transfer inside attached email')
    outer.add_attachment(nested)
    assert parse_eml_content(outer.as_bytes())['body'] == ''


def test_gmail_prefers_plain_alternative_regardless_of_order_and_skips_attachments():
    encode = lambda s: base64.urlsafe_b64encode(s.encode()).decode()
    payload = {'mimeType': 'multipart/mixed', 'parts': [
        {'mimeType': 'text/plain', 'filename': 'attached.txt', 'body': {'data': encode('Attached text')}},
        {'mimeType': 'multipart/alternative', 'parts': [
            {'mimeType': 'text/html', 'body': {'data': encode('<p>HTML alternative</p>')}},
            {'mimeType': 'text/plain', 'body': {'data': encode('Visible plain body')}}]}]}
    assert _extract_body_from_payload(payload) == 'Visible plain body'

