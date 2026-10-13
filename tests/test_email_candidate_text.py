"""Research representation invariants, not phishing accuracy tests."""

import pytest

from utils.email_candidate_text import canonical_email_candidate_text


@pytest.mark.parametrize("value", [None, "", 123, [], {}])
def test_empty_and_non_string_inputs(value):
    assert canonical_email_candidate_text(value) == ""


def test_anchor_preserves_visible_text_and_destination_presence():
    text = '<a href="https://evil.example/login">Please verify now</a>'
    assert canonical_email_candidate_text(text) == "<url> please verify now"


@pytest.mark.parametrize("destination", [
    "h\u200bttps://evil.example/login",
    "ｈｔｔｐｓ：／／evil.example/login",
])
def test_normalization_precedes_plain_url_masking(destination):
    assert canonical_email_candidate_text("Please verify " + destination + " now") == "please verify <url> now"


@pytest.mark.parametrize("destination", [
    "h\u200bttps://evil.example/login",
    "ｈｔｔｐｓ：／／evil.example/login",
])
def test_normalization_precedes_href_detection(destination):
    assert canonical_email_candidate_text(f'<a href="{destination}">Verify account</a>') == "<url> verify account"


def test_non_visible_script_style_head_and_template_are_ignored():
    text = ('<head><title>Source title</title></head>'
            '<script>archive Enron 2001</script><style>.enron {color:red}</style>'
            '<template>hidden archived message</template><p>Please verify your account</p>')
    assert canonical_email_candidate_text(text) == "please verify your account"


def test_plain_header_body_and_angle_address_are_preserved():
    text = "From: Example <person@corp.example>\nSubject: Meeting\n\nPlease review the agenda."
    assert canonical_email_candidate_text(text) == "from: example <email> subject: meeting please review the agenda."


def test_angle_address_is_preserved_when_html_follows_header():
    text = 'From: Example <person@corp.example>\n<p>Please review the agenda.</p>'
    assert canonical_email_candidate_text(text) == "from: example <email> please review the agenda."


def test_visible_text_entities_and_plain_angle_content_survive():
    assert canonical_email_candidate_text("A &amp; B: 2 < 3. <note> Please review.") == "a & b: 2 < 3. <note> please review."


def test_malformed_unclosed_anchor_retains_content_and_link_presence():
    assert canonical_email_candidate_text('<div>Hello <a href=https://evil.example/login>verify now') == "hello <url> verify now"


def test_non_url_anchor_does_not_invent_destination():
    assert canonical_email_candidate_text('<a href="#section">Review details</a>') == "review details"


def test_standard_entity_masks_are_preserved():
    text = "Contact person@corp.example at 202-555-0178 about $1,200 and https://corp.example/pay."
    assert canonical_email_candidate_text(text) == "contact <email> at <phone> about <amount> and <url>"


def test_source_names_dates_and_security_words_are_not_blacklisted():
    text = "Archive reference: Enron 2001. Please verify Microsoft security notifications."
    assert canonical_email_candidate_text(text) == text.lower()


@pytest.mark.parametrize("tokens", [
    "<URL> <EMAIL> <PHONE> <AMOUNT>",
    "<url> <email> <phone> <amount>",
])
def test_existing_mask_tokens_are_literal_text_inside_html(tokens):
    text = f"<p>Contact {tokens} for details.</p>"
    assert canonical_email_candidate_text(text) == "contact <url> <email> <phone> <amount> for details."


def test_existing_mask_token_survives_beside_html_anchor():
    text = '<p>Previously masked <URL>. <a href="https://corp.example/login">Review</a></p>'
    assert canonical_email_candidate_text(text) == "previously masked <url>. <url> review"


@pytest.mark.parametrize("separator", ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f"])
def test_control_whitespace_preserves_word_boundaries(separator):
    assert canonical_email_candidate_text("wire" + separator + "transfer") == "wire transfer"


@pytest.mark.parametrize("separator", ["&#12;", "&#x0c;", "&Tab;", "&NewLine;"])
def test_valid_encoded_whitespace_preserves_word_boundaries(separator):
    assert canonical_email_candidate_text("wire" + separator + "transfer") == "wire transfer"


def test_control_whitespace_does_not_synthesize_a_destination():
    assert canonical_email_candidate_text("Please inspect http\x0bs://example.org") == "please inspect http s://example.org"
    assert canonical_email_candidate_text("Please inspect h\u200bttps://example.org") == "please inspect <url>"
