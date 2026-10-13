import pytest

from scripts.prepare_brown_email_benchmark import extract_page, header_paragraph


def source(label='Legitimate Email'):
    return dict(label=label, title='PHISHING PAGE TITLE must not become model subject',
        url='https://it.brown.edu/phish-bowl-alerts/example', path='brown_pages/example.html', sha256='a' * 64)


def page(body, intro='', label='Legitimate Email'):
    return '<span class="flex_tag_item">' + label + '</span><div class="flex-intro">' + intro + '</div><div class="flex-body">' + body + '</div>'


MAIL = '<p>Dear customer, ' + 'Please review your genuine account notification. ' * 12 + '</p>'


def test_original_headers_in_body_removed_from_message_not_lost():
    body = '<p>From: Team &lt;team@example.org&gt;<br>Date: yesterday<br>Subject: Actual subject<br>To: person@example.org</p>' + MAIL
    row = extract_page(page(body, '<p>This is legitimate according to our security team.</p>'), source())
    assert row['sender'] == 'Team <team@example.org>'
    assert row['subject'] == 'Actual subject'
    assert 'From:' not in row['body']
    assert 'our security team' not in row['body']
    assert row['eligible_for_text_metrics']
    assert row['headers'] == ''
    assert not row['receiver_authentication_verified']


def test_intro_original_header_is_used_but_publisher_title_is_not():
    row = extract_page(page(MAIL, '<p>From: team@example.org<br>Subject: Message subject</p>'), source())
    assert row['subject'] == 'Message subject'
    assert row['eligible_for_text_metrics']
    no_header = extract_page(page(MAIL), source())
    assert no_header['subject'] == ''
    assert 'PHISHING PAGE TITLE' not in no_header['email_input']['body']


def test_original_href_is_retained_without_publisher_intro_link():
    row = extract_page(page(MAIL + '<a href="https://example.org/real-destination">Review</a>',
        '<a href="https://it.brown.edu/editorial-help">Publisher explanation</a>'), source())
    assert row['email_input']['link_destinations'] == ['https://example.org/real-destination']


def test_image_only_mail_kept_as_coverage_failure():
    row = extract_page(page('<img src="/screenshot.png" alt="message"/>'), source())
    assert not row['eligible_for_text_metrics']
    assert row['label'] == 0
    assert row['images_unscanned'][0]['src'] == '/screenshot.png'
    assert 'insufficient_visible_message_text' in row['coverage_reasons']


def test_multiple_mail_variants_are_not_flattened_into_one_validated_email():
    body = '<p>From: a@example.org<br>Subject: One</p>' + MAIL + '<p>From: b@example.org<br>Subject: Two</p>' + MAIL
    row = extract_page(page(body), source())
    assert not row['eligible_for_text_metrics']
    assert 'multiple_message_header_blocks' in row['coverage_reasons']
    assert len(row['published_header_blocks']) == 2


def test_publisher_commentary_in_body_cannot_be_scored_as_email():
    row = extract_page(page('<p>The following variations were reported over the last several months.</p>' + MAIL), source())
    assert 'publisher_commentary_or_mixed_examples_in_message_section' in row['coverage_reasons']


def test_ambiguous_header_and_body_paragraph_preserved_flagged():
    body = '<p>From: team@example.org<br>Subject: Notice<br>Dear customer, this is the body and must be retained.</p>' + MAIL
    row = extract_page(page(body), source())
    assert 'must be retained' in row['body']
    assert 'ambiguous_published_header_paragraph' in row['coverage_reasons']


def test_mismatched_publisher_verdict_rejected():
    with pytest.raises(ValueError, match='tag'):
        extract_page(page(MAIL, label='Phishing Email (on campus)'), source())


def test_actual_phishing_tag_maps_without_model_prediction():
    row = extract_page(page(MAIL, label='Phishing Email (in the wild)'), source('Phishing Email (in the wild)'))
    assert row['label'] == 1
    assert row['task_label'] == 'phishing'
    assert not row['model_scores_consulted_for_selection']
    assert row['project_reviewer_ids'] == []


def test_zero_width_header_label_does_not_hide_original_sender():
    row = extract_page(page(MAIL, '<p>\u200bFrom: team@example.org<br>Subject: Notice</p>'), source())
    assert row['sender'] == 'team@example.org'


def test_delimited_header_variants_counted_and_sent_preserved():
    row = extract_page(page('<p>=======<br>From: a@example.org<br>Sent: yesterday<br>Subject: First</p>' + MAIL +
        '<p>=======<br>From: b@example.org<br>Subject: Second</p>' + MAIL), source())
    assert len(row['published_header_blocks']) == 2
    assert 'multiple_message_header_blocks' in row['coverage_reasons']


def test_inline_publisher_header_note_not_model_identity():
    row = extract_page(page('<p>From: team@example.org Note: This is one example.<br>Subject: Mail</p>' + MAIL), source())
    assert row['sender'] == 'team@example.org'
    assert row['published_header_blocks'][0]['publisher_inline_notes']
    assert 'one example' not in row['email_input']['sender']


def test_sender_alternatives_preserved_not_fabricated_single_sender():
    row = extract_page(page('<p>From: a@example.org OR<br>b@example.org OR others<br>Subject: Mail</p>' + MAIL), source())
    assert row['sender'] == ''
    assert 'b@example.org' in row['publisher_sender_alternatives']
    assert row['eligible_for_scoring']


def test_unverified_intro_message_boundary_explicit_coverage():
    row = extract_page(page(MAIL, '<p>This might be a message paragraph or publisher discussion.</p>'), source())
    assert 'message_introduction_boundary_unverified' in row['exclusion_reasons']
