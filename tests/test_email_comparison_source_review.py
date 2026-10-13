"""Source uncertainty must not be silently converted into reliable ground truth."""
from scripts.curate_email_comparison_candidates import inspect_row


def test_recent_upload_does_not_promote_legacy_or_generated_messages():
    legacy = dict(Source='SpamAssassin.csv', **{'Created by': 'Human'}, Year=2002,
                  Body='Old ham message', Subject='Discussion', Type='Valid')
    reasons, _ = inspect_row(legacy)
    assert 'outside_claimed_contemporary_year_range' in reasons
    assert 'legacy_public_or_nonmanual_source_not_independent_recent_candidate' in reasons
    generated = dict(legacy, Source='Manual', Year=2026, **{'Created by': 'LLM'})
    assert 'human_authorship_field_missing_or_noncanonical_requires_review' in inspect_row(generated)[0]


def test_missing_content_or_authorship_is_not_invented():
    row = dict(Source='Manual', **{'Created by': 'Huamn'}, Year=2025,
               Body=None, Subject=123, Type='Spam')
    reasons, _ = inspect_row(row)
    assert 'human_authorship_field_missing_or_noncanonical_requires_review' in reasons
    assert 'missing_or_invalid_body' in reasons and 'invalid_subject_field' in reasons


def test_missing_optional_subject_does_not_make_a_valid_body_garbage():
    row = dict(Source='Manual', **{'Created by': 'Human'}, Year=2025,
               Body='Complete visible body to review.', Subject=None, Type='Valid')
    reasons, flags = inspect_row(row)
    assert not reasons
    assert 'missing_subject_optional_header' in flags
