import copy
import json

import pytest

from utils.email_input_contract import build_email_input, contract_sha256, validate_email_input


def test_original_identity_content_and_untrusted_header_assertions_preserved():
    headers = ("Reply-To: Help <reply@other.example>\nReturn-Path: <bounce@relay.example>\n"
               "Date: Tue, 1 Jan 2001 00:00:00 +0000\n")
    record = build_email_input("Please review.", subject="Original Subject", sender="Display <sender@corp.example>", headers=headers)
    assert record["body"] == "Please review."
    assert record["subject"] == "Original Subject"
    assert record["sender"] == "Display <sender@corp.example>"
    assert record["headers"] == headers
    assert record["reply_to"][0]["value"] == "Help <reply@other.example>"
    assert record["envelope_from"][0]["value"] == "<bounce@relay.example>"
    assert record["reply_to"][0]["verified"] is False
    assert "observed_at" not in record


def test_defaults_do_not_invent_identity_authentication_or_parse_success():
    record = build_email_input("Hello <person@corp.example>")
    assert record["sender"] == record["headers"] == ""
    assert record["reply_to"] == record["envelope_from"] == []
    assert record["trusted_authentication"] is None
    assert record["authentication"]["spf"] == "UNKNOWN"
    assert record["authentication"]["verified"] is False
    assert record["parsing"]["completeness"] == "UNKNOWN"
    assert record["availability"]["trusted_authentication"] is False


def test_reported_authentication_never_becomes_trusted():
    headers = "Authentication-Results: receiver.example; spf=pass; dkim=pass; dmarc=pass\n"
    record = build_email_input("Hello", headers=headers)
    assert record["availability"]["reported_authentication"] is True
    assert len(record["authentication"]["reported_results"]) == 3
    assert record["authentication"]["verified"] is False
    assert record["authentication"]["available"] is False
    assert record["trusted_authentication"] is None


def test_actual_url_metadata_outside_body_retained_with_provenance():
    record = build_email_input("Review the shared document.", supplementary_urls=["https://corp.example/doc"])
    assert record["body"] == "Review the shared document."
    assert record["link_destinations"] == ["https://corp.example/doc"]
    assert record["links"][0]["source"] == "supplementary_url"
    assert record["links"][0]["reference"] == {"field": "supplementary_urls", "index": 0}


def test_html_destination_visible_text_and_original_markup_retained():
    markup = '<p><a href="https://Other.EXAMPLE/login?x=1&amp;y=2">Microsoft account</a></p>'
    record = build_email_input("Visible plain alternative", html_body=markup)
    assert record["html_body"] == markup
    assert record["link_destinations"] == ["https://other.example/login?x=1&y=2"]
    assert record["links"][0]["anchor_text"] == "Microsoft account"
    assert record["links"][0]["reference"]["field"] == "html_body"


@pytest.mark.parametrize("raw", ["h\u200bttps://Corp.EXAMPLE/login", "ｈｔｔｐｓ：／／Corp.EXAMPLE/login"])
def test_normalized_link_retains_exact_original_slice(raw):
    body = "Review " + raw + " now."
    record = build_email_input(body)
    link = record["links"][0]
    assert record["body"] == body
    assert link["raw_value"] == raw
    assert link["destination"] == "https://corp.example/login"
    assert body[link["reference"]["start"]:link["reference"]["end"]] == raw
    assert link["normalization_applied"] is True


def test_comments_scripts_and_styles_do_not_create_link_evidence():
    markup = ('<!-- https://comment.example -->'
              '<script>https://script.example</script><style>https://style.example</style>'
              '<p>Review https://visible.example</p>')
    record = build_email_input(markup)
    assert record["link_destinations"] == ["https://visible.example"]
    assert record["parsing"]["flags"] == ["body:noncontent_html_sections_ignored"]
    assert record["body"] == markup


def test_unsupported_url_values_retained_not_scored_or_rewritten():
    values = ["ftp://files.example/a", "https://", "https://corp.example:99999/a", "/relative"]
    record = build_email_input("Hello", supplementary_urls=values)
    assert record["link_destinations"] == []
    assert [item["raw_value"] for item in record["unsupported_urls"]] == values
    assert all(item["reason"] for item in record["unsupported_urls"])
    assert record["supplementary_urls"] == values


def test_duplicate_destination_keeps_each_provenance_occurrence():
    record = build_email_input("https://corp.example/a", supplementary_urls=["https://corp.example/a"])
    assert record["link_destinations"] == ["https://corp.example/a"]
    assert len(record["links"]) == 2
    assert {item["source"] for item in record["links"]} == {"body_text", "supplementary_url"}


@pytest.mark.parametrize("info, expected", [
    (None, "UNKNOWN"), ({}, "UNKNOWN"), ({"text_complete": True}, "COMPLETE"),
    ({"text_complete": False}, "INCOMPLETE"),
    ({"text_complete": True, "errors": ["invalid_utf8"]}, "INCOMPLETE"),
    ({"text_complete": True, "unscanned_parts": [{"content_type": "application/pdf"}]}, "INCOMPLETE"),
])
def test_explicit_parsing_availability_and_completeness(info, expected):
    record = build_email_input("Hello", parsing_info=info)
    assert record["parsing_info"] == info
    assert record["parsing"]["completeness"] == expected


@pytest.mark.parametrize("kwargs", [
    {"body": None}, {"body": {}}, {"body": "Hello", "subject": 10},
    {"body": "Hello", "sender": None}, {"body": "Hello", "headers": []},
    {"body": "Hello", "html_body": False}, {"body": "Hello", "supplementary_urls": "https://corp.example"},
    {"body": "Hello", "supplementary_urls": [1]}, {"body": "Hello", "parsing_info": []},
    {"body": "Hello", "parsing_info": {"errors": "broken"}},
    {"body": "Hello", "parsing_info": {"text_complete": "true"}},
])
def test_invalid_input_types_fail_instead_of_stringifying(kwargs):
    with pytest.raises(TypeError):
        build_email_input(**kwargs)


def test_training_and_inference_json_serialization_have_identical_contract_hash():
    record = build_email_input("Review https://corp.example/doc", subject="Résumé", headers="Reply-To: person@corp.example\n")
    restored = json.loads(json.dumps(record, ensure_ascii=False))
    assert validate_email_input(restored) == record
    assert contract_sha256(restored) == contract_sha256(record)


@pytest.mark.parametrize("field, value", [
    ("link_destinations", ["https://invented.example"]),
    ("trusted_authentication", {"verified": True}),
    ("prediction", 0.99),
])
def test_stale_or_invented_derived_evidence_is_rejected(field, value):
    record = build_email_input("Hello")
    record[field] = value
    with pytest.raises(ValueError):
        validate_email_input(record)


def test_derived_boolean_schema_does_not_accept_numeric_equivalent():
    record = build_email_input("Hello")
    record["availability"]["body"] = 1
    with pytest.raises(ValueError):
        validate_email_input(record)


@pytest.mark.parametrize("info", [{"other": float("nan")}, {"other": object()}, {1: "bad key"}])
def test_non_json_parsing_evidence_is_rejected(info):
    with pytest.raises((ValueError, TypeError)):
        build_email_input("Hello", parsing_info=info)


@pytest.mark.parametrize("key", ["label", "source_label", "model_probability", "final_score", "observed_at_utc", "trusted_receiver"])
def test_caller_labels_scores_or_provenance_cannot_enter_parsing_contract(key):
    with pytest.raises(ValueError, match="Unsupported parsing_info"):
        build_email_input("Hello", parsing_info={key: "caller supplied"})


@pytest.mark.parametrize("info", [
    {"version": 1}, {"source_kind": None}, {"notes": "a note"}, {"notes": [1]},
    {"mime_defects": [None]}, {"unscanned_parts": ["attachment"]},
    {"unscanned_parts": [{"filename": 4}]},
])
def test_documented_parser_field_types_are_required(info):
    with pytest.raises(TypeError):
        build_email_input("Hello", parsing_info=info)


def test_nested_unscanned_part_cannot_carry_score_or_label():
    with pytest.raises(ValueError, match="Unsupported unscanned MIME"):
        build_email_input("Hello", parsing_info={"unscanned_parts": [{"filename": "image.png", "source_label": "Phishing"}]})


def test_all_documented_parser_fields_are_retained_exactly():
    info = {"version": "3.0", "text_complete": False, "notes": ["recovered_utf8"],
            "errors": ["invalid_declared_charset"], "unscanned_parts": [{"filename": "report.pdf", "content_type": "application/pdf"}],
            "mime_defects": ["CloseBoundaryNotFoundDefect"], "source_kind": "rfc_email"}
    assert build_email_input("Hello", parsing_info=info)["parsing_info"] == info


def test_input_containers_are_not_mutated_or_shared():
    info = {"errors": ["incomplete"], "unscanned_parts": [{"filename": "example.pdf"}]}
    urls = ["https://corp.example"]
    originals = copy.deepcopy((info, urls))
    record = build_email_input("Hello", supplementary_urls=urls, parsing_info=info)
    record["parsing_info"]["errors"].append("new")
    record["supplementary_urls"].append("https://other.example")
    assert (info, urls) == originals
