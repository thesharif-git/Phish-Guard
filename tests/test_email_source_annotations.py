from pathlib import Path

import pytest

from training.email_source_annotations import (
    _spaphish_annotation, annotation_epoch_summary, brown_publisher_verdict,
    load_spaphish_publisher_annotations, publisher_reception_day,
    reported_day_pilot_protocol,
)


def annotation(date, label="1"):
    return _spaphish_annotation({"date": date, "Label": label, "hash": "a" * 64}, 0)


def test_day_precision_is_not_invented_utc():
    row = annotation("07/06/2024")
    assert row["observed_day"] == "2024-06-07"
    assert row["observed_at_utc"] is None
    assert row["time_precision"] == "day"
    assert not row["receiver_time_verified"]


@pytest.mark.parametrize("value", ["2024-06-07", "31/02/2024", "7/6/2024", "not a date"])
def test_invalid_declared_date_format_fails(value):
    with pytest.raises(ValueError):
        publisher_reception_day(value)


def test_missing_dates_remain_missing():
    assert publisher_reception_day("") is None


def test_publisher_expert_labels_do_not_fabricate_project_review():
    row = annotation("", "0")
    assert row["task_label"] == "legitimate"
    assert row["label_authority_kind"] == "publisher_expert_consensus"
    assert row["project_reviewer_ids"] == []
    assert not row["project_independent_review_completed"]
    assert not row["approved_for_deployment"]
    assert not row["english_generalization_proven"]


def test_unknown_publisher_task_label_rejected():
    with pytest.raises(ValueError):
        annotation("", "Spam")


def test_unverified_release_bytes_cannot_acquire_expert_authority(tmp_path):
    (tmp_path / "corpus.csv").write_text("hash;Label;date\n" + "a" * 64 + ";1;07/06/2024\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_spaphish_publisher_annotations(tmp_path)


def test_descriptive_era_skew_and_boundary_counts():
    rows = [annotation("31/12/2023", "0"), annotation("01/01/2024"),
            annotation("31/12/2024", "0"), annotation("01/01/2025"), annotation("")]
    summary = annotation_epoch_summary(rows)
    assert summary["missing_publisher_day"] == 1
    assert summary["year_label_counts"][1]["phishing_fraction"] == .5
    protocol = reported_day_pilot_protocol(rows)
    assert protocol["pre_group_candidate_counts"]["validation"] == {"legitimate": 1, "phishing": 1}
    assert protocol["missing_date_rows_unassigned"] == 1
    assert protocol["template_groups_must_remain_disjoint"]
    assert not protocol["receiver_verified_time_test"]
    assert not protocol["english_generalization_evidence"]


def test_brown_publisher_verdict_is_valid_external_authority_not_new_review():
    url = "https://it.brown.edu/phish-bowl-alerts/microsoft-account-security-alert-legitimate-email"
    result = brown_publisher_verdict("publisher_verified_nonphishing", url)
    assert result["task_label"] == "legitimate"
    assert result["project_reviewer_ids"] == []
    assert not result["project_independent_review_completed"]


@pytest.mark.parametrize("label,url", [
    ("This phishing text", "https://it.brown.edu/phish-bowl-alerts/example"),
    ("publisher_reported_phishing", "https://it.brown.edu.evil.test/phish-bowl-alerts/example"),
    ("publisher_reported_phishing", "http://it.brown.edu/phish-bowl-alerts/example"),
])
def test_brown_title_words_and_unrelated_domains_do_not_create_labels(label, url):
    with pytest.raises(ValueError):
        brown_publisher_verdict(label, url)
