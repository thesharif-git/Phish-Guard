"""Synthetic fixtures validate engineering contracts, never email accuracy."""

import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts import train_email_evidence_candidate as trainer
from training.email_evidence_model import (
    NUMERIC_NAMES, EmailEvidenceFeatures, _identity_text, _numeric,
    contract_source_hashes, email_evidence_estimator,
    load_reviewed_email_evidence, predict_email_evidence_probabilities,
)
from utils.email_input_contract import build_email_input, contract_sha256


def _fixture_corpus(tmp_path):
    """Fictional reviewed metadata is test-only and written to pytest temp files."""
    rows = []
    for split, sources, count in (
        ("train", ("origin-a", "origin-b"), 3),
        ("val", ("origin-a", "origin-b"), 1),
        ("source_test", ("origin-c", "origin-d"), 1),
        ("time_test", ("origin-a", "origin-b"), 1),
    ):
        for source in sources:
            for label in ("legitimate", "phishing"):
                for index in range(count):
                    identifier = f"fixture-{len(rows)}"
                    # Unique complete-message shingles avoid template leakage.
                    unique = " ".join("q" + hashlib.sha256(f"{identifier}-{n}".encode()).hexdigest()[:12] for n in range(20))
                    words = "Routine meeting agenda" if label == "legitimate" else "Verify transfer required"
                    destination = "https://portal.example.com/agenda" if label == "legitimate" else "https://other.example.net/transfer"
                    evidence = build_email_input(words + " " + unique + " " + destination,
                        subject="Fixture notice", sender="Person <person@example.com>",
                        headers="From: Person <person@example.com>\nReply-To: person@example.com\n",
                        parsing_info={"text_complete": True})
                    rows.append(dict(id=identifier, task_label=label, split=split, source=source,
                        email_input=evidence, input_contract_sha256=contract_sha256(evidence),
                        label_review_completed=True, independent_review_completed=True,
                        extraction_verified=True, reviewer_id="fictional-test-reviewer-a",
                        second_reviewer_id="fictional-test-reviewer-b",
                        label_evidence_reference="fixture://" + identifier,
                        label_rationale="Synthetic engineering fixture; not security ground truth",
                        training_reuse_permitted=True,
                        campaign_cluster_id=identifier,
                        observed_at_utc="2025-01-01T00:00:00Z" if split == "time_test" else "2024-01-01T00:00:00Z",
                        time_authority="publisher_observation"))
    corpus = tmp_path / "training_fixture.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = dict(schema_version=3, input_kind="structured_email_evidence_v1",
        input_contract_version="email-evidence-v1", label_scope="phishing_bec_vs_nonmalicious",
        approved_for_training=True, independent_label_review_completed=True,
        extraction_review_completed=True, corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return rows, corpus, manifest_path


def _rewrite_fixture(rows, corpus, manifest_path):
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus_sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_identity_channel_preserves_destinations_but_omits_query_values_and_mailboxes():
    record = build_email_input("Routine agenda", sender="alice.private@example.com",
        supplementary_urls=["https://portal.example.com/login/123456?nonce=PRIVATEVALUE&id=999"])
    text = _identity_text(record)
    assert "example.com" in text and "/login/<number>" in text
    assert "nonce" in text and "id" in text
    assert "alice.private" not in text and "PRIVATEVALUE" not in text
    assert "123456" not in text and "999" not in text


def test_claimed_pass_vs_fail_does_not_change_model_features():
    common = dict(body="Please review meeting agenda", sender="person@example.com")
    good = build_email_input(**common, headers="Authentication-Results: relay.example; spf=pass dkim=pass dmarc=pass\n")
    bad = build_email_input(**common, headers="Authentication-Results: relay.example; spf=fail dkim=fail dmarc=fail\n")
    features = EmailEvidenceFeatures(min_df=1).fit([good, bad])
    before, after = features.transform([good]), features.transform([bad])
    assert (before != after).nnz == 0
    assert dict(zip(NUMERIC_NAMES, _numeric(good)))["trusted_authentication_missing"] == 1.


def test_observed_hostname_relations_are_facts_not_risk_scores():
    evidence = build_email_input("Please review", sender="person@corp.example.com",
        headers="Reply-To: person@other.example.net\nReturn-Path: person@corp.example.com\n",
        html_body='<a href="https://actual.example.net/login">https://displayed.example.com/login</a>')
    values = dict(zip(NUMERIC_NAMES, _numeric(evidence)))
    assert values["sender_reply_comparable"] == 1.
    assert values["sender_reply_host_equal"] == 0.
    assert values["sender_envelope_host_equal"] == 1.
    assert values["visible_target_hostname_disagreement_fraction"] == 1.
    assert not any("score" in name for name in NUMERIC_NAMES)


def test_body_only_input_stays_missing_identity_and_unsupported_when_oov():
    fitted = EmailEvidenceFeatures(min_df=1).fit([
        build_email_input("meeting agenda"), build_email_input("review document")])
    unknown = build_email_input("qzxvbrqzxvbr")
    content, identity = fitted.channel_support([unknown])
    assert not content[0] and not identity[0]
    assert not fitted.supported([unknown])[0]
    values = dict(zip(NUMERIC_NAMES, _numeric(unknown)))
    assert values["sender_domain_missing"] == 1.
    assert values["trusted_authentication_missing"] == 1.


def test_derived_contract_tampering_and_metadata_in_features_are_rejected():
    record = build_email_input("meeting agenda")
    features = EmailEvidenceFeatures(min_df=1).fit([record])
    for field, value in (("trusted_authentication", {"verified": True, "spf": "pass"}),
                         ("source", "Enron"), ("task_label", "phishing")):
        forged = dict(record, **{field: value})
        with pytest.raises(ValueError, match="differs"):
            features.transform([forged])


def test_shared_contract_serialization_keeps_inference_transform_identical(tmp_path):
    x = [build_email_input("review meeting agenda", sender="person@example.com"),
         build_email_input("verify transfer now", supplementary_urls=["https://other.example.net/login"])]
    estimator = email_evidence_estimator(min_df=1).fit(x, [0, 1])
    before = predict_email_evidence_probabilities(estimator, x)
    target = tmp_path / "fixture_model.pkl"
    joblib.dump(estimator, target)
    restored = joblib.load(target)
    np.testing.assert_array_equal(before, predict_email_evidence_probabilities(restored, x))
    assert np.isnan(predict_email_evidence_probabilities(restored, [build_email_input("qzxvbrqzxvbr")])[0])


def test_reviewed_admission_accepts_only_hash_bound_source_time_fixture(tmp_path):
    rows, corpus, manifest = _fixture_corpus(tmp_path)
    loaded, _, summary = load_reviewed_email_evidence(corpus, manifest)
    assert len(loaded) == len(rows)
    assert summary["source_test_disjoint"] and summary["time_test_strictly_later"]
    assert summary["verified_receiver_authentication_rows"] == 0


@pytest.mark.parametrize("mutation", ["unknown_label", "unchecked_extraction", "same_reviewers", "untrusted_time", "source_as_class", "duplicate_input", "reuse_unapproved", "duplicate_parse_notes"])
def test_admission_rejects_invalid_review_and_source_evidence(tmp_path, mutation):
    rows, corpus, manifest = _fixture_corpus(tmp_path)
    if mutation == "unknown_label": rows[0]["task_label"] = "spam"
    elif mutation == "unchecked_extraction": rows[0]["extraction_verified"] = "true"
    elif mutation == "same_reviewers": rows[0]["second_reviewer_id"] = rows[0]["reviewer_id"]
    elif mutation == "untrusted_time": rows[0]["time_authority"] = "sender_date"
    elif mutation == "source_as_class":
        for row in rows:
            if row["split"] == "train": row["source"] += "-" + row["task_label"]
    elif mutation == "duplicate_input":
        rows[1]["email_input"] = copy.deepcopy(rows[0]["email_input"])
        rows[1]["input_contract_sha256"] = rows[0]["input_contract_sha256"]
    elif mutation == "reuse_unapproved": rows[0]["training_reuse_permitted"] = False
    elif mutation == "duplicate_parse_notes":
        original = rows[0]["email_input"]
        kwargs = {field: original[field] for field in ("subject", "sender", "headers", "html_body", "supplementary_urls")}
        rows[1]["email_input"] = build_email_input(original["body"], **kwargs,
            parsing_info={"text_complete": True, "notes": ["Different provenance does not make a new email"]})
        rows[1]["input_contract_sha256"] = contract_sha256(rows[1]["email_input"])
    _rewrite_fixture(rows, corpus, manifest)
    with pytest.raises(ValueError):
        load_reviewed_email_evidence(corpus, manifest)


def test_unapproved_corpus_rejected_before_output_or_fit(monkeypatch, tmp_path):
    _, corpus, manifest_path = _fixture_corpus(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["approved_for_training"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(trainer, "email_evidence_estimator", lambda **kwargs: pytest.fail("Unapproved data reached training"))
    target = tmp_path / "benchmarks" / "rejected"
    with pytest.raises(ValueError, match="approved_for_training"):
        trainer.train_candidate(corpus, manifest_path, target)
    assert not target.exists()


def test_synthetic_training_exports_unapproved_models_and_separate_source_time_metrics(monkeypatch, tmp_path):
    """Fits tiny artificial fixtures; does not demonstrate detection quality."""
    rows, corpus, manifest = _fixture_corpus(tmp_path)
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    target = tmp_path / "benchmarks" / "fixture_run"
    result = trainer.train_candidate(corpus, manifest, target)
    assert result["complete"] is True and result["approved_for_deployment"] is False
    assert result["deployment_generalization_established"] is False
    assert result["calibration_fitted"] is False
    assert set(result["models"]) == {"word_only", "text_only", "structured"}
    for name in result["models"]:
        assert set(result["models"][name]["splits"]) == {"val", "source_test", "time_test"}
        assert set(result["models"][name]["splits"]["source_test"]["per_source"]) == {"origin-c", "origin-d"}
        artifact = joblib.load(target / (name + ".pkl"))
        assert artifact["schema_version"] == 3
        assert artifact["approved_for_deployment"] is False
        assert artifact["input_kind"] == "structured_email_evidence_v1"
        assert artifact["input_contract_version"] == "email-evidence-v1"
        assert artifact["contract_source_hashes"] == contract_source_hashes()
    predictions = [json.loads(line) for line in (target / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 3 * sum(row["split"] != "train" for row in rows)
    assert all(row["availability"]["trusted_authentication"] is False for row in predictions)
