"""Development comparison integrity; no saved model or corpus is loaded."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from scripts.compare_email_representations import (
    empirical_budget_threshold,
    metric,
    score_baseline,
    validate_new_output,
)
from training.email_text_candidates import CandidateEmailFeatures


@pytest.mark.parametrize("probability", [np.inf, -np.inf, -.001, 1.001])
def test_metrics_reject_invalid_returned_probability(probability):
    with pytest.raises(ValueError, match="Invalid metric probabilities"):
        metric([0], [probability])


@pytest.mark.parametrize("label", [-1, 2, .5, 1.9, np.nan])
def test_metrics_reject_non_binary_labels_without_truncation(label):
    with pytest.raises(ValueError):
        metric([label], [.3])


def test_metrics_reject_mismatched_rows():
    with pytest.raises(ValueError, match="matching shape"):
        metric([0, 1], [.2])


def test_missing_scores_keep_denominator_and_are_not_clean_predictions():
    result = metric([0, 1, 1], [.1, .9, np.nan])
    assert result["total_rows"] == 3
    assert result["scored_rows"] == 2
    assert result["unavailable_rows"] == 1
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["accuracy"] == 1.
    assert result["recall"] == 1.
    assert result["roc_auc"] == 1.


def test_all_missing_scores_do_not_manufacture_metrics():
    result = metric([0, 1], [np.nan, np.nan])
    assert result["total_rows"] == 2
    assert result["scored_rows"] == 0
    assert result["unavailable_rows"] == 2
    assert "accuracy" not in result
    assert "roc_auc" not in result


def test_validation_budget_handles_ties_conservatively():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([.8, .8, .4, .2, .1, .9])
    threshold = empirical_budget_threshold(y, p, budget=.25)
    assert threshold > .8
    assert (p[y == 0] >= threshold).sum() == 0
    assert (p[y == 1] >= threshold).sum() == 1


def test_validation_threshold_ignores_positive_score_changes():
    labels = [0, 0, 0, 0, 1, 1]
    first = empirical_budget_threshold(labels, [.9, .7, .4, .2, .01, .03], budget=.25)
    second = empirical_budget_threshold(labels, [.9, .7, .4, .2, .99, .98], budget=.25)
    assert first == second
    assert first > .7
    assert first < .9


def test_validation_threshold_discloses_missing_support_via_scored_budget():
    threshold = empirical_budget_threshold([0, 0, 0, 1], [.9, np.nan, .2, .99], budget=.5)
    assert .2 < threshold < .9


def test_validation_requires_scored_negatives():
    with pytest.raises(ValueError, match="No scored validation negatives"):
        empirical_budget_threshold([0, 1], [np.nan, .8])


class FakeVocabulary:
    def __init__(self, supported):
        self.supported = supported

    def transform(self, texts):
        return csr_matrix([[float(self.supported)] for _ in texts])


class FakeModel:
    def __init__(self, probability):
        self.probability = probability
        self.calls = 0

    def predict_proba(self, matrix):
        self.calls += 1
        return np.tile([1 - self.probability, self.probability], (matrix.shape[0], 1))


def test_baseline_all_unsupported_batch_never_calls_model():
    model = FakeModel(.9)
    result = score_baseline(model, FakeVocabulary(False), ["qzxvbr", "qzxvbr!!!"])
    assert np.isnan(result).all()
    assert model.calls == 0


@pytest.mark.parametrize("probability", [np.nan, np.inf, -.1, 1.1])
def test_baseline_invalid_predictions_are_errors_not_abstentions(probability):
    with pytest.raises(ValueError, match="Invalid baseline probabilities"):
        score_baseline(FakeModel(probability), FakeVocabulary(True), ["notice"])


def test_feature_transform_does_not_fit_validation_vocabulary():
    features = CandidateEmailFeatures(character_features=True, min_df=1)
    features.fit(["alpha notice", "beta notice"])
    words = dict(features.word_vectorizer_.vocabulary_)
    characters = dict(features.character_vectorizer_.vocabulary_)
    features.transform(["qzxvbr"])
    features.supported(["qzxvbr"])
    assert features.word_vectorizer_.vocabulary_ == words
    assert features.character_vectorizer_.vocabulary_ == characters
    assert "qzxvbr" not in words


def test_structural_counters_cannot_fabricate_text_support():
    features = CandidateEmailFeatures(structural_features=True, min_df=1)
    features.fit(["alpha notice", "beta notice"])
    matrix = features.transform(["qzxvbr!!!"])
    assert matrix[:, :len(features.word_vectorizer_.vocabulary_)].nnz == 0
    assert matrix[:, len(features.word_vectorizer_.vocabulary_):].nnz > 0
    assert not features.supported(["qzxvbr!!!"])[0]


def test_shared_vocabulary_is_identical_for_structural_ablation_pair():
    word = CandidateEmailFeatures(min_df=1).fit(["alpha notice", "beta notice"])
    structural = CandidateEmailFeatures(structural_features=True, min_df=1)
    structural.word_vectorizer_ = word.word_vectorizer_
    texts = ["alpha notice!!!", "<html><body>beta notice</body></html>"]
    left = word.transform(texts)
    right = structural.transform(texts)
    assert right.shape[1] == left.shape[1] + 16
    assert (left != right[:, :left.shape[1]]).nnz == 0


def test_character_support_is_real_even_without_word_vocabulary_match():
    features = CandidateEmailFeatures(character_features=True, min_df=1)
    features.fit(["alphabet notice", "alphabet meeting"])
    canonical = features.canonicalize(["alphabetical"])
    assert features.word_vectorizer_.transform(canonical).nnz == 0
    assert features.character_vectorizer_.transform(canonical).nnz > 0
    assert features.supported(["alphabetical"])[0]


def test_output_guard_requires_a_new_benchmark_child(monkeypatch, tmp_path):
    from scripts import compare_email_representations as experiment
    root = tmp_path / "project"
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True)
    monkeypatch.setattr(experiment, "ROOT", root)
    with pytest.raises(ValueError, match="within workspace benchmarks"):
        validate_new_output(root / "models" / "candidate")
    with pytest.raises(ValueError, match="child folder"):
        validate_new_output(benchmarks)
    candidate = benchmarks / "new_run"
    assert validate_new_output(candidate) == candidate.resolve()
    assert not candidate.exists()


def test_output_guard_preserves_previous_run(monkeypatch, tmp_path):
    from scripts import compare_email_representations as experiment
    root = tmp_path / "project"
    run = root / "benchmarks" / "completed_run"
    run.mkdir(parents=True)
    protected = run / "run_status.json"
    protected.write_text('{"stage":"completed"}', encoding="utf-8")
    before = protected.read_bytes()
    monkeypatch.setattr(experiment, "ROOT", root)
    with pytest.raises(RuntimeError, match="never overwrite"):
        validate_new_output(run)
    assert protected.read_bytes() == before
    assert set(p.name for p in run.iterdir()) == {"run_status.json"}


def test_cli_rejected_output_cannot_overwrite_its_status(tmp_path):
    root = Path(__file__).resolve().parents[1]
    protected = tmp_path / "run_status.json"
    protected.write_text('{"stage":"completed","sentinel":true}', encoding="utf-8")
    before = protected.read_bytes()
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "compare_email_representations.py"),
         "--output", str(tmp_path)],
        cwd=root, text=True, capture_output=True, timeout=45,
    )
    assert result.returncode != 0
    # A custom workspace basetemp can itself be under benchmarks. In that case
    # the nonempty-run guard rejects it; external paths fail the location guard.
    expected = "never overwrite" if tmp_path.resolve().is_relative_to((root / "benchmarks").resolve()) else "within workspace benchmarks"
    assert expected in result.stderr
    assert protected.read_bytes() == before
    assert set(p.name for p in tmp_path.iterdir()) == {"run_status.json"}


def test_spam_omission_covers_both_archives_deduplicates_ids_and_keeps_labels(monkeypatch, tmp_path):
    from scripts import compare_email_representations as experiment
    frame = pd.DataFrame({
        "split": ["train", "train", "train", "train", "val", "test", "train"],
        "label": [1, 1, 1, 0, 1, 1, 1],
        "campaign_cluster_id": ["a", "b", "c", "d", "e", "f", "g"],
    }, index=[10, 11, 12, 13, 14, 15, 16])
    before = frame.copy(deep=True)
    annotation = "spam (not a phishing-specific annotation)"

    def match(index, archive, *, upstream_label=annotation):
        return {
            "legacy_row": index,
            "legacy_label": int(frame.loc[index, "label"]),
            "split": frame.loc[index, "split"],
            "upstream_archive": archive,
            "upstream_label": upstream_label,
            "upstream_raw_sha256": f"{index}-{archive}",
        }

    records = [
        match(10, "spamassassin_spam.tar.bz2"),
        match(10, "spamassassin_spam2.tar.bz2"),  # Same row is omitted once.
        match(11, "spamassassin_spam2.tar.bz2"),
        match(12, "spamassassin_easy_ham.tar.bz2", upstream_label="ham (source annotation)"),
        match(13, "spamassassin_spam.tar.bz2"),  # Existing benign target is unchanged.
        match(14, "spamassassin_spam.tar.bz2"),  # Validation is untouched.
        match(15, "spamassassin_spam2.tar.bz2"),  # Test is untouched.
        match(16, "other_archive", upstream_label="unreviewed label"),
    ]
    (tmp_path / "full_message_overlap_review.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "verified_original_message_overlap.json").write_text("[]", encoding="utf-8")
    (tmp_path / "upstream_label_matches.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8",
    )
    monkeypatch.setattr(experiment, "AUDIT", tmp_path)
    overlap, uncertain = experiment.omission_manifest(frame)
    assert not overlap
    assert set(uncertain) == {10, 11}
    assert len(uncertain[10]) == 2  # Both provenance references remain available.
    assert all(reason["reason"] == "upstream_spam_not_phishing_adjudicated"
               for reasons in uncertain.values() for reason in reasons)
    reduced = frame[(frame.split == "train") & ~frame.index.isin(uncertain)]
    assert reduced.index.tolist() == [12, 13, 16]
    assert not set(uncertain) & set(frame[frame.split != "train"].index)
    pd.testing.assert_frame_equal(frame, before)
