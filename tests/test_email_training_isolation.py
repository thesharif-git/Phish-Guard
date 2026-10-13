"""Measure vocabulary leakage directly; never fit or overwrite real artifacts."""
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from training.email_pipeline import email_estimator, grouped_email_folds, require_reviewed_email_corpus, validate_email_splits


def test_calibration_fold_words_cannot_fit_that_folds_vocabulary():
    texts, labels, groups = [], [], []
    for group in range(4):
        for label in (0, 1):
            texts.append('Meeting document groupword' + chr(97 + group) + (' appointment' if label == 0 else ' verify request'))
            labels.append(label)
            groups.append(group)
    splits = grouped_email_folds(texts, np.array(labels), np.array(groups), n_splits=2)
    model = CalibratedClassifierCV(estimator=email_estimator(LogisticRegression(max_iter=100), min_df=1), cv=splits)
    model.fit(texts, labels)
    for calibrated, (train, heldout) in zip(model.calibrated_classifiers_, splits):
        vocabulary = calibrated.estimator.named_steps['features'].vectorizer_.vocabulary_
        for index in heldout:
            word = 'groupword' + chr(97 + groups[index])
            assert word not in vocabulary
        for index in train:
            assert 'groupword' + chr(97 + groups[index]) in vocabulary
    probabilities = model.predict_proba(texts)
    assert probabilities.shape == (8, 2) and np.isfinite(probabilities).all()


def test_unreviewed_or_stale_label_manifest_cannot_start_training(tmp_path):
    corpus = tmp_path / 'corpus.csv'
    corpus.write_text('fixture text')
    with pytest.raises(ValueError, match='manifest'):
        require_reviewed_email_corpus(corpus, None)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'approved_for_training': True, 'label_scope': 'phishing_bec_vs_nonmalicious',
                                    'independent_label_review_completed': True, 'corpus_sha256': 'stale'}))
    with pytest.raises(ValueError, match='hash'):
        require_reviewed_email_corpus(corpus, manifest)
    value = json.loads(manifest.read_text())
    value['corpus_sha256'] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    value['independent_label_review_completed'] = False
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='review'):
        require_reviewed_email_corpus(corpus, manifest)


@pytest.mark.parametrize('collision', ['campaign', 'text'])
def test_holdout_group_or_text_overlap_is_rejected(collision):
    rows = []
    for split in ('train', 'val', 'test'):
        for label in (0, 1):
            rows.append(dict(split=split, label=label, raw_text=f'Message body {split} class {label}',
                             campaign_cluster_id=f'{split}-{label}'))
    if collision == 'campaign':
        rows[2]['campaign_cluster_id'] = rows[0]['campaign_cluster_id']
    else:
        rows[2]['raw_text'] = rows[0]['raw_text']
    with pytest.raises(ValueError, match='overlap|Duplicate'):
        validate_email_splits(pd.DataFrame(rows))
