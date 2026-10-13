"""Fold-owned Email preprocessing; no production artifacts are overwritten."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from utils.text_sanitizer import clean_and_mask_email_text
from utils.feature_extractor import extract_features


class EmailFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, max_features=12000, min_df=3):
        self.max_features = max_features
        self.min_df = min_df

    def fit(self, texts, y=None):
        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features, min_df=self.min_df,
            ngram_range=(1, 2), sublinear_tf=True, token_pattern=r'(?u)\b\w+\b|<[a-zA-Z]+>')
        self.vectorizer_.fit([clean_and_mask_email_text(t) for t in texts])
        return self

    def transform(self, texts):
        text_features = self.vectorizer_.transform([clean_and_mask_email_text(t) for t in texts])
        return hstack([text_features, csr_matrix(extract_features(texts))]).tocsr()


def email_estimator(classifier, *, min_df=3):
    return Pipeline([('features', EmailFeatures(min_df=min_df)), ('classifier', classifier)])


def require_reviewed_email_corpus(corpus_path, manifest_path):
    if manifest_path is None:
        raise ValueError('Email training requires a task-valid reviewed corpus manifest; legacy binary labels cannot be promoted silently')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('approved_for_training') is not True or manifest.get('label_scope') != 'phishing_bec_vs_nonmalicious':
        raise ValueError('Email corpus is not approved for the phishing/BEC detection task')
    if manifest.get('corpus_sha256') != hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest():
        raise ValueError('Reviewed corpus hash does not match training input')
    if manifest.get('independent_label_review_completed') is not True:
        raise ValueError('Independent task-label review is incomplete')
    return manifest


def validate_email_splits(df):
    required = {'split', 'label', 'raw_text', 'campaign_cluster_id'}
    if not required.issubset(df.columns) or not set(df['split']).issubset({'train', 'val', 'test'}):
        raise ValueError('Email corpus lacks the declared split/text/label/group contract')
    if not df['label'].isin([0, 1]).all() or df['raw_text'].isna().any() or df['campaign_cluster_id'].isna().any():
        raise ValueError('Invalid or missing email labels, text, or campaign groups')
    texts = df['raw_text'].map(clean_and_mask_email_text)
    if texts.str.len().le(5).any():
        raise ValueError('Insufficient statistical text in reviewed training corpus')
    if (df.assign(_masked=texts).groupby('_masked')['label'].nunique() > 1).any():
        raise ValueError('Conflicting labels for identical masked email text')
    if texts.duplicated().any():
        raise ValueError('Duplicate masked email text requires curation before training')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        first, second = df[df.split == a], df[df.split == b]
        if set(first.label) != {0, 1} or set(second.label) != {0, 1}:
            raise ValueError('Both classes are required in every evaluation partition')
        if set(first.campaign_cluster_id) & set(second.campaign_cluster_id):
            raise ValueError('Campaign groups overlap between email partitions')
        if set(texts.loc[first.index]) & set(texts.loc[second.index]):
            raise ValueError('Masked email text overlaps between partitions')


def grouped_email_folds(texts, labels, groups, n_splits=5):
    from sklearn.model_selection import GroupKFold
    count = min(n_splits, len(np.unique(groups)))
    if count < 2:
        raise ValueError('At least two independent email groups are required')
    splits = list(GroupKFold(n_splits=count).split(texts, labels, groups=groups))
    labels = np.asarray(labels)
    for train, calibration in splits:
        if set(labels[train]) != {0, 1} or set(labels[calibration]) != {0, 1}:
            raise ValueError('Every grouped training/calibration fold must contain both labels')
    return splits
