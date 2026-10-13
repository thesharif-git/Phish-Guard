"""Small, bounded Email representation candidates; never a deployment approval.

The caller must fit these transforms on training rows only. Keeping the
representation inside an estimator also makes its saved inference transform
identical to training. The existing live transform is deliberately unchanged.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.feature_extractor import extract_features
from utils.text_sanitizer import clean_and_mask_email_text


class CandidateEmailFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, *, character_features=False, structural_features=False,
                 repaired_text=False, word_limit=12000, character_limit=16000,
                 min_df=3):
        self.character_features = character_features
        self.structural_features = structural_features
        self.repaired_text = repaired_text
        self.word_limit = word_limit
        self.character_limit = character_limit
        self.min_df = min_df

    def canonicalize(self, texts):
        if self.repaired_text:
            from utils.email_candidate_text import canonical_email_candidate_text
            return [canonical_email_candidate_text(text) for text in texts]
        return [clean_and_mask_email_text(text) for text in texts]

    def fit(self, texts, y=None):
        canonical = self.canonicalize(texts)
        self.word_vectorizer_ = TfidfVectorizer(
            max_features=self.word_limit, min_df=self.min_df,
            ngram_range=(1, 2), sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b|<[a-zA-Z]+>', dtype=np.float32)
        self.word_vectorizer_.fit(canonical)
        if self.character_features:
            self.character_vectorizer_ = TfidfVectorizer(
                analyzer='char_wb', ngram_range=(3, 5), min_df=self.min_df,
                max_features=self.character_limit, sublinear_tf=True,
                dtype=np.float32)
            self.character_vectorizer_.fit(canonical)
        return self

    def transform(self, texts):
        texts = list(texts)
        canonical = self.canonicalize(texts)
        blocks = [self.word_vectorizer_.transform(canonical)]
        if self.character_features:
            blocks.append(self.character_vectorizer_.transform(canonical))
        if self.structural_features:
            blocks.append(csr_matrix(extract_features(texts)))
        return hstack(blocks, format='csr', dtype=np.float32)

    def supported(self, texts):
        """Structural counters alone cannot manufacture vocabulary support."""
        canonical = self.canonicalize(texts)
        support = self.word_vectorizer_.transform(canonical).getnnz(axis=1) > 0
        if self.character_features:
            support |= self.character_vectorizer_.transform(canonical).getnnz(axis=1) > 0
        return support
