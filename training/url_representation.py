"""A common textual surface for bare and scheme-qualified web URLs.

The raw URL stays in corpus records and security evidence. HTTP versus HTTPS is
not used as a lexical phishing shortcut in this baseline classifier.
"""
import re

from sklearn.base import BaseEstimator, TransformerMixin


class URLSurfaceView(TransformerMixin, BaseEstimator):
    def fit(self, inputs, y=None):
        return self

    def transform(self, inputs):
        return [re.sub(r"^https?://", "", value.strip(), flags=re.IGNORECASE)
                for value in inputs]
