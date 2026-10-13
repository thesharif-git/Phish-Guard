import re
import numpy as np
import pandas as pd
from typing import Union, List


def clean_text(text: str) -> str:
    """
    Cleans raw email text for TF-IDF vectorization:
    - Lowercases text
    - Strips HTML tags
    - Strips URLs
    - Keeps only alphanumeric characters and spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_features_single(text: str) -> np.ndarray:
    """
    Extracts 16 structural/behavioral numerical features from a single raw text string.
    Returns shape (1, 16) array.
    """
    t = text if isinstance(text, str) else ""
    words = t.split()
    word_count = len(words)
    text_len = len(t)
    avg_word_len = text_len / max(word_count, 1)

    features = [
        # URL signals
        len(re.findall(r'https?://', t)),
        int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', t))),
        int(bool(re.search(r'\.(xyz|biz|info|click|link|top|tk|pw|gq|ml)\b', t, re.I))),
        int(bool(re.search(r'bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly', t, re.I))),
        # Urgency / manipulation signals
        len(re.findall(r'urgent|verify|suspended|click here|act now|expires|limited time|immediate|unauthorized|confirm|update your', t, re.I)),
        t.count('!'),
        t.count('$'),
        len(re.findall(r'\b[A-Z]{3,}\b', t)),
        # Brand impersonation signals
        len(re.findall(r'paypal|microsoft|apple|amazon|google|netflix|bank of america|docusign|dropbox', t, re.I)),
        # Structure signals
        int(bool(re.search(r'<html|<body|<table|<div', t, re.I))),
        len(re.findall(r'<a\s+href', t, re.I)),
        text_len,
        word_count,
        avg_word_len,
        # Punctuation / style signals
        t.count('?'),
        t.count(','),
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)


def extract_features(texts: Union[pd.Series, List[str], str]) -> np.ndarray:
    """
    Unified feature extractor supporting pandas Series, list of strings, or single string.
    Returns 2D numpy array of shape (N, 16).
    """
    if isinstance(texts, str):
        return extract_features_single(texts)
    
    if isinstance(texts, pd.Series):
        s_texts = texts.fillna('').astype(str)
    else:
        s_texts = pd.Series([str(t) if t is not None else '' for t in texts])

    df = pd.DataFrame({'text': s_texts})
    features = pd.DataFrame()

    # URL signals
    features['url_count']          = df['text'].str.count(r'https?://')
    features['has_ip_url']         = df['text'].str.contains(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', regex=True).astype(int)
    features['has_suspicious_tld'] = df['text'].str.contains(r'\.(?:xyz|biz|info|click|link|top|tk|pw|gq|ml)\b', regex=True, case=False).astype(int)
    features['has_shortener']      = df['text'].str.contains(r'bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly', regex=True, case=False).astype(int)

    # Urgency / manipulation signals
    features['urgent_words']       = df['text'].str.count(r'(?i)urgent|verify|suspended|click here|act now|expires|limited time|immediate|unauthorized|confirm|update your')
    features['exclamation_count']  = df['text'].str.count(r'!')
    features['dollar_signs']       = df['text'].str.count(r'\$')
    features['all_caps_words']     = df['text'].apply(lambda t: len(re.findall(r'\b[A-Z]{3,}\b', t)))

    # Brand impersonation signals
    features['brand_keywords']     = df['text'].str.count(r'(?i)paypal|microsoft|apple|amazon|google|netflix|bank of america|docusign|dropbox')

    # Structure signals
    features['has_html']           = df['text'].str.contains(r'(?i)<html|<body|<table|<div', regex=True).astype(int)
    features['html_link_count']    = df['text'].str.count(r'(?i)<a\s+href')
    features['text_length']        = df['text'].str.len()
    features['word_count']         = df['text'].str.split().str.len()
    features['avg_word_len']       = features['text_length'] / np.maximum(features['word_count'], 1)

    # Punctuation / style signals
    features['question_marks']     = df['text'].str.count(r'\?')
    features['comma_count']        = df['text'].str.count(r',')

    return features.fillna(0).values.astype(np.float32)
