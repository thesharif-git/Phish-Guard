"""Unapproved, bounded Email evidence candidates and reviewed-data admission.

Identity relations describe supplied assertions, never authenticated identity.
Reported Authentication-Results are excluded: the v1 input contract has no
trusted receiver verification provider. These models are research artifacts.
"""

import hashlib
import ipaddress
import json
import re
import unicodedata
from collections import Counter
from email.utils import getaddresses
from importlib.metadata import version
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

import numpy as np
import pandas as pd
import tldextract
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from training.email_evidence import validate_email_evidence_partitions
from utils.email_candidate_text import canonical_email_candidate_text
from utils.email_input_contract import (
    EMAIL_INPUT_CONTRACT_VERSION, contract_sha256, evidence_sha256, validate_email_input,
)
from utils.text_sanitizer import CONTROL_CHARS
from utils.jsonl import read_jsonl

EMAIL_EVIDENCE_MODEL_INPUT_KIND = "structured_email_evidence_v1"
TASK_LABELS = {"phishing": 1, "bec": 1, "legitimate": 0}
MODEL_PARAMS = dict(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced",
                    max_iter=500, random_state=20261004)
PSL = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None,
                          include_psl_private_domains=True)
NUMERIC_NAMES = (
    "body_missing", "subject_missing", "sender_assertion_missing", "headers_missing",
    "html_missing", "sender_domain_missing", "reply_to_domain_missing",
    "envelope_domain_missing", "observed_links_missing", "trusted_authentication_missing",
    "sender_reply_comparable", "sender_reply_host_equal", "sender_reply_registered_equal",
    "sender_envelope_comparable", "sender_envelope_host_equal", "sender_envelope_registered_equal",
    "sender_links_comparable", "sender_link_registered_agreement_fraction",
    "visible_target_comparison_count_log", "visible_target_hostname_disagreement_fraction",
    "unique_url_count_log", "url_ip_host_fraction", "url_http_fraction",
    "url_userinfo_fraction", "url_punycode_fraction", "url_mean_hostname_length_log",
    "url_mean_path_length_log", "url_mean_query_key_count_log", "parse_incomplete",
)


def _host(value):
    value = CONTROL_CHARS.sub("", unicodedata.normalize("NFKC", value)).strip().rstrip(".").lower()
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        try:
            encoded = value.encode("idna").decode("ascii")
        except UnicodeError:
            return None
        if len(encoded) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                                    for label in encoded.split(".")):
            return None
        return encoded


def _mail_hosts(values):
    result = []
    for _, mailbox in getaddresses(values):
        if mailbox.count("@") == 1:
            local, domain = mailbox.rsplit("@", 1)
            hostname = _host(domain) if local else None
            if hostname and hostname not in result:
                result.append(hostname)
    return result


def _registered(hostname):
    if not hostname:
        return None
    try:
        ipaddress.ip_address(hostname)
        return None
    except ValueError:
        result = PSL(hostname)
        return result.top_domain_under_public_suffix or None


def _text(record):
    body = canonical_email_candidate_text(record["body"])
    if not body:
        body = canonical_email_candidate_text(record["html_body"])
    return body + " " + canonical_email_candidate_text(record["subject"])


def _url_parts(record):
    result = []
    for target in record["link_destinations"]:
        parsed = urlsplit(target)
        result.append((parsed, _host(parsed.hostname or "")))
    return result


def _identity_text(record):
    # No full mailbox/local part, URL query values, fragment or source metadata.
    tokens = []
    roles = [("sender", [record["sender"]]),
             ("reply", [r["value"] for r in record["reply_to"]]),
             ("envelope", [r["value"] for r in record["envelope_from"]])]
    for role, values in roles:
        tokens.extend(role + ":" + hostname for hostname in _mail_hosts(values))
    for parsed, hostname in _url_parts(record):
        if not hostname:
            continue
        # Numeric/UUID path identifiers are shapes; destination host and path
        # words remain observable evidence. Query keys remain; values do not.
        path = unicodedata.normalize("NFKC", unquote(parsed.path)).lower()
        path = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<identifier>", path)
        path = re.sub(r"\d+", "<number>", path)
        query_keys = sorted(set(key.casefold() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)))
        tokens.append("target:" + hostname + " path:" + path + " querykeys:" + " ".join(query_keys))
    return " ".join(tokens)


def _numeric(record):
    sender = _mail_hosts([record["sender"]])
    reply = _mail_hosts([r["value"] for r in record["reply_to"]])
    envelope = _mail_hosts([r["value"] for r in record["envelope_from"]])
    # Ambiguous/multiple sender assertions do not pretend to be one identity.
    sender_host = sender[0] if len(sender) == 1 else None
    sender_root = _registered(sender_host)
    urls = _url_parts(record)
    hosts = [hostname for _, hostname in urls if hostname]
    roots = [value for value in map(_registered, hosts) if value]
    def agreement(other):
        comparable = bool(sender_host and other)
        registered = [value for value in map(_registered, other) if value]
        return [float(comparable), float(comparable and sender_host in other),
                float(bool(sender_root and registered) and sender_root in registered)]
    visible_comparisons, disagreements = 0, 0
    for link in record["links"]:
        visible = (link.get("anchor_text") or "").strip()
        if visible.startswith("www."):
            visible = "https://" + visible
        if not re.fullmatch(r"https?://[^\s]+", visible, re.I):
            continue
        try:
            displayed = _host(urlsplit(visible).hostname or "")
            actual = _host(urlsplit(link["destination"]).hostname or "")
        except ValueError:
            continue
        if displayed and actual:
            visible_comparisons += 1
            disagreements += displayed != actual
    def fraction(values):
        return sum(values) / len(values) if values else 0.
    def average_log(values):
        return float(np.log1p(sum(values) / len(values))) if values else 0.
    ip_flags = []
    for hostname in hosts:
        try:
            ipaddress.ip_address(hostname)
            ip_flags.append(True)
        except ValueError:
            ip_flags.append(False)
    available = record["availability"]
    values = [float(not available[name]) for name in ("body", "subject", "sender", "headers", "html_body")]
    values += [float(not sender_host), float(not reply), float(not envelope), float(not urls),
               float(record["trusted_authentication"] is None)]
    values += agreement(reply) + agreement(envelope)
    values += [float(bool(sender_root and roots)), fraction([root == sender_root for root in roots]) if sender_root else 0.,
               float(np.log1p(visible_comparisons)), disagreements / visible_comparisons if visible_comparisons else 0.,
               float(np.log1p(len(urls))), fraction(ip_flags),
               fraction([parsed.scheme == "http" for parsed, _ in urls]),
               fraction([parsed.username is not None for parsed, _ in urls]),
               fraction([any(label.startswith("xn--") for label in host.split(".")) for host in hosts]),
               average_log([len(host) for host in hosts]),
               average_log([len(parsed.path) for parsed, _ in urls]),
               average_log([len(set(key for key, _ in parse_qsl(parsed.query, keep_blank_values=True))) for parsed, _ in urls]),
               float(record["parsing"]["completeness"] == "INCOMPLETE")]
    assert len(values) == len(NUMERIC_NAMES)
    return np.asarray(values, dtype=np.float32)


class EmailEvidenceFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, *, character_features=True, evidence_features=True,
                 word_limit=12000, character_limit=16000, identity_limit=8000, min_df=3):
        self.character_features = character_features
        self.evidence_features = evidence_features
        self.word_limit = word_limit
        self.character_limit = character_limit
        self.identity_limit = identity_limit
        self.min_df = min_df

    def fit(self, records, y=None):
        records = [validate_email_input(row) for row in records]
        texts = [_text(row) for row in records]
        self.word_vectorizer_ = TfidfVectorizer(max_features=self.word_limit, min_df=self.min_df,
            ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32,
            token_pattern=r"(?u)\b\w+\b|<[a-zA-Z]+>").fit(texts)
        if self.character_features:
            self.character_vectorizer_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                max_features=self.character_limit, min_df=self.min_df,
                sublinear_tf=True, dtype=np.float32).fit(texts)
        self.identity_vectorizer_ = None
        if self.evidence_features:
            identity = [_identity_text(row) for row in records]
            candidate = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                max_features=self.identity_limit, min_df=self.min_df,
                sublinear_tf=True, dtype=np.float32)
            try:
                self.identity_vectorizer_ = candidate.fit(identity)
            except ValueError as error:
                # Entirely missing/too rare identity remains missing, not a
                # fabricated vocabulary or a swallowed estimator error.
                if "empty vocabulary" not in str(error) and "After pruning, no terms remain" not in str(error):
                    raise
        return self

    def transform(self, records):
        records = [validate_email_input(row) for row in records]
        texts = [_text(row) for row in records]
        blocks = [self.word_vectorizer_.transform(texts)]
        if self.character_features:
            blocks.append(self.character_vectorizer_.transform(texts))
        if self.evidence_features:
            if self.identity_vectorizer_ is not None:
                blocks.append(self.identity_vectorizer_.transform([_identity_text(row) for row in records]))
            blocks.append(csr_matrix(np.asarray([_numeric(row) for row in records], dtype=np.float32).reshape(len(records), -1)))
        return hstack(blocks, format="csr", dtype=np.float32)

    def channel_support(self, records):
        records = [validate_email_input(row) for row in records]
        texts = [_text(row) for row in records]
        content = self.word_vectorizer_.transform(texts).getnnz(axis=1) > 0
        if self.character_features:
            content |= self.character_vectorizer_.transform(texts).getnnz(axis=1) > 0
        identity = np.zeros(len(records), dtype=bool)
        if self.evidence_features and self.identity_vectorizer_ is not None:
            identity = self.identity_vectorizer_.transform([_identity_text(row) for row in records]).getnnz(axis=1) > 0
        return content, identity

    def supported(self, records):
        content, identity = self.channel_support(records)
        return content | identity


def email_evidence_estimator(*, character_features=True, evidence_features=True, min_df=3):
    return Pipeline([("features", EmailEvidenceFeatures(character_features=character_features,
        evidence_features=evidence_features, min_df=min_df)),
        ("classifier", LogisticRegression(**MODEL_PARAMS))])


def predict_email_evidence_probabilities(model, records, batch_size=64):
    result = np.full(len(records), np.nan)
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        support = model.named_steps["features"].supported(batch)
        if support.any():
            values = model.predict_proba([row for row, ok in zip(batch, support) if ok])[:, 1]
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Invalid Email evidence candidate probabilities")
            result[start + np.flatnonzero(support)] = values
    return result


def contract_source_hashes(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    names = ("training/email_evidence_model.py", "training/email_evidence.py",
             "utils/email_input_contract.py", "utils/email_candidate_text.py",
             "utils/email_authentication.py", "utils/text_sanitizer.py", "utils/jsonl.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def library_versions():
    return {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "tldextract", "beautifulsoup4")}


def validate_evidence_records(records):
    """Pure admission checks; passing does not grant training/deployment approval."""
    if not isinstance(records, list) or not records:
        raise ValueError("Reviewed Email corpus is empty")
    seen_ids, seen_inputs, frame_rows = set(), {}, []
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in seen_ids:
            raise ValueError("Email record needs a unique nonempty id")
        seen_ids.add(row["id"])
        if not isinstance(row.get("task_label"), str) or row["task_label"] not in TASK_LABELS:
            raise ValueError("Only explicitly reviewed phishing, bec or legitimate task labels are allowed")
        for key in ("label_review_completed", "independent_review_completed", "extraction_verified", "training_reuse_permitted"):
            if row.get(key) is not True:
                raise ValueError("Every Email row needs completed review: " + key)
        for key in ("reviewer_id", "second_reviewer_id", "label_evidence_reference", "label_rationale", "campaign_cluster_id", "source"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("Email review metadata missing: " + key)
        if row["reviewer_id"].strip() == row["second_reviewer_id"].strip():
            raise ValueError("Independent Email reviewers must differ")
        evidence = validate_email_input(row.get("email_input"))
        fingerprint = contract_sha256(evidence)
        if row.get("input_contract_sha256") != fingerprint:
            raise ValueError("Email evidence contract hash mismatch")
        duplicate_key = evidence_sha256(evidence)
        if duplicate_key in seen_inputs:
            raise ValueError("Duplicate complete Email evidence inputs require curation; never drop/relabel silently")
        seen_inputs[duplicate_key] = row["id"]
        frame_rows.append(dict(raw_text=_text(evidence),
            label=TASK_LABELS[row["task_label"]], split=row.get("split"),
            campaign_cluster_id=row["campaign_cluster_id"], source=row["source"],
            label_review_completed=True, reviewer_id=row["reviewer_id"],
            label_evidence_reference=row["label_evidence_reference"],
            observed_at_utc=row.get("observed_at_utc"), time_authority=row.get("time_authority")))
    frame = pd.DataFrame(frame_rows)
    checked = validate_email_evidence_partitions(frame, canonicalizer=canonical_email_candidate_text)
    for (split, source), part in frame.groupby(["split", "source"]):
        if set(part.label) != {0, 1}:
            raise ValueError(f"Each {split} source must contain both task classes: {source}")
    summary = dict(checked, rows=len(records), task_label_counts=dict(Counter(row["task_label"] for row in records)),
        split_source_class_counts=frame.groupby(["split", "source", "label"]).size().reset_index(name="rows").to_dict("records"),
        verified_receiver_authentication_rows=0,
        limitations=["Admission checks verify recorded attestations/provenance; code cannot establish that humans actually reviewed labels",
                     "Current input contract has no trusted receiver authentication provider",
                     "No model or calibration is approved by passing data admission"])
    return summary


def load_reviewed_email_evidence(corpus_path, manifest_path):
    path = Path(corpus_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    requirements = {"schema_version": 3, "input_kind": EMAIL_EVIDENCE_MODEL_INPUT_KIND,
        "input_contract_version": EMAIL_INPUT_CONTRACT_VERSION,
        "label_scope": "phishing_bec_vs_nonmalicious", "approved_for_training": True,
        "independent_label_review_completed": True, "extraction_review_completed": True}
    for key, expected in requirements.items():
        if type(manifest.get(key)) is not type(expected) or manifest[key] != expected:
            raise ValueError("Reviewed Email manifest requirement missing: " + key)
    corpus_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("corpus_sha256") != corpus_hash:
        raise ValueError("Reviewed Email corpus hash mismatch")
    records = read_jsonl(path)
    return records, manifest, validate_evidence_records(records)
