"""Publisher-label provenance for research; this module never approves deployment.

Published expert consensus is a legitimate annotation authority. It is not a
new independent review by this project, verified sender identity, or receiver
authentication. Day-precision publisher dates retain their actual precision.
"""

import csv
import hashlib
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit


SPAPHISH_V4_URL = "https://data.mendeley.com/datasets/hz2d6gz7pc/4"
SPAPHISH_V4_CORPUS_SHA256 = "fdd74842d0a19fd4332bd91f90b0bcb06e045ceb2b599051b4598b65055a9cc5"
SPAPHISH_V4_ROWS = 1395


def publisher_reception_day(value):
    """Parse the released DD/MM/YYYY day without inventing a UTC timestamp."""
    if not isinstance(value, str):
        raise TypeError("Publisher reception date must be a string")
    if not value.strip():
        return None
    if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
        raise ValueError("Expected the publisher's DD/MM/YYYY day format")
    return datetime.strptime(value, "%d/%m/%Y").date().isoformat()


def _spaphish_annotation(original, row_index):
    if original.get("Label") not in ("0", "1"):
        raise ValueError("Invalid released SpaPhish binary label")
    message_hash = original.get("hash", "")
    if not re.fullmatch(r"[0-9a-f]{64}", message_hash):
        raise ValueError("Invalid released SpaPhish message identifier")
    return {
        "id": "spaphish-v4-" + message_hash,
        "source_row_index": row_index,
        "task_label": "phishing" if original["Label"] == "1" else "legitimate",
        "label_authority_kind": "publisher_expert_consensus",
        "label_authority_reference": SPAPHISH_V4_URL,
        "label_authority_protocol": "Publisher documents three independent domain experts, majority vote, and fourth-expert adjudication",
        "project_independent_review_completed": False,
        "project_reviewer_ids": [],
        "published_label_available": True,
        "observed_day": publisher_reception_day(original.get("date", "")),
        "time_authority": "publisher_reported_reception_day",
        "time_precision": "day",
        "observed_at_utc": None,
        "receiver_time_verified": False,
        "trusted_receiver_authentication_available": False,
        "language_scope": "Spanish",
        "content_anonymized_by_publisher": True,
        "training_reuse_license": "CC BY 4.0",
        "approved_for_deployment": False,
        "english_generalization_proven": False,
    }


def annotation_epoch_summary(annotations):
    """Descriptive class/era association, never a held-out accuracy measurement."""
    year_counts = Counter()
    unknown = 0
    for row in annotations:
        day = row["observed_day"]
        if day is None:
            unknown += 1
        else:
            year_counts[(day[:4], row["task_label"])] += 1
    yearly = []
    for year in sorted({year for year, _ in year_counts}):
        positive = year_counts[(year, "phishing")]
        negative = year_counts[(year, "legitimate")]
        yearly.append(dict(year=year, phishing=positive, legitimate=negative,
                           phishing_fraction=positive / (positive + negative)))
    return dict(
        rows=len(annotations), missing_publisher_day=unknown,
        task_label_counts=dict(Counter(r["task_label"] for r in annotations)),
        year_label_counts=yearly,
        interpretation="Class prevalence changes with era. A random split can reward vocabulary associated with collection period. This summary does not establish leakage or causal attribution.",
    )


def reported_day_pilot_protocol(annotations):
    """Suggest an auditable chronological pilot, subject to template grouping.

    Years give wide boundaries, so unknown timezones do not justify invented
    instants. This is publisher-asserted chronology, not receiver-verified time.
    No actual split or model-selection approval is performed here.
    """
    counts = Counter()
    missing = 0
    for row in annotations:
        day = row["observed_day"]
        if day is None:
            missing += 1
            continue
        role = "train" if day < "2024-01-01" else "validation" if day < "2025-01-01" else "reported_day_test"
        counts[(role, row["task_label"])] += 1
    return {
        "scope": "Spanish publisher-reported-day chronological research pilot",
        "boundaries": {"train": "before 2024-01-01", "validation": "2024", "reported_day_test": "2025"},
        "pre_group_candidate_counts": {role: {label: counts[(role, label)] for label in ("legitimate", "phishing")}
            for role in ("train", "validation", "reported_day_test")},
        "missing_date_rows_unassigned": missing,
        "template_groups_must_remain_disjoint": True,
        "cross_boundary_templates_require_documented_assignment_or_exclusion": True,
        "dates_and_publisher_annotations_are_excluded_from_features": True,
        "independent_source_test_available": False,
        "receiver_verified_time_test": False,
        "english_generalization_evidence": False,
        "approved_for_deployment": False,
    }


def load_spaphish_publisher_annotations(directory):
    """Recognize only the verified released v4 bytes, not a caller's label claims."""
    path = Path(directory) / "corpus.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SPAPHISH_V4_CORPUS_SHA256:
        raise ValueError("SpaPhish v4 released corpus hash mismatch")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        originals = list(csv.DictReader(handle, delimiter=";"))
    if len(originals) != SPAPHISH_V4_ROWS:
        raise ValueError("SpaPhish v4 released row count mismatch")
    annotations = [_spaphish_annotation(original, index) for index, original in enumerate(originals)]
    if len({row["id"] for row in annotations}) != len(annotations):
        raise ValueError("Duplicate released SpaPhish identifiers")
    return {
        "source_url": SPAPHISH_V4_URL,
        "source_corpus_sha256": SPAPHISH_V4_CORPUS_SHA256,
        "research_annotation_authority": "verified-release publisher expert consensus",
        "project_independent_review_completed": False,
        "approved_for_deployment": False,
        "annotations": annotations,
        "epoch_summary": annotation_epoch_summary(annotations),
        "chronological_pilot": reported_day_pilot_protocol(annotations),
    }


def brown_publisher_verdict(label, url):
    """Map documented publisher verdicts for evaluation, not project review.

    The caller must bind the actual snapshot/extraction to publisher provenance.
    This function does not establish completeness, license, authenticity or a
    fresh holdout. Bare phishing-like words in page titles are never labels.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "it.brown.edu" or not parsed.path.startswith("/phish-bowl-alerts/"):
        raise ValueError("Unexpected Brown publisher evidence URL")
    mapping = {"publisher_reported_phishing": "phishing", "publisher_verified_nonphishing": "legitimate"}
    if label not in mapping:
        raise ValueError("A recorded Brown publisher verdict is required")
    return dict(task_label=mapping[label], label_authority_kind="publisher_security_team_verdict",
        label_authority_reference=url, project_independent_review_completed=False,
        project_reviewer_ids=[], approved_for_deployment=False)
