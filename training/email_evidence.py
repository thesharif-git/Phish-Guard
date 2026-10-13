"""Email evidence admission and full-message grouping, separate from ML inference."""
import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from utils.text_sanitizer import clean_and_mask_email_text


def full_message_tokens(text, *, canonicalizer=clean_and_mask_email_text):
    # Punctuation/spacing variants of the same complete body belong together.
    # Do not collapse rows: different headers or link destinations remain evidence.
    return re.findall(r'<[a-z]+>|\w+', canonicalizer(text))


def full_message_family(text, *, canonicalizer=clean_and_mask_email_text):
    tokens = full_message_tokens(text, canonicalizer=canonicalizer)
    return hashlib.sha256(' '.join(tokens).encode()).hexdigest() if tokens else None


def group_email_templates(records, threshold=.9, *, canonicalizer=clean_and_mask_email_text):
    """Exhaustive full-body trigram Jaccard for candidates sharing a trigram.

    Every positive-threshold match must share a trigram, so the inverted index
    is a lossless filter. No first-80-token limit or random approximate hash.
    Empty/short messages get separate groups and remain flagged for review.
    """
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    buckets, signatures, token_families = defaultdict(list), [], {}
    qualifying = []
    for i, row in enumerate(records):
        words = full_message_tokens(row.get('body', '') + ' ' + row.get('subject', ''), canonicalizer=canonicalizer)
        signature = {tuple(words[j:j+3]) for j in range(len(words)-2)}
        signatures.append(signature)
        family = full_message_family(row.get('body', '') + ' ' + row.get('subject', ''), canonicalizer=canonicalizer)
        if family and family in token_families:
            union(i, token_families[family])
        if family:
            token_families[family] = i
        if len(words) < 6 or not signature:
            continue
        candidates = set()
        for shingle in signature:
            candidates.update(buckets[shingle])
        for j in sorted(candidates):
            other = signatures[j]
            if min(len(signature), len(other)) < threshold * max(len(signature), len(other)):
                continue
            intersection = len(signature & other)
            similarity = intersection / (len(signature) + len(other) - intersection)
            if similarity >= threshold:
                union(i, j)
                qualifying.append({'first': records[j]['id'], 'second': row['id'], 'full_trigram_jaccard': similarity})
        for shingle in signature:
            buckets[shingle].append(i)
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row['id'])
    group_ids = {key: 'email-template-' + hashlib.sha256('|'.join(sorted(ids)).encode()).hexdigest()[:20] for key, ids in groups.items()}
    return [group_ids[find(i)] for i in range(len(records))], qualifying


def validate_email_evidence_partitions(frame, *, canonicalizer=clean_and_mask_email_text):
    """Reject task/source/time/provenance gaps before approved Email training."""
    required = {'raw_text', 'label', 'split', 'campaign_cluster_id', 'source', 'label_review_completed',
                'reviewer_id', 'label_evidence_reference', 'observed_at_utc', 'time_authority'}
    if not required.issubset(frame.columns):
        raise ValueError('Reviewed Email corpus lacks evidence/source/time columns')
    if frame[sorted(required)].isna().any().any():
        raise ValueError('Missing evidence/source/time annotation')
    allowed = {'train', 'val', 'source_test', 'time_test'}
    if set(frame.split) != allowed or not frame.label.isin([0, 1]).all():
        raise ValueError('Require train, val, source_test and time_test partitions with task labels 0/1')
    reviewed = frame.label_review_completed.map(lambda value: str(value).casefold() in ('true', '1'))
    if not reviewed.all() or frame.reviewer_id.astype(str).str.strip().eq('').any() or frame.label_evidence_reference.astype(str).str.strip().eq('').any():
        raise ValueError('Every Email label requires an identifiable completed review and evidence reference')
    for split in allowed:
        if set(frame.loc[frame.split == split, 'label']) != {0, 1}:
            raise ValueError('Both phishing and nonmalicious emails are required in every partition')
    train_val = frame[frame.split.isin(['train', 'val'])]
    source_test = frame[frame.split == 'source_test']
    if set(train_val.source) & set(source_test.source):
        raise ValueError('Source-test origins overlap training/calibration origins')
    for label in (0, 1):
        if frame[(frame.split == 'train') & (frame.label == label)].source.nunique() < 2:
            raise ValueError('Training each class requires at least two independent origins; do not encode source as class')
    groups = frame.groupby('campaign_cluster_id').split.nunique()
    if (groups > 1).any():
        raise ValueError('Full-message template groups cross partitions')
    families = frame.raw_text.map(lambda text: full_message_family(text, canonicalizer=canonicalizer))
    if frame.assign(_family=families).groupby('_family').split.nunique().gt(1).any():
        raise ValueError('Full-message content variants cross partitions')
    # Recompute near-template groups; do not trust manually supplied campaign IDs.
    records = [{'id': str(i), 'body': str(row.raw_text), 'subject': ''} for i, row in frame.iterrows()]
    computed, _ = group_email_templates(records, canonicalizer=canonicalizer)
    if frame.assign(_computed=computed).groupby('_computed').split.nunique().gt(1).any():
        raise ValueError('Full-message near templates cross partitions')
    def timestamp(value):
        date = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if date.tzinfo is None:
            raise ValueError('Time-separated validation requires timezone-aware observations')
        return date.astimezone(timezone.utc)
    if not frame.time_authority.isin(['trusted_receiver', 'publisher_observation']).all():
        raise ValueError('Sender Date or author-claimed year cannot establish trusted time separation')
    times = frame.observed_at_utc.map(timestamp)
    earlier = times[frame.split.isin(['train', 'val'])]
    later = times[frame.split == 'time_test']
    if later.min() <= earlier.max():
        raise ValueError('Time-test observations must strictly follow training/calibration observations')
    return {'source_test_disjoint': True, 'time_test_strictly_later': True, 'full_message_overlap_checked': True}
