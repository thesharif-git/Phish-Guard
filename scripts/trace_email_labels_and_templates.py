"""Trace observed labels to upstream raw mail; never relabel or train."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import csv
import email
import hashlib
import json
import re
import tarfile
from email import policy
from collections import defaultdict
from utils.text_sanitizer import clean_and_mask_email_text
from utils.email_parser import parse_eml_content, visible_html_text

OUT = ROOT / 'benchmarks/email_generalization_root_20261003'
SOURCES = ROOT / 'data/email_root_audit_sources_20261003'


def signature(text):
    return hashlib.sha256(' '.join(re.findall(r'\w+', text.lower())).encode()).hexdigest()


def shingles(text):
    words = re.findall(r'\w+|<[a-z]+>', clean_and_mask_email_text(text))
    return {tuple(words[i:i+3]) for i in range(len(words)-2)}


def main():
    csv.field_size_limit(2**31-1)
    with (ROOT / 'data/clean_email_corpus.csv').open(encoding='utf-8', newline='') as f:
        corpus = list(csv.DictReader(f))
    index = defaultdict(list)
    for i, r in enumerate(corpus):
        index[signature(r['raw_text'])].append(i)
    matches = []
    for archive in sorted(SOURCES.glob('spamassassin_*.tar.bz2')):
        # Inspect regular members without extracting archive paths or executing attachments.
        with tarfile.open(archive, 'r:bz2') as tar:
            for member in tar:
                if not member.isfile() or member.size > 2*1024*1024:
                    continue
                raw = tar.extractfile(member).read()
                message = email.message_from_bytes(raw, policy=policy.default)
                parsed = parse_eml_content(raw)
                variants = [parsed['body'], parsed.get('html_body', '')]
                for part in message.walk():
                    if part.get_content_maintype() == 'text':
                        b = part.get_payload(decode=True)
                        if b:
                            try:
                                s = b.decode(part.get_content_charset() or 'utf-8', errors='replace')
                            except LookupError:
                                s = b.decode('utf-8', errors='replace')
                            variants += [s, visible_html_text(s) if part.get_content_type() == 'text/html' else s]
                found = set()
                for text in variants:
                    for value in (text, text + ' ' + parsed['subject'], parsed['subject'] + ' ' + text):
                        found.update(index.get(signature(value), []))
                for row in sorted(found):
                    r = corpus[row]
                    matches.append({'legacy_row': row, 'legacy_label': int(r['label']), 'split': r['split'],
                        'source': r['source'], 'upstream_archive': archive.name, 'upstream_member': member.name,
                        'upstream_raw_sha256': hashlib.sha256(raw).hexdigest(), 'upstream_label': 'spam (not a phishing-specific annotation)',
                        'reported_date': str(message.get('Date', '')), 'subject': parsed['subject'],
                        'match': 'Exact full alphanumeric token sequence of a decoded body variant; not approximate or model-selected'})
    (OUT / 'upstream_label_matches.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in matches), encoding='utf-8')
    unique = {r['legacy_row'] for r in matches}
    from utils.jsonl import read_jsonl
    near = read_jsonl(OUT / 'nearest_training_messages.jsonl')
    reviewed = []
    for r in near:
        if r['scope'] != 'internal_test' or r['trigram_cosine'] < .95:
            continue
        first, second = corpus[int(r['row'])], corpus[r['train_row']]
        a, b = shingles(first['raw_text']), shingles(second['raw_text'])
        jaccard = len(a & b)/len(a | b) if a | b else 0.
        reviewed.append(dict(r, full_unique_trigram_jaccard=jaccard, test_label=int(first['label']),
            test_group=first['campaign_cluster_id'], same_declared_group=first['campaign_cluster_id'] == second['campaign_cluster_id'],
            candidate_status='high_full_message_overlap_requires_campaign_review' if jaccard >= .9 else 'projected_similarity_only',
            test_preview=first['raw_text'][:1000], train_preview=second['raw_text'][:1000]))
    (OUT / 'full_message_overlap_review.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in reviewed), encoding='utf-8')
    result = {'complete': True, 'upstream_matched_unique_rows': len(unique),
        'upstream_spam_tagged_phishing_unique_rows': len({r['legacy_row'] for r in matches if r['legacy_label'] == 1}),
        'upstream_spam_tagged_phishing_test_rows': len({r['legacy_row'] for r in matches if r['legacy_label'] == 1 and r['split'] == 'test'}),
        'upstream_status': 'Spam membership is verified; actual phishing-vs-nonmalicious labels still require review. Not all spam is benign or phishing.',
        'high_cosine_reviewed_full_text': len(reviewed), 'full_unique_trigram_jaccard_ge_0_9': sum(r['full_unique_trigram_jaccard'] >= .9 for r in reviewed),
        'declared_cross_split_groups_in_high_jaccard': sum(not r['same_declared_group'] and r['full_unique_trigram_jaccard'] >= .9 for r in reviewed),
        'corpus_changed': False}
    (OUT / 'provenance_and_overlap.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
