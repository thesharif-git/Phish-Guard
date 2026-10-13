"""Non-destructive, provenance-preserving Email review corpus. No label approval."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import email
import hashlib
import json
import mailbox
import re
from email import policy
from email.utils import parsedate_to_datetime
from collections import Counter
from bs4 import BeautifulSoup
from training.email_evidence import group_email_templates
from utils.email_parser import parse_eml_content, extract_email_anchor_links, visible_html_text

OUT = ROOT / 'data/email_evidence_review_20261003'
SNAPSHOTS = ROOT / 'data/email_root_audit_sources_20261003'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def record(source, form, reference, identity, label, **evidence):
    return {'id': source + '-' + sha(identity.encode())[:20], 'source': source, 'input_form': form,
        'original_reference': reference, 'source_claimed_label': label, 'training_label': None,
        'label_review_completed': False, 'reviewer_id': None, 'label_evidence_reference': None,
        'review_status': 'awaiting_task_and_extraction_review', 'approved_for_training': False,
        'sender': '', 'subject': '', 'body': '', 'html_body': '', 'headers': '', 'header_fields': [],
        'reported_authentication_headers': [], 'verified_authentication': {'spf': 'UNKNOWN', 'dkim': 'UNKNOWN', 'dmarc': 'UNKNOWN'},
        'trusted_receiver_authentication_available': False, 'urls': [], 'attachments': [],
        'observed_at_utc': None, 'time_authority': 'unverified', 'source_claimed_year': None, 'quality_flags': [], **evidence}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, duplicates, seen_raw = [], [], {}
    for year in (2024, 2025):
        path = ROOT / f'data/external_email_validation/nazario_2024_2025/phishing-{year}.mbox'
        archive_sha = sha(path.read_bytes())
        box = mailbox.mbox(path, create=False)
        for index in box.iterkeys():
            raw = box.get_bytes(index)
            raw_sha = sha(raw)
            reference = {'path': str(path.relative_to(ROOT)), 'archive_sha256': archive_sha, 'message_index': int(index), 'raw_message_sha256': raw_sha}
            if raw_sha in seen_raw:
                duplicates.append({'reference': reference, 'duplicate_of': seen_raw[raw_sha], 'criterion': 'identical full raw message bytes'})
                continue
            parsed = parse_eml_content(raw)
            message = email.message_from_bytes(raw, policy=policy.default)
            fields = [(k, str(v)) for k, v in message.items()]
            attachments = []
            for part in message.walk():
                if part.get_content_disposition() == 'attachment' or part.get_filename():
                    data = part.get_payload(decode=True)
                    attachments.append({'filename': part.get_filename(), 'mime_type': part.get_content_type(),
                        'bytes': len(data) if data else None, 'payload_sha256': sha(data) if data else None,
                        'payload_scanned': False, 'payload_preserved_in_raw_archive': True})
            flags = list(parsed.get('parsing', {}).get('errors', []))
            if parsed.get('parsing', {}).get('text_complete') is False:
                flags.append('incomplete_decoded_text')
            if parsed.get('parsing', {}).get('unscanned_parts'):
                flags.append('unscanned_mime_parts_present')
            if not parsed['body'].strip():
                flags.append('no_visible_email_body')
            row = record('nazario', 'raw_mime', reference, raw_sha, 'collector_claimed_phishing',
                sender=parsed['sender'], subject=parsed['subject'], body=parsed['body'], html_body=parsed['html_body'], headers=parsed['headers'],
                header_fields=fields, reported_authentication_headers=[{'name': k, 'value': v} for k, v in fields if k.lower() in ('authentication-results', 'received-spf')],
                reported_message_date=str(message.get('Date', '')), source_claimed_year=year, attachments=attachments,
                urls=extract_email_anchor_links(parsed['html_body']), parsing_info=parsed.get('parsing', {}), quality_flags=flags,
                evaluation_role='previously_inspected_development' if year == 2024 else 'reserved_unscored_future_review',
                reuse_terms_reference='data/external_email_validation/nazario_2024_2025/LICENSE.txt')
            seen_raw[raw_sha] = row['id']
            rows.append(row)
        box.close()
    manifest = json.loads((SNAPSHOTS / 'acquisition.json').read_text(encoding='utf-8'))
    for item in manifest:
        if not item['path'].startswith('brown_pages/'):
            continue
        path = SNAPSHOTS / item['path']
        soup = BeautifulSoup(path.read_bytes(), 'html.parser')
        intro, body = soup.select_one('.flex-intro'), soup.select_one('.flex-body')
        body_html = str(body) if body else ''
        text = visible_html_text(body_html)
        intro_text = intro.get_text('\n', strip=True) if intro else ''
        # Page titles often literally contain "Legitimate Email" or "Phishing".
        # Keep these as provenance, never use them as model subject/body.
        header_excerpt = BeautifulSoup(str(intro or ''), 'html.parser')
        for br in header_excerpt.find_all('br'):
            br.replace_with('\n')
        header_text = header_excerpt.get_text(' ', strip=False)
        header_text = re.sub(r'[^\S\n]+', ' ', header_text)
        original_fields = {name.lower(): value.strip() for name, value in re.findall(r'(?im)^\s*(From|Subject|Date|To):\s*([^\n]+)', header_text)}
        flags = []
        if 'subject' not in original_fields:
            flags.append('original_subject_unavailable_no_page_title_substitution')
        if len(text.strip()) < 40:
            flags.append('email_body_unavailable_or_image_only')
        if body and body.find('img'):
            flags.append('published_email_has_images_not_transcribed_or_scanned')
        if any(term in text.lower() for term in ('this is a legitimate', 'this is not a legitimate', 'this phishing', 'this scam')):
            flags.append('possible_publisher_commentary_in_extracted_body')
        row = record('brown_phish_bowl', 'published_email_excerpt', {'url': item['url'], 'path': str(path.relative_to(ROOT)),
            'snapshot_sha256': sha(path.read_bytes())}, item['url'], 'publisher_verified_nonphishing' if item['label'] == 'Legitimate Email' else 'publisher_reported_phishing',
            subject=original_fields.get('subject', ''), sender=original_fields.get('from', ''),
            body=text, html_body=body_html, published_header_excerpt=intro_text, publisher_page_title=item['title'],
            reported_message_date=original_fields.get('date', ''),
            publisher_adjudication=visible_html_text(str(soup.select_one('.flex-intro') or '')),
            observed_at_utc=item.get('published'), time_authority='publisher_observation', quality_flags=flags,
            urls=extract_email_anchor_links(body_html), evaluation_role='new_source_review_candidate',
            reuse_terms_reference='Publisher page copyright; permission for training reuse not established',
            original_mime_available=False, complete_headers_available=False)
        # Excerpts and actual MIME are distinct forms; never fabricate RFC headers from page commentary.
        rows.append(row)
    original_path = ROOT / 'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    from utils.jsonl import read_jsonl
    inventory = read_jsonl(ROOT / 'benchmarks/email_comparison_source_review/comparison_candidates.jsonl')
    for item in inventory:
        original_row = original[item['source_row_index']]
        body = original_row.get('Body') or ''
        html_body = body if any(t in body.lower() for t in ('<html', '<body', '<p>', '<a ')) else ''
        url_metadata = original_row.get('URL') or []
        url_metadata = url_metadata if isinstance(url_metadata, list) else [url_metadata]
        row = record('phishfuzzer_manual', 'author_original_json', {'path': str(original_path.relative_to(ROOT)),
            'source_sha256': sha(original_path.read_bytes()), 'row_index': item['source_row_index']}, item['review_id'], original_row.get('Type'),
            sender=original_row.get('Sender') or '', subject=original_row.get('Subject') or '',
            body=visible_html_text(html_body) if html_body else body, html_body=html_body,
            urls=extract_email_anchor_links(html_body), source_url_metadata=[u for u in url_metadata if isinstance(u, str)],
            source_attachment_metadata=original_row.get('File') or [], source_claimed_year=original_row.get('Year'),
            quality_flags=['original_mime_and_receiver_authentication_unavailable'], evaluation_role='previously_inspected_external_comparison',
            original_mime_available=False, complete_headers_available=False,
            reuse_terms_reference='Explicit dataset training reuse permission not established')
        rows.append(row)
    print('grouping ' + str(len(rows)) + ' preserved evidence records', flush=True)
    groups, pairs = group_email_templates(rows)
    for row, group in zip(rows, groups):
        row['campaign_cluster_id'] = group
    save_jsonl(OUT / 'review_corpus.jsonl', rows)
    save_jsonl(OUT / 'raw_duplicate_exclusions.jsonl', duplicates)
    save_jsonl(OUT / 'full_template_pairs.jsonl', pairs)
    # Reviewers receive source evidence without model predictions; disagreement
    # cannot be resolved by copying the model's answer into the label column.
    import csv
    with (OUT / 'annotation_review.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'source', 'source_claimed_label', 'evidence_reference',
            'reviewed_task_label', 'reviewer_id', 'second_reviewer_id', 'label_rationale', 'extraction_verified', 'adjudication_status'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'id': row['id'], 'source': row['source'], 'source_claimed_label': row['source_claimed_label'],
                             'evidence_reference': json.dumps(row['original_reference'])})
    result = {'schema_version': 1, 'purpose': 'Task/extraction review corpus, not a training corpus or certified benchmark',
        'records': len(rows), 'raw_duplicate_exclusions': len(duplicates), 'sources': dict(Counter(r['source'] for r in rows)),
        'source_claimed_labels': dict(Counter(str((r['source'], r['source_claimed_label'])) for r in rows)),
        'quality_flags': dict(Counter(f for r in rows for f in r['quality_flags'])), 'template_groups': len(set(groups)),
        'full_template_pairs': len(pairs), 'corpus_sha256': sha((OUT / 'review_corpus.jsonl').read_bytes()),
        'approved_for_training': False, 'independent_label_review_completed': False,
        'trusted_receiver_authentication_rows': 0, 'original_corpora_or_models_modified': False,
        'admission_blockers': ['Task labels and extraction require adjudication', 'No sufficient recent independently labelled raw benign email source',
            'Source/time test partitions must each contain both classes; do not pair a benign-only source with a malicious-only source',
            'Training reuse permissions must be established for publisher excerpts', 'Sender Date and claimed years are not trusted receiver observations'],
        'no_silent_exclusions': 'Malformed, short, non-English, attachment-only and image-only candidates are retained with quality flags, not inferred labels'}
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
