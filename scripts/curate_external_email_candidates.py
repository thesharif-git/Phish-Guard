"""Inventory external full messages without fitting, relabeling, or scoring them.

Exact duplicate/overlap exclusions are reproducible. SimHash and size/encoding
outliers are review flags, not invented security labels or automatic deletions.
"""
import ast
import hashlib
import json
import mailbox
from collections import Counter, defaultdict
from email.utils import parsedate_to_datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from utils.email_parser import parse_eml_content, extract_email_anchor_links
from utils.text_sanitizer import clean_and_mask_email_text
from utils.feature_extractor import extract_features_single

OUT = ROOT / 'data/external_email_validation/nazario_2024_2025'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def decoded_mime_defects(message):
    # Base64 defects are populated lazily, during transfer decoding. Inspecting
    # .defects before decoding silently misses malformed payloads.
    for part in message.walk():
        if not part.is_multipart():
            part.get_payload(decode=True)
    return sorted({type(d).__name__ for part in message.walk() for d in part.defects})


def main():
    acquisition = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert acquisition['complete']
    for item in acquisition['files']:
        assert sha((ROOT / item['path']).read_bytes()) == item['sha256']
    corpus_path = ROOT / 'data/clean_email_corpus.csv'
    original_hash = sha(corpus_path.read_bytes())
    old = pd.read_csv(corpus_path, keep_default_na=False)
    old_masked = defaultdict(list)
    for idx, text in enumerate(old.masked_text):
        old_masked[text].append(idx)
    # Reuse only the exact fingerprint implementation; do not import/run the
    # combined dataset curation module or its URL/Domain workflow.
    tree = ast.parse((ROOT / 'scripts/clean_and_curate_master_datasets.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_simhash_fast')
    namespace = {'np': np, 'hashlib': hashlib}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'email_fingerprint', 'exec'), namespace)
    fingerprint = namespace['compute_simhash_fast']
    reference = [fingerprint(text) for text in old.masked_text]
    buckets = [defaultdict(set) for _ in range(8)]
    for idx, value in enumerate(reference):
        if value:
            for chunk in range(8):
                buckets[chunk][(value >> (chunk * 8)) & 255].add(idx)
    # Development/test distributions cannot tune quality thresholds.
    training_reference = old[old['split'] == 'train']
    length_low, length_high = np.quantile(training_reference.raw_text.str.len(), [.001, .999])
    raw_seen, model_seen, lexical_seen, candidate_hashes = {}, {}, {}, []
    records = []
    summary = {}
    for year in (2024, 2025):
        path = OUT / f'phishing-{year}.mbox'
        box = mailbox.mbox(path, create=False)
        counts = Counter()
        for key in box.iterkeys():
            raw = box.get_bytes(key, from_=False)
            raw_hash = sha(raw)
            record = {'source': 'nazario', 'source_mailbox_year': year,
                      'source_path': path.relative_to(ROOT).as_posix(), 'source_message_key': int(key),
                      'source_message_sha256': raw_hash, 'source_reported_label': 'phishing',
                      'independent_security_label': None, 'review_status': 'UNREVIEWED',
                      'intended_role': 'development_challenge' if year == 2024 else 'reserved_external_candidate',
                      'exclusion_reasons': [], 'review_flags': [], 'model_predictions_computed': False}
            if raw_hash in raw_seen:
                record['exclusion_reasons'].append('exact_message_duplicate')
                record['duplicate_of'] = raw_seen[raw_hash]
            else:
                raw_seen[raw_hash] = f'{year}/{key}'
            value = 0
            try:
                message = box.get_message(key)
                parsed = parse_eml_content(raw)
                body, subject = parsed['body'], str(parsed['subject'])
                text = body + ' ' + subject
                masked = clean_and_mask_email_text(text)
                masked_body = clean_and_mask_email_text(body)
                structural_features = extract_features_single(text)[0].tolist()
                # The model uses BOTH masked text and raw-text structural values.
                # Masked-text identity alone must not erase distinct structures.
                text_hash = sha(json.dumps([masked, structural_features], ensure_ascii=False).encode())
                lexical_hash = sha(masked.encode())
                record.update({'model_input_sha256': text_hash, 'body_characters': len(body),
                               'model_input_characters': len(masked), 'raw_bytes': len(raw),
                               'html_available': bool(parsed['html_body']),
                               'absolute_anchor_destinations': len(extract_email_anchor_links(parsed['html_body'])),
                               'image_parts': len(parsed['attachments']),
                               'attachment_mime_types': [p.get_content_type() for p in message.walk()
                                   if p.get_content_disposition() == 'attachment'],
                               'mime_defects': decoded_mime_defects(message)})
                if len(masked) <= 5:
                    record['exclusion_reasons'].append('insufficient_text_for_statistical_model')
                    record['review_flags'].append('may_be_valid_attachment_only_message')
                if masked and text_hash in model_seen:
                    record['exclusion_reasons'].append('exact_model_input_contract_duplicate')
                    record['same_model_input_as'] = model_seen[text_hash]
                elif masked:
                    model_seen[text_hash] = f'{year}/{key}'
                if masked and lexical_hash in lexical_seen:
                    record['review_flags'].append('masked_lexical_text_duplicate')
                elif masked:
                    lexical_seen[lexical_hash] = f'{year}/{key}'
                overlaps = sorted(set(old_masked.get(masked, []) + old_masked.get(masked_body, [])))
                if overlaps:
                    record['exclusion_reasons'].append('exact_body_or_model_input_overlap_with_existing_corpus')
                    record['existing_corpus_overlap_rows'] = overlaps
                value = fingerprint(masked)
                candidates = set()
                if value:
                    for chunk in range(8):
                        candidates.update(buckets[chunk][(value >> (chunk * 8)) & 255])
                near_old = [idx for idx in sorted(candidates) if (value ^ reference[idx]).bit_count() <= 6]
                if near_old:
                    record['review_flags'].append('approximate_existing_corpus_similarity')
                    record['near_existing_corpus_rows'] = near_old
                near_new = [idx for idx, previous in enumerate(candidate_hashes)
                            if value and previous and (value ^ previous).bit_count() <= 6]
                if near_new:
                    record['review_flags'].append('approximate_external_message_similarity')
                    record['near_external_inventory_rows'] = near_new
                if not length_low <= len(body) <= length_high:
                    record['review_flags'].append('size_outlier_vs_legacy_training_distribution')
                if '\ufffd' in body:
                    record['review_flags'].append('replacement_character_encoding_warning')
                if record['mime_defects']:
                    record['review_flags'].append('mime_parser_defect')
                try:
                    date = parsedate_to_datetime(str(message.get('Date', '')))
                    record['claimed_date'] = date.isoformat()
                    record['claimed_date_is_verified'] = False
                    if date.year != year:
                        record['review_flags'].append('claimed_date_differs_from_mailbox_year')
                except (ValueError, TypeError, OverflowError):
                    record['claimed_date'] = None
                    record['review_flags'].append('missing_or_unparseable_date')
            except Exception as error:
                record['exclusion_reasons'].append('parse_failure')
                record['parse_error'] = str(error)
            candidate_hashes.append(value)
            record['text_candidate_eligible'] = not record['exclusion_reasons']
            # Nobody has independently adjudicated these labels yet.
            record['approved_for_training'] = record['approved_for_final_benchmark'] = False
            records.append(record)
            counts['messages'] += 1
            counts['text_candidates'] += int(record['text_candidate_eligible'])
            counts.update('excluded/' + reason for reason in record['exclusion_reasons'])
            counts.update('flag/' + flag for flag in record['review_flags'])
        box.close()
        summary[str(year)] = dict(counts)
        print('Email candidate inventory ' + str(year) + ': ' + json.dumps(dict(counts)), flush=True)
    target = OUT / 'candidate_inventory.jsonl'
    with target.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    candidate_target = OUT / 'deduplicated_text_candidates.jsonl'
    selected = [record for record in records if record['text_candidate_eligible']]
    assert len({record['model_input_sha256'] for record in selected}) == len(selected)
    with candidate_target.open('w', encoding='utf-8') as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    assert sha(corpus_path.read_bytes()) == original_hash
    assert all(sha((ROOT / item['path']).read_bytes()) == item['sha256'] for item in acquisition['files'])
    report = {'complete': True, 'counts': summary, 'inventory_sha256': sha(target.read_bytes()),
              'deduplicated_text_candidates_sha256': sha(candidate_target.read_bytes()),
              'existing_corpus_sha256': original_hash, 'existing_corpus_changed': False,
              'source_snapshots_changed': False, 'model_predictions_computed': False,
              'models_trained': False, 'independent_benchmark_ready': False,
              'outlier_reference_body_length_quantiles': {'0.001': float(length_low), '0.999': float(length_high)},
              'policy': 'Exact duplicates/overlaps and unusable statistical text excluded from candidate text evaluation; approximate matches and outliers flagged, never relabeled or deleted',
              'simhash_scope': '64-bit whitespace token fingerprint over first 80 tokens, hamming <=6; approximate screening, not campaign ground truth',
              'remaining': ['Manual review of near-overlaps, MIME/encoding flags and source labels',
                            'Independently reviewed contemporary benign comparison set',
                            'Freeze campaign-separated accepted development/final evaluation manifests']}
    (OUT / 'curation_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
