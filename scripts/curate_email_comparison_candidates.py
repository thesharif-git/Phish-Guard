"""Inspect privately sourced original comparison seeds without scoring/admission."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from utils.text_sanitizer import clean_and_mask_email_text
from utils.feature_extractor import extract_features_single

SOURCE = ROOT / 'data/external_email_validation/phishfuzzer_original_seeds'
OUT = ROOT / 'benchmarks/email_comparison_source_review'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def inspect_row(row):
    reasons, flags = [], []
    if row.get('Source') != 'Manual':
        reasons.append('legacy_public_or_nonmanual_source_not_independent_recent_candidate')
    if row.get('Created by') != 'Human':
        reasons.append('human_authorship_field_missing_or_noncanonical_requires_review')
    year = row.get('Year')
    if not isinstance(year, int) or not 2024 <= year <= 2026:
        reasons.append('outside_claimed_contemporary_year_range')
    if not isinstance(row.get('Body'), str) or not row['Body'].strip():
        reasons.append('missing_or_invalid_body')
    if row.get('Subject') is None:
        flags.append('missing_subject_optional_header')
    elif not isinstance(row.get('Subject'), str):
        reasons.append('invalid_subject_field')
    if row.get('Type') not in ('Phishing', 'Spam', 'Valid'):
        reasons.append('unknown_source_label')
    if isinstance(row.get('Body'), str) and '\ufffd' in row['Body']:
        flags.append('body_replacement_character_encoding_warning')
    return reasons, flags


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete']
    protected_paths = [ROOT / r['path'] for r in manifest['files']] + [
        ROOT / 'data/clean_email_corpus.csv', ROOT / 'models/phish_model.pkl', ROOT / 'models/vectorizer.pkl',
        ROOT / 'models/conformal_params.json', ROOT / 'models/conformal_evaluation_results.json',
        ROOT / 'utils/text_sanitizer.py', ROOT / 'utils/feature_extractor.py', Path(__file__)]
    protected = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in protected_paths}
    assert all(sha((ROOT / r['path']).read_bytes()) == r['sha256'] for r in manifest['files'])
    rows = json.loads((SOURCE / 'original_seeds.json').read_text(encoding='utf-8'))
    old = pd.read_csv(ROOT / 'data/clean_email_corpus.csv', keep_default_na=False)
    previous = defaultdict(list)
    for idx, masked in enumerate(old.masked_text):
        previous[masked].append(idx)
    seen, inventory, candidates = {}, [], []
    counts = Counter()
    for index, row in enumerate(rows):
        reasons, flags = inspect_row(row)
        body = row.get('Body') if isinstance(row.get('Body'), str) else ''
        subject = row.get('Subject') if isinstance(row.get('Subject'), str) else ''
        masked = clean_and_mask_email_text(body + ' ' + subject)
        features = extract_features_single(body + ' ' + subject)[0].tolist()
        contract = sha(json.dumps([masked, features], ensure_ascii=False).encode())
        overlap = sorted(set(previous.get(masked, []) + previous.get(clean_and_mask_email_text(body), [])))
        if overlap:
            reasons.append('exact_masked_body_or_input_overlap_with_seen_corpus')
        if not reasons and contract in seen:
            reasons.append('exact_candidate_model_input_duplicate')
        elif not reasons:
            seen[contract] = index
        record = {'review_id': 'comparison-' + sha(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())[:16],
                  'source_row_index': index, 'source_original_id': row.get('Original_ID'),
                  'source_reported_label': row.get('Type'), 'source_reported_origin': row.get('Source'),
                  'source_reported_year': row.get('Year'), 'claimed_year_is_verified': False,
                  'source_reported_author': row.get('Created by'), 'model_input_sha256': contract,
                  'exclusion_reasons': reasons, 'review_flags': flags,
                  'existing_corpus_overlap_rows': overlap, 'candidate_eligible': not reasons,
                  'independent_security_label': None, 'approved_for_training': False,
                  'approved_for_final_benchmark': False, 'model_predictions_computed': False}
        inventory.append(record)
        counts.update('excluded/' + r for r in reasons)
        if row.get('Source') == 'Manual':
            counts['manual_source_rows'] += 1
            counts['manual_source_label/' + str(row.get('Type'))] += 1
        if not reasons:
            candidates.append(record)
            counts['candidates'] += 1
            counts['candidate_source_label/' + str(row.get('Type'))] += 1
            counts['candidate_claimed_year/' + str(row.get('Year'))] += 1
    for p in protected_paths:
        assert sha(p.read_bytes()) == protected[str(p.relative_to(ROOT))]
    for name, records in [('candidate_inventory.jsonl', inventory), ('comparison_candidates.jsonl', candidates)]:
        (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    results = {'complete': True, 'source_rows': len(rows), 'counts': dict(counts),
               'protected_hashes': protected, 'model_predictions_computed': False,
               'training_or_calibration_performed': False, 'independent_benchmark_ready': False,
               'limitations': ['Source labels are not independent adjudications',
                               'No original MIME, trustworthy reception times or verified authentication',
                               'Approximate overlap and campaign grouping still pending',
                               'Explicit dataset reuse license not established',
                               'Source actual manual class counts differ from README initial 100/class description']}
    (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    report = '# Comparison source inspection\n\nAuthor source: https://github.com/DataPhish/PhishFuzzer\n'
    report += 'Publication: https://arxiv.org/abs/2511.21448\n\nThe entire original snapshot is preserved. No expanded synthetic variants were downloaded.\n\n'
    report += '\n'.join('- ' + k + ': ' + str(v) for k, v in counts.items())
    report += ('\n\nOnly canonical Human + Manual + claimed 2024–2026 original seeds without exact old-corpus overlap or exact input repeats '
               'enter the candidate pool. This filter is based on provenance, never model correctness. It does not establish human authenticity, '
               'label correctness or independent campaigns. Noncanonical authorship is flagged rather than silently corrected.\n\n'
               'Actual source class counts are preserved; Spam is not relabelled as phishing. Candidate Valid messages provide a possible '
               'content-only benign comparison, subject to label, near-overlap and reuse review. Structured fields must not be reconstructed '
               'as purported raw original mail or used to claim sender-authentication validation. No candidate is approved for training or a final benchmark.\n')
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps({k: v for k, v in results.items() if k != 'protected_hashes'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
