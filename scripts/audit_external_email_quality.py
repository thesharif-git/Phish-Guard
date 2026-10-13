"""Offline source-byte review. No predictions, fitted transforms, or gold labels.

The alternate text here is a diagnostic, not a change to production preprocessing.
Missing charsets are recoverable only when the original payload is strict UTF-8.
Invalid declared encodings are retained for review, never guessed as Latin-1.
"""
import hashlib
import html
import json
import mailbox
import re
from collections import Counter
from email import policy
from email.parser import BytesParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bs4 import BeautifulSoup
from utils.email_parser import parse_eml_content, extract_email_anchor_links
from utils.feature_extractor import extract_features_single
from utils.text_sanitizer import clean_and_mask_email_text

SOURCE = ROOT / 'data/external_email_validation/nazario_2024_2025'
OUT = ROOT / 'benchmarks/email_data_quality_review'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def text_parts(message):
    """Exclude attached and nested messages; their text is separate evidence."""
    if message.get_content_disposition() == 'attachment' or message.get_content_type() == 'message/rfc822':
        return
    if message.is_multipart():
        for child in message.iter_parts():
            yield from text_parts(child)
    elif message.get_content_type() in ('text/plain', 'text/html'):
        yield message


def decode_part(part):
    payload = part.get_payload(decode=True)
    if payload is None:
        payload = b''
    declared = part.get_content_charset()
    codec = declared or 'ascii'
    notes = []
    try:
        decoded = payload.decode(codec, errors='strict')
    except (LookupError, UnicodeDecodeError):
        if not declared:
            try:
                decoded = payload.decode('utf-8', errors='strict')
                codec = 'utf-8'
                notes.append('missing_charset_strict_utf8_recovery')
            except UnicodeDecodeError:
                decoded = part.get_content()
                notes.append('missing_charset_non_utf8_unresolved')
        else:
            decoded = part.get_content()
            notes.append('declared_charset_invalid_payload_unresolved')
    if not isinstance(decoded, str):
        decoded = str(decoded)
    if '\ufffd' in decoded:
        notes.append('replacement_characters_remain')
    return decoded, {'content_type': part.get_content_type(), 'declared_charset': declared,
                     'diagnostic_codec': codec, 'payload_sha256': digest(payload),
                     'payload_bytes': len(payload), 'notes': notes,
                     'replacement_characters': decoded.count('\ufffd')}


def html_text(value):
    soup = BeautifulSoup(value, 'html.parser')
    for tag in soup(['script', 'style', 'head', 'template']):
        tag.decompose()
    return soup.get_text(' ', strip=True)


def inspect_message(raw):
    message = BytesParser(policy=policy.default).parsebytes(raw)
    production = parse_eml_content(raw)
    plain, markup, parts = [], [], []
    for part in text_parts(message):
        text, report = decode_part(part)
        parts.append(report)
        (plain if part.get_content_type() == 'text/plain' else markup).append(text)
    diagnostic_body = next((p.strip() for p in plain if p.strip()), '')
    if not diagnostic_body:
        diagnostic_body = next((html_text(p) for p in markup if html_text(p)), '')
    flags = []
    if message.get_content_type() == 'text/html' and production['body'] != diagnostic_body:
        flags.append('single_part_html_markup_enters_production_body')
    if any('missing_charset_strict_utf8_recovery' in p['notes'] for p in parts):
        flags.append('missing_charset_strict_utf8_recovery')
    if any('unresolved' in n for p in parts for n in p['notes']):
        flags.append('unresolved_encoding')
    if any(p['replacement_characters'] for p in parts):
        flags.append('replacement_characters_remain')
    if len(plain) > 1:
        flags.append('multiple_inline_plain_parts_require_body_selection_review')
    mime_defects = sorted({type(d).__name__ for p in message.walk() for d in p.defects})
    if mime_defects:
        flags.append('mime_defect_requires_review')
    original_input = production['body'] + ' ' + str(production['subject'])
    diagnostic_input = diagnostic_body + ' ' + str(production['subject'])
    before = extract_features_single(original_input)[0].tolist()
    after = extract_features_single(diagnostic_input)[0].tolist()
    diagnostic_masked = clean_and_mask_email_text(diagnostic_input)
    return {'flags': flags, 'parts': parts, 'mime_defects': mime_defects,
            'production_body_characters': len(production['body']),
            'diagnostic_body_characters': len(diagnostic_body),
            'production_body_sha256': digest(production['body'].encode()),
            'diagnostic_body_sha256': digest(diagnostic_body.encode()),
            'structural_feature_indices_changed': [i for i, (a, b) in enumerate(zip(before, after)) if a != b],
            'masked_input_changed': clean_and_mask_email_text(original_input) != diagnostic_masked,
            'diagnostic_masked_input_sha256': digest(diagnostic_masked.encode()),
            'independent_security_label': None,
            'approved_for_training': False, 'approved_for_final_benchmark': False}, {
                'sender': str(production['sender']), 'subject': str(production['subject']),
                'headers': production['headers'], 'production_body': production['body'],
                'diagnostic_body': diagnostic_body,
                'anchor_destinations': extract_email_anchor_links('\n'.join(markup))}, diagnostic_masked


def review_document(cards):
    """Escaped static text only: no scripts, live URLs, or email HTML rendering."""
    def escaped(value):
        return html.escape(str(value), quote=True)
    sections = []
    for card in cards:
        sections.append('<article><h2>' + escaped(card['review_id']) + '</h2><p>' +
                        escaped(card['subject']) + '</p><p>From: ' + escaped(card['sender']) +
                        '</p><p>Technical flags: ' + escaped(', '.join(card['technical_flags'])) +
                        '</p><details><summary>Headers (unverified claims)</summary><pre>' +
                        escaped(card['headers']) + '</pre></details><h3>Diagnostic decoded body</h3><pre>' +
                        escaped(card['diagnostic_body']) + '</pre><details><summary>Current production body</summary><pre>' +
                        escaped(card['production_body']) + '</pre></details><details><summary>Link text and destinations (not opened)</summary><pre>' +
                        escaped(json.dumps(card['anchor_destinations'], ensure_ascii=False, indent=2)) +
                        '</pre></details></article>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; '
            'img-src \'none\'; connect-src \'none\'; frame-src \'none\'; form-action \'none\'; base-uri \'none\'">'
            '<title>Email source quality review</title><style>body{font:16px system-ui;max-width:1000px;margin:32px auto;'
            'padding:0 20px;color:#17243a}article{border-top:1px solid #b9c4d5;padding:24px 0}pre{white-space:pre-wrap;'
            'overflow-wrap:anywhere;background:#f3f5f8;padding:16px}h2{font-size:20px}</style>'
            '<h1>Email source quality review</h1><p>This technical subset is selected by data-quality flags, not model '
            'errors. It is not a representative accuracy sample. Collector labels and model predictions are withheld; '
            'the selection still comes from a collector phishing corpus and is not fully label-blind.</p>'
            '<p>No destinations or remote images load. Alternate decoding is diagnostic only. Sender and authentication '
            'headers are unverified. Record labels as PHISHING, BEC, SPAM, BENIGN or INSUFFICIENT_EVIDENCE with cited '
            'evidence; a persuasive topic or familiar sender alone does not establish a label.</p>' + ''.join(sections) + '</html>')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acquisition = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    inventory_path = SOURCE / 'candidate_inventory.jsonl'
    from utils.jsonl import read_jsonl
    inventory = read_jsonl(inventory_path)
    quality = json.loads((SOURCE / 'curation_results.json').read_text(encoding='utf-8'))
    size_bounds = quality['outlier_reference_body_length_quantiles']
    protected_paths = [ROOT / f['path'] for f in acquisition['files']] + [
        ROOT / 'data/clean_email_corpus.csv', ROOT / 'models/phish_model.pkl', ROOT / 'models/vectorizer.pkl',
        ROOT / 'models/conformal_params.json', ROOT / 'models/conformal_evaluation_results.json', inventory_path,
        SOURCE / 'curation_results.json', ROOT / 'utils/email_parser.py', ROOT / 'utils/text_sanitizer.py',
        ROOT / 'utils/feature_extractor.py', Path(__file__)]
    protected = {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in protected_paths}
    assert all(protected[f['path']] == f['sha256'] for f in acquisition['files'])
    boxes = {year: mailbox.mbox(SOURCE / f'phishing-{year}.mbox', create=False) for year in (2024, 2025)}
    reports, cards, content_seen = [], [], {}
    counts = Counter()
    try:
        for record in inventory:
            year, key = record['source_mailbox_year'], record['source_message_key']
            raw = boxes[year].get_bytes(key, from_=False)
            assert digest(raw) == record['source_message_sha256']
            report, card, masked = inspect_message(raw)
            review_id = 'email-' + digest(raw)[:16]
            report.update({'review_id': review_id, 'source_path': record['source_path'],
                           'source_message_key': key, 'source_message_sha256': digest(raw),
                           'intended_role': record['intended_role'], 'inventory_flags': record['review_flags'],
                           'statistical_input_exclusions': record['exclusion_reasons']})
            if 'size_outlier_vs_legacy_training_distribution' in record['review_flags']:
                counts['initial_size_outliers'] += 1
                inside = size_bounds['0.001'] <= len(card['diagnostic_body']) <= size_bounds['0.999']
                report['diagnostic_visible_length_inside_initial_size_bounds'] = inside
                counts['initial_size_outliers_inside_bounds_after_diagnostic_text_extraction'] += int(inside)
            # Full diagnostic text, not the first 80 tokens; still not campaign truth.
            if masked and masked in content_seen:
                report['same_full_diagnostic_masked_text_as'] = content_seen[masked]
                report['flags'].append('full_diagnostic_masked_text_repeat_requires_group_review')
                counts['full_diagnostic_masked_text_repeats'] += 1
            elif masked:
                content_seen[masked] = review_id
            counts.update(report['flags'])
            counts['messages'] += 1
            counts['masked_input_changed'] += int(report['masked_input_changed'])
            counts['structural_input_changed'] += bool(report['structural_feature_indices_changed'])
            reports.append(report)
            # Compact technical sample; all 884 are checked in the JSON ledger.
            if set(report['flags']) & {'missing_charset_strict_utf8_recovery', 'unresolved_encoding',
                                      'mime_defect_requires_review'} or 'approximate_existing_corpus_similarity' in record['review_flags']:
                card.update({'review_id': review_id, 'technical_flags': report['flags'] + record['review_flags']})
                cards.append(card)
    finally:
        for box in boxes.values():
            box.close()
    for p in protected_paths:
        assert digest(p.read_bytes()) == protected[p.relative_to(ROOT).as_posix()]
    ledger = OUT / 'technical_review.jsonl'
    ledger.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reports), encoding='utf-8')
    (OUT / 'review.html').write_text(review_document(cards), encoding='utf-8')
    template = [{'review_id': r['review_id'], 'source_message_sha256': r['source_message_sha256'],
                 'reviewer_id': None, 'reviewer_type': None, 'security_label': None,
                 'evidence': [], 'quality_disposition': None, 'campaign_group': None,
                 'review_complete': False} for r in reports]
    target = OUT / 'label_review_template.jsonl'
    # A rerun cannot erase completed user reviews.
    if not target.exists():
        target.write_text(''.join(json.dumps(r) + '\n' for r in template), encoding='utf-8')
    result = {'complete': True, 'counts': dict(counts), 'technical_review_cards': len(cards),
              'production_preprocessing_changed': False, 'model_predictions_computed': False,
              'labels_adjudicated': 0, 'independent_benchmark_ready': False,
              'protected_hashes': protected, 'technical_ledger_sha256': digest(ledger.read_bytes()),
              'limitations': ['Alternate parser text is a diagnostic, not a trained/evaluated replacement',
                              'Exact full masked-text repeats are not verified campaigns',
                              'No human or independently adjudicated labels yet',
                              'Contemporary benign comparison data still required']}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report = '# Email source-byte quality audit\n\nAll 884 original messages were checked offline; no model predictions or labels were inferred.\n\n'
    report += '\n'.join('- ' + name + ': ' + str(value) for name, value in counts.items())
    report += ('\n\nSingle-part HTML is currently sent as raw markup, whereas multipart HTML-only messages are converted to text. '
               'This is a demonstrated transport-dependent representation defect, not proof of its contribution to model error. '
               'Missing charset messages can lose valid UTF-8 characters under the default ASCII decoder; strict UTF-8 recoveries '
               'are recorded separately from corrupt/misdeclared payloads. Invalid declared bytes remain unresolved, without a guessed codec.\n\n'
               'HTML extraction in the diagnostic removes script/style/head/template content; there is no network access or HTML rendering. '
               'Size outliers and malformed MIME remain original evidence, not automatic garbage or new labels. Exact full diagnostic '
               'masked-text repeats require campaign grouping even when their structural model features differ.\n\n'
               'review.html presents a small technical subset with source labels and model outputs withheld; the known single-class source '
               'prevents claiming a completely blind label study. label_review_template.jsonl is intentionally unfilled. A human review must '
               'cite evidence and may select insufficient evidence. Two independent human decisions and conflict adjudication are required '
               'before accepting definitive benchmark labels; AI suggestions cannot masquerade as independent human review.\n\n'
               'Original source, inventory, corpus, trained artifacts and calibration hashes were preserved. No production parsing change, '
               'training or calibration was performed in this audit.\n')
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'protected_hashes'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
