"""Freeze a publisher-labeled English excerpt benchmark without model scoring.

All acquired Brown pages remain visible, including screenshots and mixed-page
coverage failures. Inputs contain message excerpts, never page titles/verdicts.
This is an evaluation source; this script does not authorize training reuse.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup
from training.email_evidence import full_message_family, full_message_tokens, group_email_templates
from training.email_source_annotations import brown_publisher_verdict
from utils.email_candidate_text import canonical_email_candidate_text
from utils.email_input_contract import build_email_input, contract_sha256
from utils.email_parser import visible_html_text
from utils.jsonl import read_jsonl

ACQUISITION_SHA256 = '4a55ea7905ea7154cb2b2be34adbd3bf5b4f112f3cd8b69d524b7e1effb1db7e'
REVIEW_CORPUS_SHA256 = '620540204a62d85c2eddfad2d679a28fcc580b663ac4ead9a99a67908a20a4cf'
LABELS = {'Legitimate Email': 'publisher_verified_nonphishing',
          'Phishing Email (on campus)': 'publisher_reported_phishing',
          'Phishing Email (in the wild)': 'publisher_reported_phishing'}
HEADER = re.compile(r'^\s*(From|Subject|Date|Sent|To|Reply-To):\s*(.*)$', re.I)
CONTROL = re.compile('[\u200b-\u200f\u2060\ufeff]')
COMMENTARY_PATTERNS = (
    r'\bthe following variations were reported\b',
    r'\b(?:note|notice)\s*:\s*(?:this is one example|a spoofed|the mismatch)\b',
    r'\bnote the mismatch of\b',
    r'\bmeeting notifications are prompted\b',
    r'\ban excerpt from a statement\b',
    r'\b(?:updated message|image examples)\s*:',
    r'\b(?:REAL|FAKE)\s*=+',
    r'\b(?:example of a reported email|email copied below)\b',
)
PROTOCOL = {
    'schema_version': 1,
    'purpose': 'Frozen English publisher-verdict message-excerpt evaluation; no training or model selection',
    'sampling': 'Every one of the 90 acquired pages, with original publisher class counts retained',
    'label_authority': 'Exact Brown page tag matched to the acquired index; no model-derived labels or project-review claims',
    'header_policy': 'Original header paragraphs located in both .flex-intro and .flex-body; page titles never substitute for Subject; inline publisher Notes preserved outside input; composite sender alternatives not asserted as a single original sender',
    'body_policy': 'Only .flex-body message HTML; wholly identified header paragraphs removed, raw section and publisher commentary preserved separately',
    'text_metric_eligibility_rules': {
        'minimum_message_body_words': 40,
        'no_images_in_message_section': True,
        'at_most_one_from_header_block': True,
        'no_ambiguous_header_body_paragraph': True,
        'no_detected_publisher_commentary_or_mixed_real_fake_examples': True,
        'message_introduction_boundary_identifiable': True,
        'no_exact_or_0_9_full_trigram_overlap_with_existing_legacy_corpus': True,
    },
    'coverage_reporting': 'Unsupported rows remain in all_records; their missing evidence is not a correct prediction or safe verdict',
    'missing_evidence': 'No raw MIME, receiver authentication, attachments, removed destinations or screenshot text is fabricated',
    'within_source_dependence': 'Template groups preserved for grouped evaluation/intervals; no duplicated rows silently discarded',
    'threshold_policy': 'Freeze from training/validation only; no threshold fitting on this source',
    'freshness': 'Previously acquired and semantically inspected source; no classifier scores used to choose extraction eligibility',
    'scope_limitations': ['Publisher-sanitized English excerpts, not complete inbox messages',
        'One institution and small class counts cannot certify a population FPR budget',
        'Publication dates are not verified message reception times',
        'This is not a 2026 recent-inbox benchmark', 'No model is approved by corpus preparation'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def header_paragraph(node):
    """Recognize standalone published header fields, not arbitrary body text."""
    clone = BeautifulSoup(str(node), 'html.parser')
    for br in clone.find_all('br'):
        br.replace_with('\n')
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip()
             for line in clone.get_text('', strip=False).split('\n')]
    lines = [line for line in lines if line and not re.fullmatch(r'[=\-_]{3,}', line)]
    if not lines or not HEADER.match(CONTROL.sub('', lines[0])):
        return None
    fields, unknown, notes = {}, [], []
    for line in lines:
        match = HEADER.match(CONTROL.sub('', line))
        if match:
            key, value = match.group(1).lower(), match.group(2).strip()
            note = re.search(r'\s+Note\s*:', value, re.I)
            if note:
                notes.append(value[note.start():].strip())
                value = value[:note.start()].strip()
            if key in fields:
                unknown.append('repeated_header_field:' + key)
            fields[key] = value
        elif fields and (' OR' in line or line.startswith('OR ') or re.fullmatch(r'[^\s<>]+@[^\s<>]+(?:\s+OR(?:\s+others)?)?', line, re.I)):
            # Preserve multiple published sender alternatives as ambiguous
            # assertions. They are not authenticated original mail headers.
            fields[next(reversed(fields))] += ' ' + line
        else:
            unknown.append(line)
    return {'fields': fields, 'raw_excerpt': '\n'.join(lines), 'ambiguous': bool(unknown),
            'unclassified_lines': unknown, 'publisher_inline_notes': notes}


def extract_page(markup, acquisition):
    soup = BeautifulSoup(markup, 'html.parser')
    tags = [node.get_text(' ', strip=True) for node in soup.select('.flex_tag_item')]
    recognized = [tag for tag in tags if tag in LABELS]
    if recognized != [acquisition['label']]:
        raise ValueError('Actual publisher tag does not match acquired verdict: ' + acquisition['url'])
    verdict = brown_publisher_verdict(LABELS[recognized[0]], acquisition['url'])
    intro, body = soup.select_one('.flex-intro'), soup.select_one('.flex-body')
    body_original = str(body) if body else ''
    body_copy = BeautifulSoup(body_original, 'html.parser')
    header_blocks = []
    for section_name, section in (('intro', intro), ('body', body_copy)):
        if section is None:
            continue
        # Header excerpts are paragraph elements in this publisher's layout.
        for paragraph in list(section.find_all('p')):
            parsed = header_paragraph(paragraph)
            if parsed:
                header_blocks.append(dict(section=section_name, **parsed))
                if section_name == 'body' and not parsed['ambiguous']:
                    paragraph.decompose()
    from_blocks = [block for block in header_blocks if 'from' in block['fields']]
    fields = header_blocks[0]['fields'] if len(from_blocks) <= 1 and header_blocks else {}
    if len(from_blocks) == 1:
        fields = from_blocks[0]['fields']
    body_html = str(body_copy)
    body_text = visible_html_text(body_html)
    intro_text = visible_html_text(str(intro)) if intro else ''
    images = [{'src': image.get('src', ''), 'alt': image.get('alt', '')}
              for image in body_copy.find_all('img')]
    reasons = []
    if len(re.findall(r'\b\w+\b', body_text)) < PROTOCOL['text_metric_eligibility_rules']['minimum_message_body_words']:
        reasons.append('insufficient_visible_message_text')
    if images:
        reasons.append('unscanned_message_section_images')
    if len(from_blocks) > 1:
        reasons.append('multiple_message_header_blocks')
    if any(block['ambiguous'] for block in header_blocks):
        reasons.append('ambiguous_published_header_paragraph')
    if not header_blocks and intro_text.strip():
        # Some introductions are publisher discussion, some are the start of
        # an actual message. Neither can be assumed without source review.
        reasons.append('message_introduction_boundary_unverified')
    matched_commentary = [pattern for pattern in COMMENTARY_PATTERNS if re.search(pattern, body_text, re.I)]
    if matched_commentary:
        reasons.append('publisher_commentary_or_mixed_examples_in_message_section')
    sender = fields.get('from', '')
    sender_alternatives = bool(re.search(r'\bOR\b', sender))
    email_input = build_email_input(body_text, subject=fields.get('subject', ''),
        sender='' if sender_alternatives else sender, html_body=body_html, headers='')
    identity = 'brown_phish_bowl-' + hashlib.sha256(acquisition['url'].encode()).hexdigest()[:20]
    return {
        'id': identity, 'source': 'brown_phish_bowl', 'label': int(verdict['task_label'] == 'phishing'),
        **verdict, 'actual_publisher_tag': recognized[0],
        'original_reference': dict(url=acquisition['url'], snapshot_path=acquisition['path'],
                                  snapshot_sha256=acquisition['sha256']),
        'publisher_title': acquisition.get('title'), 'publisher_published_at': acquisition.get('published'),
        'publication_is_not_message_reception_time': True,
        'publisher_intro_html': str(intro) if intro else '', 'publisher_body_html': body_original,
        'published_header_blocks': header_blocks, 'published_header_fields': fields,
        'publisher_sender_alternatives': sender if sender_alternatives else None,
        'original_mime_available': False, 'receiver_authentication_verified': False,
        'email_input': email_input, 'input_contract_sha256': contract_sha256(email_input),
        'body': body_text, 'subject': fields.get('subject', ''), 'sender': '' if sender_alternatives else sender,
        'headers': '', 'html_body': body_html, 'images_unscanned': images,
        'detected_commentary_patterns': matched_commentary,
        'coverage_reasons': sorted(set(reasons)), 'exclusion_reasons': sorted(set(reasons)),
        'eligible_for_text_metrics': not reasons, 'eligible_for_scoring': not reasons,
        'extraction_review_kind': 'Deterministic source-layout extraction; project-independent human review not claimed',
        'training_reuse_permission_established': False,
        'model_scores_consulted_for_selection': False,
    }


def annotate_legacy_overlap(rows, legacy_path):
    """Bounded-memory exhaustive candidate check against every legacy raw text."""
    signatures, families, index = {}, {}, defaultdict(set)
    for row in rows:
        text = row['body'] + ' ' + row['subject']
        tokens = full_message_tokens(text, canonicalizer=canonical_email_candidate_text)
        signature = {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}
        signatures[row['id']] = signature
        families[row['id']] = full_message_family(text, canonicalizer=canonical_email_candidate_text)
        for shingle in signature:
            index[shingle].add(row['id'])
        row['legacy_overlap_matches'] = []
    by_id = {row['id']: row for row in rows}
    with Path(legacy_path).open(encoding='utf-8-sig', newline='') as handle:
        for number, original in enumerate(csv.DictReader(handle)):
            tokens = full_message_tokens(original['raw_text'], canonicalizer=canonical_email_candidate_text)
            signature = {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}
            candidates = set()
            for shingle in signature:
                candidates.update(index.get(shingle, ()))
            family = full_message_family(original['raw_text'], canonicalizer=canonical_email_candidate_text)
            for identity in candidates:
                target = signatures[identity]
                if min(len(signature), len(target)) < .9 * max(len(signature), len(target)):
                    continue
                union = len(signature | target)
                similarity = len(signature & target) / union if union else 0.
                if family == families[identity] or similarity >= .9:
                    by_id[identity]['legacy_overlap_matches'].append(dict(legacy_csv_row=number,
                        split=original['split'], source=original['source'], full_trigram_jaccard=similarity))
    for row in rows:
        if row['legacy_overlap_matches']:
            row['coverage_reasons'].append('legacy_corpus_content_overlap')
            row['coverage_reasons'] = sorted(set(row['coverage_reasons']))
            row['exclusion_reasons'] = row['coverage_reasons']
            row['eligible_for_text_metrics'] = False
            row['eligible_for_scoring'] = False


def prepare(snapshot_dir, prior_review, legacy_path, output):
    snapshot_dir, prior_review, legacy_path, output = map(Path, (snapshot_dir, prior_review, legacy_path, output))
    if output.exists():
        raise ValueError('Frozen Brown benchmark output already exists; do not overwrite an inspected benchmark')
    acquisition_path = snapshot_dir / 'acquisition.json'
    if sha(acquisition_path) != ACQUISITION_SHA256 or sha(prior_review) != REVIEW_CORPUS_SHA256:
        raise ValueError('Frozen acquisition or prior review corpus hash mismatch')
    acquired = [item for item in json.loads(acquisition_path.read_text(encoding='utf-8'))
                if item['path'].startswith('brown_pages/')]
    prior = [row for row in read_jsonl(prior_review) if row['source'] == 'brown_phish_bowl']
    if len(acquired) != 90 or {r['original_reference']['url'] for r in prior} != {r['url'] for r in acquired}:
        raise ValueError('Brown acquisition does not match the preserved 90-page review pool')
    for item in acquired:
        path = (snapshot_dir / item['path']).resolve()
        if not path.is_relative_to(snapshot_dir.resolve()) or sha(path) != item['sha256']:
            raise ValueError('Brown snapshot path/hash mismatch')
    # Freeze rules and all source hashes before producing the benchmark. This
    # script has no model loader, classifier predictions or threshold tuning.
    output.mkdir(parents=True)
    frozen = dict(PROTOCOL, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        acquisition_sha256=sha(acquisition_path), prior_review_sha256=sha(prior_review),
        legacy_corpus_sha256=sha(legacy_path), preparation_code_sha256=sha(__file__),
        source_snapshot_hashes={r['path']: r['sha256'] for r in acquired})
    write_json(output / 'protocol.json', frozen)
    rows = [extract_page((snapshot_dir / item['path']).read_bytes(), item)
            for item in sorted(acquired, key=lambda item: item['url'])]
    annotate_legacy_overlap(rows, legacy_path)
    grouped, pairs = group_email_templates(rows, canonicalizer=canonical_email_candidate_text)
    for row, group in zip(rows, grouped):
        row['campaign_cluster_id'] = group
    write_jsonl(output / 'all_records.jsonl', rows)
    eligible = [row for row in rows if row['eligible_for_text_metrics']]
    write_jsonl(output / 'evaluation_records.jsonl', eligible)
    write_jsonl(output / 'template_pairs.jsonl', pairs)
    manifest = dict(schema_version=1, complete=True, purpose=PROTOCOL['purpose'],
        all_records=len(rows), publisher_class_counts=dict(Counter(r['task_label'] for r in rows)),
        eligible_text_records=len(eligible), eligible_class_counts=dict(Counter(r['task_label'] for r in eligible)),
        coverage_reason_counts=dict(Counter(reason for r in rows for reason in r['coverage_reasons'])),
        original_sender_available=sum(bool(r['sender']) for r in rows),
        original_subject_available=sum(bool(r['subject']) for r in rows),
        eligible_template_groups=len({r['campaign_cluster_id'] for r in eligible}),
        full_template_pairs=len(pairs), all_records_sha256=sha(output / 'all_records.jsonl'),
        evaluation_records_sha256=sha(output / 'evaluation_records.jsonl'), protocol_sha256=sha(output / 'protocol.json'),
        classifier_scores_computed=False, source_not_used_for_training=True,
        project_independent_label_review_completed=False, approved_for_deployment=False,
        task_valid_authority='Brown OIT actual publisher verdict tags',
        missing_authentication_not_fabricated=True, inputs_do_not_include_publisher_verdict_or_title=True,
        limitations=PROTOCOL['scope_limitations'])
    write_json(output / 'manifest.json', manifest)
    selections = [r for r in rows if r['original_reference']['url'].endswith((
        '/nih-annual-validation-your-personal-information', '/your-brown-dropbox-account-scheduled-deletion',
        '/microsoft-account-security-alert-legitimate-email',
        '/attention-required-critical-irs-notice-regarding-your-w-2', '/can-you-provide-assistance',
        '/chase-bank-guard-shared-item', '/lab-results-available-your-provider-register-now-legitimate-email'))]
    report = ['# Brown Email excerpt extraction review', '',
        'No classifier was loaded or scored. Eligibility rules and snapshot hashes were frozen in protocol.json before extraction.', '',
        f"All {len(rows)} pages remain in all_records.jsonl: {manifest['publisher_class_counts']}. The text-only measurement pool has {len(eligible)} rows: {manifest['eligible_class_counts']}.", '',
        'Labels come from actual Brown OIT page tags, verified against the archived acquisition index. This is credible publisher adjudication, not a fabricated independent project review.', '',
        'Original headers were located in both introduction and message sections. Header paragraphs are kept as source evidence; page headings and verdicts never enter model inputs. Complete raw MIME and verified authentication remain unavailable.', '',
        'Screenshots, insufficient visible text, ambiguous header paragraphs, mixed-message pages, detected publisher commentary and legacy overlap remain explicit coverage states. They are not counted as correct predictions or safe mail. The conservative no-images subset is narrower than full scanner coverage.', '',
        f"Coverage reasons: {json.dumps(manifest['coverage_reason_counts'], sort_keys=True)}", '',
        '## Actual source examples', '']
    for row in selections:
        report.extend([f"- [{row['subject'] or row['id']}]({row['original_reference']['url']}): publisher {row['task_label']}; text eligible={row['eligible_for_text_metrics']}; sender assertion `{row['sender']}`; coverage reasons={row['coverage_reasons']}.",
            '  Body excerpt: ' + row['body'][:240].replace('\n', ' '), ''])
    report.extend(['## Scope and evaluation requirements', '',
        'This is a distinct English institution source with security/action notices as hard negatives. It is suitable for a scoped unseen-source excerpt evaluation after checking any candidate training sources; it cannot certify modern inbox performance, complete-message attachment detection, authenticated identity, or a population 1% false-positive budget.', '',
        'Publisher publication times are not message reception timestamps. The selected pages are mostly older examples. Template dependence and the small phishing sample require clear intervals and denominator reporting. Use fixed training/validation thresholds and report failures alongside the preserved PhishFuzzer results; do not tune on these pages.', '',
        'Training reuse permission is not established by this evaluation preparation. No dataset, model or calibration was promoted.'])
    (output / 'EXTRACTION_REVIEW.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, default=ROOT / 'data/email_root_audit_sources_20261003')
    parser.add_argument('--prior-review', type=Path, default=ROOT / 'data/email_evidence_review_20261003/review_corpus.jsonl')
    parser.add_argument('--legacy-corpus', type=Path, default=ROOT / 'data/clean_email_corpus.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/brown_email_benchmark_20261005')
    args = parser.parse_args()
    print(json.dumps(prepare(args.snapshots, args.prior_review, args.legacy_corpus, args.out), indent=2))
