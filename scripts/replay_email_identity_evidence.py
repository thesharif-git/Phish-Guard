"""Replay connected identity checks on preserved Email inputs, without fitting ML.

Historical probabilities/actions remain archival observations. This component
comparison cannot establish classifier accuracy, ground truth or deployment safety.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.origintrace_pillars import CrossEntityContradictionEngine, ProvenanceMatrix
from utils.email_input_contract import contract_sha256, validate_email_input
from utils.jsonl import read_jsonl


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(*, previous_corpus, corpus, captured_results, out_dir):
    previous = {row['id']: row for row in read_jsonl(previous_corpus)}
    current = {row['id']: row for row in read_jsonl(corpus)}
    captures = read_jsonl(captured_results)
    if not captures or len({row['record_id'] for row in captures}) != len(captures):
        raise ValueError('Nonempty unique archived Email IDs required')
    out_dir = Path(out_dir).resolve()
    out_dir.relative_to((ROOT / 'benchmarks').resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_label = {}
    for capture in captures:
        record_id = capture['record_id']
        record = current[record_id]
        email = validate_email_input(record['email_input'])
        if contract_sha256(email) != record['input_contract_sha256']:
            raise ValueError('Email evidence hash mismatch: ' + record_id)
        for field in ('body', 'subject', 'sender', 'headers', 'html_body'):
            if previous[record_id].get(field, '') != email[field]:
                raise ValueError(f'Original Email {field} changed: {record_id}')
        archived = capture['result']
        old = archived['contradiction_analysis']
        old_p1 = archived['pillars']['pillar1_provenance']
        links = [{'text': link['anchor_text'], 'href': link['destination']}
                 for link in email['links'] if link['source'].endswith('_html_anchor')]
        current_result = CrossEntityContradictionEngine.analyze(
            sender_header=email['sender'], raw_headers=email['headers'],
            urls=email['link_destinations'], anchor_links=links,
            destination_domain=archived['url_intelligence'].get('domain', ''),
            destination_asn=archived['url_intelligence'].get('asn', ''),
            destination_isp=archived['url_intelligence'].get('isp', ''))
        p1 = ProvenanceMatrix.analyze(email['sender'], email['headers'], urls=email['link_destinations'])
        label_key = f"{record['source']} | {record['source_claimed_label']}"
        counts = per_label.setdefault(label_key, Counter())
        counts.update(rows=1, old_identity_review=int(old['is_contradictory']),
                      new_identity_review=int(current_result['is_contradictory']),
                      old_provenance_lookalike=int(old_p1.get('lookalike_domain', False)),
                      new_provenance_lookalike=int(p1['lookalike_domain']))
        rows.append({
            'record_id': record_id, 'source': record['source'],
            'source_claimed_label': record['source_claimed_label'],
            'source_label_is_independent_project_ground_truth': False,
            'input_contract_sha256': record['input_contract_sha256'],
            'original_core_fields_unchanged': True,
            'archived_content_model_probability': archived['content_analysis']['ml_probability'],
            'new_content_model_inference_performed': False,
            'archived_action': archived['security_action'],
            'archived_calibration_valid': archived['conformal_guarantee']['calibration_valid'],
            'old_contradiction': old, 'new_contradiction': current_result,
            'old_provenance': old_p1, 'new_provenance': p1,
        })
    counts = {
        'rows': len(rows),
        'old_identity_review': sum(row['old_contradiction']['is_contradictory'] for row in rows),
        'new_identity_review': sum(row['new_contradiction']['is_contradictory'] for row in rows),
        'identity_review_changed_rows': sum(row['old_contradiction']['is_contradictory'] != row['new_contradiction']['is_contradictory'] for row in rows),
        'old_provenance_lookalike': sum(row['old_provenance'].get('lookalike_domain', False) for row in rows),
        'new_provenance_lookalike': sum(row['new_provenance']['lookalike_domain'] for row in rows),
    }
    row_path = out_dir / 'identity_component_replay.jsonl'
    row_path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    manifest = {'complete': True, 'scope': 'identity_component_replay_not_classifier_accuracy',
        'counts': counts, 'source_label_breakdown': per_label,
        'input_sha256': {str(Path(path).relative_to(ROOT)): sha256(path)
                         for path in (previous_corpus, corpus, captured_results)},
        'code_sha256': {str(Path(path).relative_to(ROOT)): sha256(path) for path in (
            ROOT / 'services/origintrace_pillars.py', ROOT / 'utils/email_input_contract.py',
            ROOT / 'utils/email_authentication.py', ROOT / 'utils/jsonl.py', Path(__file__))},
        'replay_sha256': sha256(row_path), 'models_datasets_calibration_modified': False,
        'no_model_weights_or_network_used': True, 'external_generalization_improvement_claim': None,
        'original_core_fields_unchanged': True, 'archived_actions_not_reclassified': True}
    (out_dir / 'identity_component_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    report = ['# Connected Email identity evidence repair', '',
        'This is a component replay of preserved Email inputs. It is not a new accuracy measurement. No model inference, retraining, network or calibration was performed.', '',
        f"{counts['rows']} recorded messages have identical original body, subject, sender, headers and HTML. Their archived content probabilities and decisions remain unchanged in the replay.", '',
        '| Source-label group | Emails | Old identity review | Corrected identity review |',
        '|---|---:|---:|---:|']
    for key, group in per_label.items():
        report.append(f"| {key.replace('|', '/')} | {group['rows']} | {group['old_identity_review']} | {group['new_identity_review']} |")
    report.extend(['',
        f"Identity review changed for {counts['identity_review_changed_rows']} recorded messages. Old/new lookalike candidates: {counts['old_provenance_lookalike']}/{counts['new_provenance_lookalike']}.", '',
        'The archived 42-message full-engine run produced warnings for every message because deployment calibration was invalid. It cannot measure accuracy or a false-positive rate. Source labels shown above are provenance, not independent project adjudication.', '',
        'Corrected relationships distinguish display-name claims, unverified signature assertions and partner/tracking hosts from an explicit displayed URL/href difference. No relationship is promoted to proof of cryptographic failure, impersonation or phishing.', '',
        'The historical direct-model 56.01% external accuracy / 0.5371 AUC and best tested 62.89% / 0.6216 remain unresolved. These identity repairs cannot fix those rankings. They remove unsupported engine evidence and prevent its amplification or rendering failure.', ''])
    (out_dir / 'IDENTITY_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'benchmarks/email_connected_engine_repair_20261005')
    args = parser.parse_args()
    result = replay(previous_corpus=ROOT / 'data/email_evidence_review_20261003/review_corpus.jsonl',
        corpus=ROOT / 'data/email_evidence_v1_20261004_v2/review_corpus.jsonl',
        captured_results=ROOT / 'benchmarks/email_generalization_root_20261003/operational_evidence_check_v3/results.jsonl',
        out_dir=args.out_dir)
    print(json.dumps(result['counts']))
