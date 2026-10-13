"""Generate a report from retained measurements, never target performance values."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import pandas as pd
from scripts.audit_email_generalization_root import measure
from services.security_pipeline import security_pipeline_id

OUT = ROOT / 'benchmarks/email_generalization_root_20261003'
REPLAY = OUT / 'operational_evidence_check_v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return f'{100*value:.2f}%'


def main():
    results = read(OUT / 'results.json')
    provenance = read(OUT / 'provenance_and_overlap.json')
    attribution = read(OUT / 'failure_attribution.json')
    manifest = read(ROOT / 'data/email_evidence_review_20261003/manifest.json')
    replay = read(REPLAY / 'summary.json')
    protocol = read(REPLAY / 'protocol.json')
    old_protocol = read(OUT / 'protocol.json')
    assert results['complete'] and provenance['complete'] and replay['complete']
    current_identity = security_pipeline_id()
    assert current_identity == replay['pipeline_id'] == protocol['pipeline_id'], 'Replay scorer changed'
    assert sha(ROOT / 'data/email_evidence_review_20261003/review_corpus.jsonl') == manifest['corpus_sha256'] == protocol['corpus_sha256']
    protected = {name: sha(ROOT / name) == value for name, value in old_protocol['protected'].items()}
    assert all(protected.values()), 'Protected artifacts changed'
    # Recompute published classifier metrics from per-message predictions, not aggregate claims.
    decisions = pd.read_csv(ROOT / 'benchmarks/email_retraining_20261003/decisions.csv')
    measured = {}
    for split in ('test', 'external'):
        part = decisions[decisions.split == split].dropna(subset=['baseline_probability', 'candidate_probability'])
        labels = (part.source_label == 'Phishing').astype(int) if split == 'external' else part.source_label.astype(int)
        measured[split] = {}
        for model in ('baseline', 'candidate'):
            value = measure(labels, part[model + '_probability'])
            for key in ('n', 'tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 'roc_auc', 'brier', 'ece_10_equal_width'):
                assert np.isclose(value[key], results['metrics'][split][model][key]), (split, model, key)
            measured[split][model] = value
    from utils.jsonl import read_jsonl
    footer = read_jsonl(OUT / 'archive_footer_probe.jsonl')
    footer_flips = sum((row['before'] >= .5) != (row['after'] >= .5) for row in footer)
    assert footer_flips == attribution['archive_footer_probe']['changed_decisions_at_0_5']
    suites = ET.parse(OUT / 'tests.xml').getroot()
    suites = list(suites) if suites.tag == 'testsuites' else [suites]
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    assert tests['tests'] > 0 and not tests['failures'] and not tests['errors']
    captured = read_jsonl(REPLAY / 'results.jsonl')
    assert len(captured) == replay['rows'] and not any(row['verification_issues'] for row in captured)
    times = np.array([row['elapsed_ms'] for row in captured])
    timings = defaultdict(list)
    for row in captured[1:]:
        for stage, timing in row['stage_timings'].items():
            timings[stage].append(timing)
    warm_stage = {stage: {field: float(np.median([t[field] for t in rows])) for field in ('wall_ms', 'process_cpu_ms')} for stage, rows in timings.items()}
    diagnostics = read(ROOT / 'benchmarks/email_retraining_20261003/generalization_diagnostics.json')
    overlap = read(OUT / 'verified_original_message_overlap.json')
    lines = ['# Email generalization: measured root-cause audit', '',
        'Email reliability still fails. No candidate was promoted, no model was retrained and no calibration was fitted in this audit. Standalone URL and Domain work remains paused.', '',
        '## Measurements from saved per-message predictions', '',
        'Threshold is 0.5. External labels are the source author\'s Phishing versus Spam/Valid labels, not independently adjudicated project ground truth. One external row has no usable vocabulary; its unavailable probability remains missing. It is not silently turned into a clean prediction.', '',
        '| Evaluation | Model | N | Accuracy | Precision | Recall | F1 | AUC | FPR | FNR | Brier | ECE |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split, name in [('test', 'Internal development'), ('external', 'Separate source labels')]:
        for model, label in [('baseline', 'Deployed baseline'), ('candidate', 'Earlier fold-owned candidate')]:
            m = measured[split][model]
            lines.append(f"| {name} | {label} | {m['n']} | {percent(m['accuracy'])} | {percent(m['precision'])} | {percent(m['recall'])} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {percent(m['fpr'])} | {percent(m['fnr'])} | {m['brier']:.4f} | {m['ece_10_equal_width']:.4f} |")
    lines += ['', 'Confusion matrices use rows [nonphishing, phishing] and columns [predicted nonphishing, predicted phishing]:', '']
    for split in measured:
        for model, m in measured[split].items():
            lines.append(f"- {split}/{model}: [[{m['tn']}, {m['fp']}], [{m['fn']}, {m['tp']}]].")
    base, candidate = measured['external']['baseline'], measured['external']['candidate']
    lines += ['', f"The previous preprocessing-isolation correction changed external accuracy from {percent(base['accuracy'])} to {percent(candidate['accuracy'])}, with AUC {base['roc_auc']:.4f} to {candidate['roc_auc']:.4f}. This did not resolve generalization; the candidate remains rejected.", '',
        '## Root cause supported by implementation and probes', '',
        f"1. Source shortcuts are demonstrated, not merely suspected. Appending the neutral archive footer `Archive reference: Enron 2001.` flipped {footer_flips}/{len(footer)} decisions from predicted phishing to predicted nonphishing. Those are prediction changes, not {footer_flips} confirmed attacks. This synthetic sensitivity probe does not alter original sender, links or intent and is not a new accuracy benchmark. Feature attribution also assigns substantial contributions to source/year tokens and raw formatting.",
        f"2. Training target provenance is wrong or unresolved. {provenance['upstream_matched_unique_rows']} current rows match original SpamAssassin spam-message bodies; {provenance['upstream_spam_tagged_phishing_unique_rows']} are tagged phishing in the current corpus, including {provenance['upstream_spam_tagged_phishing_test_rows']} test rows. Spam membership is established; phishing identity is not. Spam is neither automatically benign nor automatically phishing. Relabeling every match would be another unsupported assumption.",
        f"3. Validation is too similar to training and its grouping is incomplete. Of {provenance['high_cosine_reviewed_full_text']} cosine review flags, {provenance['full_unique_trigram_jaccard_ge_0_9']} full-message near-template pairs exceed 0.90 trigram Jaccard across declared groups. {len(overlap)} original raw-message hashes map to corpus rows across partitions; multiple raw versions can share a body. These are concrete overlap findings, not proof that all cosine flags are leaks.",
        '4. The current specialist uses masked body/subject TF-IDF plus 16 raw structural features and a calibrated LightGBM ensemble. Its training CSV lacks original header/authentication/time evidence. URL target identity is masked in text. Sender evidence and off-body URL metadata are not learned by this content classifier. Ordinary legitimate security wording can resemble a phishing imitation; body text alone cannot resolve that identity distinction.',
        '5. The former engine compounded uncertainty with unvalidated heuristic scores and overly certain conclusions. Assigned reliability priors are not learned epistemic uncertainty; model probabilities on shifted data are not established deployment confidence. Calibration cannot restore near-chance ranking or missing evidence.', '']
    source_inputs = diagnostics['external_by_class']['Phishing']
    sensitivity = attribution['phishing_vs_valid_excluding_spam_sensitivity_analysis']['baseline']
    leakage_sensitivity = attribution['internal_sensitivity_excluding_14_high_overlap_test_rows']['baseline']
    lines += [f"Input audit: {source_inputs['source_http_url_metadata_not_in_model_text_rows']}/{source_inputs['rows']} source-labeled phishing messages have URL metadata absent from the model body. Supplying that metadata to the evidence scanner now preserves it without inventing body text.", '',
        f"Alternative explanations were checked. Removing the Spam class from the external comparison leaves accuracy {percent(sensitivity['accuracy'])} and AUC {sensitivity['roc_auc']:.4f}; therefore the Spam mapping does not explain away failure. Excluding the confirmed high-overlap test rows leaves internal accuracy {percent(leakage_sensitivity['accuracy'])}, AUC {leakage_sensitivity['roc_auc']:.4f}; those overlaps alone do not explain the transfer gap. Neither sensitivity analysis replaces the original failing benchmark.", '',
        '**Primary evidenced bottleneck:** the learned target and source-dependent representation do not transfer to the intended task. Model capacity alone has not been established as the cause. A transformer substitution, extra fusion features or another threshold fit cannot be claimed as a fix without a controlled test.', '',
        '## Implemented repairs and their limits', '',
        '- Future Email training requires a hash-bound reviewed task-label manifest, reviewer/evidence references, both classes in every partition, multiple origins per training class, disjoint source tests, strictly later trusted observation dates and recomputed full-message template isolation. Fold-local features remain inside the estimator. Candidate artifacts never overwrite deployed models.',
        '- Legacy spam-as-phishing curation is blocked by default; explicit historical reproduction exports to a separate research location. Production recalibration on the failing legacy split is blocked. Development calibration can only write an isolated benchmark folder.',
        '- Original MIME, HTML, headers, links and source metadata are retained separately. Publisher headlines that reveal labels are never substituted for missing original email subjects. Decoding defects, incomplete/image-only inputs and unusual messages are flagged rather than automatically treated as clean or discarded.',
        '- Email evidence now distinguishes local domain-list matches from live threat feeds, uses hostname boundaries, does not infer redirects from link scores or TLDs from URL paths, and identifies heuristic review scores and assigned priors. Account-security terms, QR presence, relay/display-name differences and unverified Received headers do not establish fraud, spoofing or hosting abuse.',
        '- Final explanation follows the actual action and deduplicates repeated evidence. Unverified reported authentication is not cryptographic validation. Warning banners remain delivered review recommendations, not proven interception or executed quarantine.', '',
        f"Python verification: {tests['tests']} tests; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped. These test engineering behavior, not statistical generalization.", '',
        f"Real-model offline integration: {replay['rows']} fixed source-ordered emails, {replay['verification_issues']} detected violations of the recorded checks. All actions: {json.dumps(replay['actions'])}; {replay['calibration_valid_rows']} rows accepted old calibration. This review-only outcome is deliberate because the scorer changed, and is not accuracy or acceptable false-positive-rate evidence. Remote feeds, redirect network checks and verified campaign memory were unavailable; the replay did not access those services or write scans.", '',
        '## Observed latency', '',
        f"Current offline full scanner: median {float(np.median(times)):.2f} ms, p95 {float(np.quantile(times, .95)):.2f} ms, p99 {float(np.quantile(times, .99)):.2f} ms, first request {times[0]:.2f} ms. Excluding that first request, warm median {float(np.median(times[1:])):.2f} ms and p95 {float(np.quantile(times[1:], .95)):.2f} ms. This is a shared-laptop observation, not an isolated CPU latency certification.", '',
        '| Stage | Warm median wall ms | Warm median process CPU ms |', '|---|---:|---:|']
    for stage, timing in warm_stage.items():
        lines.append(f"| {stage} | {timing['wall_ms']:.2f} | {timing['process_cpu_ms']:.2f} |")
    cold_bec = captured[0]['stage_timings'].get('bec_intent', {})
    lines += ['', f"The first request's BEC stage consumed {cold_bec.get('wall_ms', 0):.2f} ms wall time and {cold_bec.get('process_cpu_ms', 0):.2f} ms process CPU time. The difference does not by itself identify disk, scheduling or other stalls. An earlier captured run had a much larger first-request delay; it remains in operational_evidence_check/summary.json rather than being replaced or hidden.", '',
        'Prior fixed-model warm timings, retained from the earlier comparison:', '']
    for model, timing in diagnostics['single_message_model_latency_ms'].items():
        lines.append(f"- {model}: {timing['runs']} single-message runs; median {timing['median']:.2f} ms, p95 {timing['p95']:.2f} ms. Scope: {timing['scope']}.")
    lines += ['', '## Data prepared for review, not claimed clean gold data', '',
        f"New evidence-review corpus: {manifest['records']} retained records; sources {json.dumps(manifest['sources'])}; {manifest['template_groups']} full-message groups with {manifest['full_template_pairs']} qualifying near-template pairs. No full raw-message duplicates were excluded, but near-template overlap exists and must be grouped. There are {manifest['trusted_receiver_authentication_rows']} trusted receiver authentication records. Independent label review and training approval are both false.", '',
        'Brown contributes publisher-identified legitimate security messages and phishing examples, but the latest acquired legitimate example is from 2024. These are excerpts with reuse/extraction questions, not a sufficient recent raw benign corpus. Nazario positives plus older benign text cannot establish representative 2026 performance. Blank review fields have not been filled by pretending independent experts reviewed them.', '',
        'Quality counts: ' + json.dumps(manifest['quality_flags']), '',
        'The review CSV and original evidence references support a practical next step: adjudicate ambiguous spam separately from phishing/BEC, verify message extraction, secure sufficient recent raw benign messages and positives across independent origins, and freeze source/time evaluation before model experiments. Legitimate security alerts, newsletters, invoices and relay senders must be represented as reviewed hard negatives.', '',
        '## Work still required before Email can pass', '',
        '1. Complete evidence-backed task/extraction review and obtain representative both-class sources with usable trusted time provenance. Unknown labels remain unknown. Publisher/category labels alone do not constitute independent project review.',
        '2. Build reviewed train/validation/source-test/time-test partitions through the implemented admission checks; do not train on already inspected external failures or misrepresent them as untouched evaluation.',
        '3. Compare the frozen existing model with a simple regularized text baseline and corrected LightGBM under identical admitted inputs and budgets. Only change representations or architecture when an ablation demonstrates benefit. Remove source cues through appropriate data/representation design rather than a hardcoded Enron blacklist.',
        '4. Require reliable unseen-source/time discrimination, acceptable measured false alarms and missed attacks, and calibrated probabilities before promotion. Record threshold budgets, confidence intervals, hard-negative subgroups, complete-message coverage and inference time. A small sample with zero errors does not prove a very low population error rate.',
        '5. After that gate passes, fit independent calibration for the unchanged full production decision function, bind it to the scorer/model/runtime hashes and rerun the complete Email engine. Marginal or class-conditional conformal coverage requires its stated exchangeability assumptions; neither is an FDR guarantee. The current production calibration gate intentionally has no legacy shortcut.', '',
        'The generalization issue remains unresolved. Engineering repairs are verified; trustworthy replacement training data and a validated replacement model are not yet established. Restarting the application loads the repaired code, but it does not make the old model reliable or restore stale calibration.', '',
        '## Evidence and reproduction', '',
        'Key machine-readable artifacts: results.json, failure_attribution.json, archive_footer_probe.jsonl, provenance_and_overlap.json, upstream_label_matches.jsonl, verified_original_message_overlap.json, full_message_overlap_review.jsonl, operational_evidence_check_v3/results.jsonl and tests.xml. Original failing decisions remain in ../email_retraining_20261003/decisions.csv.', '',
        'Generate this report with `python scripts/report_email_generalization_audit.py` using the project dependencies. It recomputes classifier metrics from per-message predictions, checks protected hashes, verifies replay identity, checks the recorded footer flips and parses actual test results. It does not fit any model or calibration.', '',
        'Primary source context: [SpamAssassin corpus documentation](https://spamassassin.apache.org/old/publiccorpus/readme.html), [Brown Phish Bowl](https://it.brown.edu/alerts/phish-bowl), [PhishFuzzer repository](https://github.com/DataPhish/PhishFuzzer). Acquisition records and raw snapshots retain URLs, hashes and extraction flags.', '',
        f"Current pipeline: `{current_identity}`. Saved calibration pipeline: `{protocol['saved_calibration_pipeline_id']}`. Protected datasets, deployed models and saved calibration all match the audit-start hashes."]
    evidence_files = [OUT / name for name in ('results.json', 'failure_attribution.json', 'archive_footer_probe.jsonl', 'provenance_and_overlap.json', 'protocol.json', 'tests.xml')]
    evidence_files += [REPLAY / name for name in ('protocol.json', 'results.jsonl', 'summary.json')]
    evidence_files += [ROOT / 'benchmarks/email_retraining_20261003/decisions.csv']
    (OUT / 'report_verification.json').write_text(json.dumps({'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'pipeline_id': current_identity, 'protected_unchanged': protected, 'tests': tests,
        'recomputed_metrics': measured, 'evidence_sha256': {str(p.relative_to(ROOT)): sha(p) for p in evidence_files},
        'warm_stage_timings': warm_stage, 'complete_email_reliability': False}, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(OUT / 'REPORT.md'), 'tests': tests, 'protected_unchanged': all(protected.values()), 'pipeline_id': current_identity}))


if __name__ == '__main__':
    main()
