"""Real Email models/sensors and final rendering; no refitting or database writes."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import json
import hashlib
import time
import warnings
from collections import Counter
import requests
from data import store
from services import scan_engine as engine, vector_store
from services.security_pipeline import security_pipeline_id
from services.conformal_evidential_engine import ConformalPredictor
import utils.threat_intel as intel

OUT = ROOT / 'benchmarks/email_generalization_root_20261003/operational_evidence_check_v3'


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus_path = ROOT / 'data/email_evidence_review_20261003/review_corpus.jsonl'
    from utils.jsonl import read_jsonl
    rows = read_jsonl(corpus_path)
    selected = []
    # Bounded integration check by source class/order, never prediction or correctness.
    for source, label, count in [('phishfuzzer_manual', 'Phishing', 6), ('phishfuzzer_manual', 'Spam', 6),
                                  ('phishfuzzer_manual', 'Valid', 6), ('brown_phish_bowl', 'publisher_verified_nonphishing', 12),
                                  ('brown_phish_bowl', 'publisher_reported_phishing', 12)]:
        selected += [r for r in rows if r['source'] == source and r['source_claimed_label'] == label and len(r['body'].strip()) >= 40][:count]
    identity = security_pipeline_id()
    protocol = {'pipeline_id': identity, 'corpus_sha256': hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        'rows': len(selected), 'selected_ids': [r['id'] for r in selected],
        'scope': 'Bounded real-model final-result integration check; not a new accuracy benchmark or complete-message certification',
        'network': 'Disabled; redirects and remote reputation unavailable, never claimed checked',
        'database_writes': False, 'calibration_fitted': False, 'model_retrained': False,
        'saved_calibration_pipeline_id': json.loads((ROOT / 'models/conformal_params.json').read_text())['pipeline_id']}
    if (OUT / 'protocol.json').exists():
        assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    else:
        save('protocol.json', protocol)
    def no_network(*args, **kwargs):
        raise RuntimeError('Network unavailable in offline evidence check')
    requests.sessions.Session.request = no_network
    engine.resolve_short_url = lambda value: value  # Unknown redirect, not a verified final destination.
    intel.check_virustotal = lambda value: {'available': False, 'reason': 'No remote feed snapshot in offline evidence check'}
    vector_store.search_similar_campaigns = lambda text: {'available': False, 'has_match': False, 'matches': [],
        'verified_campaign_evidence': False, 'reason': 'No independently verified campaign-memory dataset'}
    store.is_blocked = lambda *args, **kwargs: False
    store.add_scan = lambda *args, **kwargs: None
    engine.load_ml_model()
    for model in (engine.URL_MODEL, engine.DOMAIN_MODEL):
        for fold in getattr(model, 'calibrated_classifiers_', []):
            fold.estimator.set_params(n_jobs=1)
    import torch
    torch.set_num_threads(1)
    stages = {}
    def timed(name, function):
        def call(*a, **kw):
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return function(*a, **kw)
            finally:
                stages[name] = {'wall_ms': (time.perf_counter()-wall)*1000, 'process_cpu_ms': (time.process_time()-cpu)*1000}
        return call
    for name in ('score_content', 'score_urls', 'score_headers', 'build_threat_intel'):
        setattr(engine, name, timed(name, getattr(engine, name)))
    from services import bec_detector
    bec_detector.predict_bec_intent = timed('bec_intent', bec_detector.predict_bec_intent)
    output = OUT / 'results.jsonl'
    completed = read_jsonl(output) if output.exists() else []
    completed_ids = {r['record_id'] for r in completed}
    for row in selected:
        if row['id'] in completed_ids:
            continue
        started = time.perf_counter()
        stages.clear()
        result = engine.scan_email(row['body'], None, sender=row['sender'], subject=row['subject'], headers=row['headers'],
            html_body=row['html_body'], supplementary_urls=[u for u in row.get('source_url_metadata', []) if u.startswith(('http://', 'https://'))],
            parsing_info={'text_complete': True, 'source_kind': row['input_form']})
        elapsed = (time.perf_counter()-started)*1000
        issues = []
        if result['conformal_guarantee']['calibration_valid']:
            issues.append('Old calibration incorrectly accepted after changed scorer')
        if result['security_action'] != 'WARNING_BANNER':
            issues.append('Stale calibration allowed an automated decision')
        if 'review' not in result['explanation']['summary'].lower():
            issues.append('Final summary contradicts review action')
        if result['header_authentication'].get('verified'):
            issues.append('Untrusted source headers presented as verified authentication')
        if any('URL Threat Feed' == m.get('source') for m in result['threat_intelligence']['matches']):
            issues.append('Local list incorrectly presented as live feed')
        if result['action_execution']['executed']:
            issues.append('Offline scanner claims an action was executed')
        semantic = result['pillars']['pillar3_semantic']
        if semantic.get('threat_probability_validated') is not False:
            issues.append('Heuristic semantic score presented as a validated probability')
        if any('Executive Coercion:' in s or 'Visual Quishing Deception:' in s for s in semantic.get('signals', [])):
            issues.append('Topic/QR observation presented as established deception')
        infrastructure = result['pillars']['pillar2_infrastructure']
        if infrastructure.get('threat_probability_validated') is not False:
            issues.append('Infrastructure review score presented as validated probability')
        urls = result['url_intelligence']
        if urls.get('has_redirect') != any(u.get('resolved_url') for u in urls.get('analyzed_urls', [])):
            issues.append('Redirect claimed without observed destination change')
        if any('classic phishing pressure tactic' in s for s in result['explanation']['risk_signals']):
            issues.append('Topic match presented as established manipulation')
        captured = {'record_id': row['id'], 'source': row['source'], 'source_claimed_label': row['source_claimed_label'],
            'input_quality_flags': row['quality_flags'], 'elapsed_ms': elapsed, 'stage_timings': dict(stages), 'verification_issues': issues, 'result': result}
        # Full result sections retained. No invented probability for unavailable/OOV rows.
        with output.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(captured, ensure_ascii=False, allow_nan=False)+'\n')
        completed.append(captured)
        save('run_status.json', {'state': 'running', 'done': len(completed), 'total': len(selected)})
        print(str(len(completed))+'/'+str(len(selected)), flush=True)
    import numpy as np
    timing = np.array([r['elapsed_ms'] for r in completed])
    summary = {'complete': len(completed) == len(selected), 'rows': len(completed), 'pipeline_id': identity,
        'source_label_counts': dict(Counter(str((r['source'], r['source_claimed_label'])) for r in completed)),
        'verification_issues': sum(len(r['verification_issues']) for r in completed),
        'actions': dict(Counter(r['result']['security_action'] for r in completed)),
        'calibration_valid_rows': sum(r['result']['conformal_guarantee']['calibration_valid'] for r in completed),
        'latency_ms': {'median': float(np.median(timing)), 'p95': float(np.quantile(timing, .95)), 'p99': float(np.quantile(timing, .99)),
                       'cold_first': float(timing[0]), 'scope': 'Observed full offline scanner including cached model initialization; shared laptop, not CPU-only certification'},
        'accuracy_claim': 'None: review-only stale-calibration policy and unadjudicated extraction are not a classifier accuracy measurement',
        'models_or_calibration_modified': False}
    save('summary.json', summary)
    save('run_status.json', {'state': 'completed', 'done': len(completed), 'total': len(selected)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save('run_status.json', {'state': 'failed', 'error': repr(error)})
        raise
