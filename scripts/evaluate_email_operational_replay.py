"""Frozen external Email replay: real local sensors, explicitly unavailable remote evidence.

Never connects to inboxes, visits destinations, modifies campaign memory, or fits
anything. Missing sender authentication/attachments cannot be reconstructed from
source JSON. This is an offline operational policy test, not deployment certification.
"""
import json
import time
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_email_retraining_experiment import ROOT, OUT, sha
from services import scan_engine as engine, bec_detector, vector_store
from services.security_pipeline import security_pipeline_id
from services.conformal_evidential_engine import ConformalPredictor
import requests
import utils.threat_intel as intel


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    destination = OUT / 'operational_replay_with_source_links'
    destination.mkdir(exist_ok=True)
    identity = security_pipeline_id()
    from utils.jsonl import read_jsonl
    frozen = read_jsonl(OUT / 'external_inputs.jsonl')
    original = json.loads((ROOT / 'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json').read_text(encoding='utf-8'))
    inventory = {r['review_id']: r for r in read_jsonl(ROOT / 'benchmarks/email_comparison_source_review/comparison_candidates.jsonl')}
    protocol = {'pipeline_id': identity, 'external_input_sha256': sha(OUT / 'external_inputs.jsonl'),
                'rows': len(frozen), 'source_labels': 'Author labels; not independently adjudicated',
                'actual_sensors': ['content ML', 'lexical embedded-link ML', 'cached BEC NLI if locally loadable',
                                  'BEC review rules', 'provenance/contradiction/semantic/infrastructure local logic'],
                'supplementary_url_source': 'Original author URL metadata, passed separately; no invented body/MIME',
                'source_original_sha256': sha(ROOT / 'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json'),
                'unavailable_evidence': ['live URL redirects', 'VirusTotal feeds', 'receiver-verified authentication',
                                         'original MIME/attachments', 'independently verified campaign memory'],
                'network_enabled': False, 'campaign_memory_writes': False,
                'threshold_source': 'Existing corpus validation only; no tuning on external rows',
                'claim': 'Offline policy/latency observation; no guarantee of deployment accuracy or prevention'}
    if (destination / 'protocol.json').exists():
        assert json.loads((destination / 'protocol.json').read_text()) == protocol, 'Frozen replay inputs changed'
    else:
        write(destination / 'protocol.json', protocol)
    def blocked(*args, **kwargs):
        raise RuntimeError('Network disabled in frozen offline Email replay')
    requests.sessions.Session.request = blocked
    engine.resolve_short_url = lambda url: url  # Unknown redirect evidence is declared in protocol, never fabricated.
    intel.check_virustotal = lambda url: {'available': False, 'reason': 'No frozen feed snapshot in this offline replay'}
    vector_store.search_similar_campaigns = lambda text: {'available': False, 'has_match': False, 'matches': [],
            'verified_campaign_evidence': False, 'reason': 'No independently verified campaign corpus in replay'}
    engine.load_ml_model()
    for c in engine.ML_MODEL.calibrated_classifiers_:
        c.estimator.set_params(n_jobs=1)
    for model in (engine.URL_MODEL, engine.DOMAIN_MODEL):
        for c in getattr(model, 'calibrated_classifiers_', []):
            c.estimator.set_params(n_jobs=1)
    import torch
    torch.set_num_threads(1)
    predictor = ConformalPredictor(parameter_path=destination / 'conformal_params.json')
    validation = pd.read_csv(ROOT / 'data/clean_email_corpus.csv', keep_default_na=False)
    validation = validation[validation.split == 'val']
    predictor.calibrate(engine.predict_email_probabilities((validation.raw_text + ' ').tolist()),
                        validation.label.to_numpy(), pipeline_id=identity)
    stage = {}
    def timed(name, function):
        def call(*a, **kw):
            start = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                stage[name] = stage.get(name, 0.0) + (time.perf_counter()-start)*1000
        return call
    for name in ('score_content', 'score_urls', 'score_headers', 'build_threat_intel'):
        setattr(engine, name, timed(name, getattr(engine, name)))
    bec_detector.predict_bec_intent = timed('bec_intent', bec_detector.predict_bec_intent)
    output = destination / 'decisions.jsonl'
    completed = read_jsonl(output) if output.exists() else []
    completed_ids = {r['review_id'] for r in completed}
    for record in frozen:
        if record['review_id'] in completed_ids:
            continue
        stage.clear()
        original_row = original[inventory[record['review_id']]['source_row_index']]
        url_field = original_row.get('URL') or []
        url_list = url_field if isinstance(url_field, list) else [url_field]
        import re
        supplementary = [u for u in url_list if isinstance(u, str) and re.match(r'^https?://[^\s]+$', u, re.I)]
        supplied_files = original_row.get('File') or []
        parsing = {'source_kind': 'author_structured_JSON_not_original_MIME', 'text_complete': True}
        if supplied_files:
            parsing['unscanned_parts'] = supplied_files if isinstance(supplied_files, list) else [supplied_files]
        start = time.perf_counter()
        result = engine.evaluate_email_security(record['body'], sender=record['sender'], subject=record['subject'],
                                               html_body=record['html_body'], predictor=predictor,
                                               supplementary_urls=supplementary, parsing_info=parsing)
        elapsed = (time.perf_counter()-start)*1000
        row = {'review_id': record['review_id'], 'source_label': record['source_label'],
               'probability': result['content_result']['ml_probability'],
               'action': result['security_action'], 'decision_basis': result['conformal_res']['decision_basis'],
               'calibration_valid': result['conformal_res']['calibration_valid'],
               'sensor_errors': result['sensor_errors'], 'urls_found': result['url_result']['urls_found'],
               'url_risk_score': result['url_result']['score'], 'bec_result': result['bec_result'],
               'contradiction': result['contra_result'], 'source_supplementary_urls': supplementary,
               'source_attachment_names_without_payloads': supplied_files,
               'elapsed_ms': elapsed, 'stage_ms': dict(stage)}
        with output.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        completed.append(row)
        if len(completed) % 10 == 0 or len(completed) == len(frozen):
            write(destination / 'run_status.json', {'stage': 'replaying', 'completed': len(completed), 'total': len(frozen)})
            print(f'Offline Email policy replay {len(completed)}/{len(frozen)}', flush=True)
    assert identity == security_pipeline_id()
    measurements = {'complete': True, 'pipeline_id': identity, 'rows': len(completed),
                    'per_source_class': {}, 'decision_bases': dict(Counter(r['decision_basis'] for r in completed)),
                    'model_unavailable_rows': sum(r['probability'] is None for r in completed),
                    'bec_transformer_available_rows': sum(r['bec_result']['transformer_available'] for r in completed),
                    'bec_transformer_truncated_rows': sum(r['bec_result'].get('transformer_text_truncated', False) for r in completed)}
    for cls in ('Phishing', 'Spam', 'Valid'):
        selected = [r for r in completed if r['source_label'] == cls]
        measurements['per_source_class'][cls] = {'rows': len(selected), 'actions': dict(Counter(r['action'] for r in selected))}
    durations = np.array([r['elapsed_ms'] for r in completed])
    measurements['latency_ms'] = {'includes_cold_first_message': True, 'first_message': float(durations[0]),
            'mean': float(durations.mean()), 'median': float(np.median(durations)),
            'p95': float(np.quantile(durations, .95)), 'p99': float(np.quantile(durations, .99))}
    measurements['stage_latency_ms'] = {name: {'mean': float(np.mean([r['stage_ms'].get(name, 0) for r in completed])),
           'median': float(np.median([r['stage_ms'].get(name, 0) for r in completed]))} for name in stage}
    write(destination / 'results.json', measurements)
    write(destination / 'run_status.json', {'stage': 'completed', 'completed': len(completed), 'total': len(frozen)})
    print(json.dumps(measurements, indent=2), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    try:
        main()
    except Exception as error:
        directory = OUT / 'operational_replay_with_source_links'
        if directory.exists():
            write(directory / 'run_status.json', {'stage': 'failed', 'error': repr(error)})
        raise
