"""Measure cold/warm Email scoring without training, calibration or DB writes.

Timers wrap existing callables in this process only. Import and nested-call
durations overlap; do not add them together as independent stages.
"""
import argparse
import builtins
import ctypes
from ctypes import wintypes
import functools
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('peak_ws', 'ws', 'peak_paged', 'paged',
        'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]


def memory():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_mib': c.ws / 2**20, 'private_committed_mib': c.private / 2**20,
            'peak_working_set_mib': c.peak_ws / 2**20,
            'peak_process_commit_mib': c.peak_pagefile / 2**20}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = []
    state = threading.local()
    started = time.perf_counter()
    report = {'scope': 'Same synthetic linkless Email, same scorer, one CPU process; no accuracy estimate',
              'complete': False, 'nested_durations_overlap': True, 'trace': trace, 'scans': []}
    paths = [ROOT / 'models' / n for n in ('phish_model.pkl', 'vectorizer.pkl', 'url_model.pkl', 'domain_model.pkl', 'conformal_params.json')]
    paths += [ROOT / 'models/conformal_evaluation_results.json',
              ROOT / 'data/clean_email_corpus.csv',
              ROOT / 'data/email_evidence_review_20261003/review_corpus.jsonl']
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report['artifact_hashes_before'] = before
    source_paths = [ROOT / name for name in (
        'services/scan_engine.py', 'services/conformal_evidential_engine.py',
        'services/security_pipeline.py', 'services/bec_detector.py',
        'services/vector_store.py', 'services/email_runtime.py',
        'scripts/profile_email_runtime.py')]
    source_before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths}
    report['source_hashes_before'] = source_before

    def save():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def timed(name, fn):
        @functools.wraps(fn)
        def call(*a, **kw):
            previous = getattr(state, 'parent', None)
            event = {'name': name, 'parent': previous, 'start_seconds': time.perf_counter() - started,
                     'scan': getattr(state, 'scan', None)}
            event_id = len(trace)
            trace.append(event)
            state.parent = event_id
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return fn(*a, **kw)
            except Exception as e:
                event['error_type'] = type(e).__name__
                raise
            finally:
                event.update({'wall_seconds': time.perf_counter()-wall,
                              'process_cpu_seconds': time.process_time()-cpu, 'memory_after': memory()})
                state.parent = previous
                save()
                print(json.dumps({k: event[k] for k in ('name', 'scan', 'wall_seconds', 'process_cpu_seconds')}), flush=True)
        return call

    original_import = builtins.__import__
    monitored = {'torch', 'transformers', 'sentence_transformers', 'chromadb', 'onnxruntime'}
    observed_imports = set()
    def observed_import(name, globals=None, locals=None, fromlist=(), level=0):
        label = name
        if name == 'transformers' and 'pipeline' in (fromlist or ()):
            label += ':pipeline'
        if name in monitored and label not in observed_imports:
            observed_imports.add(label)
            return timed('import:' + label, original_import)(name, globals, locals, fromlist, level)
        return original_import(name, globals, locals, fromlist, level)
    builtins.__import__ = observed_import
    try:
        def import_scanner():
            from services import scan_engine
            return scan_engine
        engine = timed('scanner_import', import_scanner)()
        from services import bec_detector, vector_store
        for module, names in (
            (engine, ['load_ml_model', 'score_content', 'score_urls', 'score_headers', 'build_threat_intel']),
            (vector_store, ['_get_embedding_model', '_get_chroma_collection', 'search_similar_campaigns']),
            (bec_detector, ['_get_bec_classifier', 'predict_bec_intent'])):
            for name in names:
                setattr(module, name, timed(module.__name__ + '.' + name, getattr(module, name)))
        content = 'This is a synthetic local application test. The team meeting is tomorrow at ten. Please bring the project notes. No action on any account is requested.'
        subject = 'Synthetic meeting reminder'
        for run in ('cold', 'warm_1', 'warm_2'):
            state.scan = run
            result = timed('evaluate_email_security', engine.evaluate_email_security)(
                content, sender='tester@example.invalid', subject=subject)
            snapshot = {key: result.get(key) for key in ('content_result', 'url_result', 'header_result', 'bec_result', 'conformal_res', 'sensor_errors')}
            report['scans'].append({'run': run, 'observed': snapshot})
            save()
        report['score_outputs_identical'] = all(
            entry['observed'] == report['scans'][0]['observed'] for entry in report['scans'][1:])
        report['pipeline_id'] = engine.security_pipeline_id()
        report['required_components_available'] = (
            engine.ML_MODEL is not None and vector_store._EMBEDDING_MODEL not in (None, False)
            and vector_store._COLLECTION is not None and bec_detector._BEC_PIPELINE is not None)
        report['measurement_valid'] = report['required_components_available'] and all(
            not row['observed'].get('sensor_errors') for row in report['scans'])
        report['artifact_hashes_after'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['protected_artifacts_unchanged'] = report['artifact_hashes_after'] == before
        report['source_hashes_after'] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths}
        report['sources_unchanged_during_measurement'] = report['source_hashes_after'] == source_before
        report['measurement_valid'] = report['measurement_valid'] and report['sources_unchanged_during_measurement']
        report['complete'] = True
    finally:
        builtins.__import__ = original_import
        save()
    print('Report saved: ' + str(args.output / 'results.json'), flush=True)


if __name__ == '__main__':
    main()
