"""Offline before/after verification; no fitting, downloads, or calibration writes."""
import ast
import hashlib
import json
import mailbox
import warnings
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'benchmarks/email_ingestion_repairs'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from utils.email_parser import parse_eml_content
    from services import scan_engine as engine
    from services.security_pipeline import security_pipeline_id
    import numpy as np
    import pandas as pd
    manifest = json.loads((OUT / 'before_manifest.json').read_text())
    protected = dict(manifest['protected'])
    protected.update({p: manifest['before_sources'][p] for p in (
        'models/conformal_params.json', 'models/conformal_evaluation_results.json')})
    assert all(digest(ROOT / p) == h for p, h in protected.items())
    old_parser = {'__name__': 'before_email_parser'}
    exec(compile((OUT / 'utils__email_parser.py.before').read_text(encoding='utf-8'), 'before_parser', 'exec'), old_parser)
    counts = Counter()
    ledger = []
    source = ROOT / 'data/external_email_validation/nazario_2024_2025'
    source_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT / r['path']) == r['sha256'] for r in source_manifest['files'])
    for year in (2024, 2025):
        box = mailbox.mbox(source / f'phishing-{year}.mbox', create=False)
        try:
            for key in box.keys():
                raw = box.get_bytes(key, from_=False)
                before = old_parser['parse_eml_content'](raw)
                after = parse_eml_content(raw)
                parsing = after['parsing']
                counts['messages'] += 1
                counts['body_changed'] += before['body'] != after['body']
                counts['html_evidence_changed'] += before.get('html_body', '') != after['html_body']
                counts['body_replacement_characters_before'] += before['body'].count('\ufffd')
                counts['body_replacement_characters_after'] += after['body'].count('\ufffd')
                counts['messages_with_explicit_text_errors'] += bool(parsing['errors'])
                counts['messages_with_unscanned_parts'] += bool(parsing['unscanned_parts'])
                counts.update(parsing['notes'])
                counts.update('error:' + e for e in parsing['errors'])
                ledger.append({'source_year': year, 'source_key': key, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                    'before_body_sha256': hashlib.sha256(before['body'].encode()).hexdigest(),
                    'after_body_sha256': hashlib.sha256(after['body'].encode()).hexdigest(),
                    'parsing': parsing, 'gold_security_label': None})
        finally:
            box.close()
    (OUT / 'raw_message_verification.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ledger), encoding='utf-8')
    print('Verified all original raw messages: ' + json.dumps(counts), flush=True)
    engine.load_ml_model()
    if engine.ML_MODEL is None or engine.TFIDF_VECTORIZER is None:
        raise RuntimeError('Deployed legacy email artifacts unavailable: ' + str(engine.MODEL_LOAD_ERRORS))
    for calibrated in engine.ML_MODEL.calibrated_classifiers_:
        calibrated.estimator.set_params(n_jobs=1)  # Memory only, never persist.
    old_scope = dict(vars(engine))
    old_source = ast.parse((OUT / 'services__scan_engine.py.before').read_text(encoding='utf-8'))
    node = next(n for n in old_source.body if isinstance(n, ast.FunctionDef) and n.name == 'predict_email_probabilities')
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'before_prediction', 'exec'), old_scope)
    df = pd.read_csv(ROOT / 'data/clean_email_corpus.csv')
    replay = {}
    started = time.perf_counter()
    for split in ('val', 'test'):
        texts = (df.loc[df.split == split, 'raw_text'].astype(str) + ' ').tolist()
        checked = 0
        changed = 0
        unavailable = []
        max_delta = 0.0
        for offset in range(0, len(texts), 128):
            batch = texts[offset:offset + 128]
            old = old_scope['predict_email_probabilities'](batch)
            try:
                new = engine.predict_email_probabilities(batch)
                delta = np.abs(old - new)
                changed += int(np.count_nonzero(delta))
                max_delta = max(max_delta, float(delta.max()))
                checked += len(batch)
            except ValueError:
                # Locate unsupported rows instead of silently dropping a batch.
                for local_index, text in enumerate(batch):
                    try:
                        new = engine.predict_email_probabilities([text])[0]
                        delta = abs(float(old[local_index]) - float(new))
                        max_delta = max(max_delta, delta)
                        changed += delta != 0
                        checked += 1
                    except ValueError as e:
                        unavailable.append({'split_position': offset + local_index, 'error': str(e)})
        replay[split] = {'rows': len(texts), 'valid_predictions_compared': checked, 'changed_predictions': int(changed),
                        'max_probability_difference': max_delta, 'unavailable': unavailable}
        print('Held-out fixed-text replay: ' + split + ' ' + json.dumps(replay[split]), flush=True)
    params = json.loads((ROOT / 'models/conformal_params.json').read_text())
    current_id = security_pipeline_id()
    result = {'complete': True, 'raw_message_counts': dict(counts), 'fixed_text_replay': replay,
              'replay_wall_seconds': time.perf_counter() - started,
              'current_pipeline_id': current_id, 'saved_calibration_pipeline_id': params['pipeline_id'],
              'saved_calibration_matches_current_pipeline': params['pipeline_id'] == current_id,
              'calibration_refitted_or_rebound': False, 'trained_models_or_corpora_modified': False,
              'independent_phishing_accuracy_established': False,
              'source_mailbox_hashes': {r['path']: r['sha256'] for r in source_manifest['files']},
              'protected_hashes': protected}
    assert all(digest(ROOT / p) == h for p, h in protected.items())
    (OUT / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('Verification completed; calibration intentionally unchanged.', flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    main()
