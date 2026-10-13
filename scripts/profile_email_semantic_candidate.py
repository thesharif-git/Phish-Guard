"""Isolated CPU process profiles on identical, fixed Email message inputs."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from scripts.profile_email_runtime import memory
from scripts.run_email_semantic_comparison import digest, save, DEFAULT_ASSETS, REFERENCE, REFERENCE_DIRECTORY
from utils.jsonl import read_jsonl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=('semantic','text_reference','deployed'),required=True)
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    directory=args.experiment.resolve()
    directory.relative_to(ROOT/'benchmarks')
    protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    result=json.loads((directory/'results.json').read_text(encoding='utf-8'))
    if not result.get('complete') or result['protocol_sha256']!=digest(directory/'protocol.json'):
        raise ValueError('Require a completed hash-bound semantic experiment')
    brown_path=ROOT/'data/brown_email_benchmark_20261005/all_records.jsonl'
    rows=[r for r in read_jsonl(brown_path) if r['eligible_for_scoring']]
    spanish=[r for r in read_jsonl(REFERENCE_DIRECTORY/'partition_rows.jsonl')
             if r['pool']==protocol['training_pool'] and r['split']=='published_day_test']
    # Fixed selection by message ID, never by model score; paired timing inputs.
    rows+=sorted(spanish,key=lambda r:r['id'])[:32]
    started=time.perf_counter()
    before=memory()
    paths=[brown_path,REFERENCE_DIRECTORY/'partition_rows.jsonl',directory/'protocol.json',directory/'results.json',Path(__file__)]
    if args.model=='semantic':
        from training.email_semantic import FrozenEmailEncoder, content_text
        paths+=[directory/'candidate.pkl',Path(protocol['assets_directory'])/'manifest.json']
        artifact=joblib.load(directory/'candidate.pkl')
        if artifact['protocol_sha256']!=result['protocol_sha256']:
            raise ValueError('Candidate protocol mismatch')
        encoder=FrozenEmailEncoder(protocol['assets_directory'])
        classifier=artifact['classifier']
        def predict(row):
            embedding,_=encoder.encode(content_text(row['email_input']))
            return float(classifier.predict_proba(embedding[None,:])[0,list(classifier.classes_).index(1)])
    elif args.model=='text_reference':
        from training.email_evidence_model import predict_email_evidence_probabilities
        path=REFERENCE_DIRECTORY/(REFERENCE+'.pkl')
        paths+=[path]
        artifact=joblib.load(path)
        model=artifact['model']
        def predict(row):
            return float(predict_email_evidence_probabilities(model,[row['email_input']])[0])
    else:
        from scripts.compare_email_representations import score_baseline
        paths+=[ROOT/'models/phish_model.pkl',ROOT/'models/vectorizer.pkl']
        model=joblib.load(ROOT/'models/phish_model.pkl')
        vectorizer=joblib.load(ROOT/'models/vectorizer.pkl')
        for fold in getattr(model,'calibrated_classifiers_',[]):
            fold.estimator.set_params(n_jobs=1)
        def predict(row):
            record=row['email_input']
            return float(score_baseline(model,vectorizer,[record['body']+' '+record['subject']])[0])
    load_seconds=time.perf_counter()-started
    after_load=memory()
    hashes={str(p):digest(p) for p in paths}
    traces=[]
    for index,row in enumerate(rows):
        started=time.perf_counter()
        value=predict(row)
        traces.append(dict(row_id=row['id'],input_contract_sha256=row['input_contract_sha256'],
                           probability=value if np.isfinite(value) else None,
                           wall_ms=(time.perf_counter()-started)*1000))
    first_pass=traces
    traces=[]
    # Repeat the same varied messages after first pass; no synthetic short-only benchmark.
    for row in rows:
        started=time.perf_counter()
        value=predict(row)
        traces.append(dict(row_id=row['id'],input_contract_sha256=row['input_contract_sha256'],
                           probability=value if np.isfinite(value) else None,
                           wall_ms=(time.perf_counter()-started)*1000))
    timings=[r['wall_ms'] for r in traces]
    if any(digest(path)!=expected for path,expected in hashes.items()):
        raise ValueError('Profile inputs changed during measurement')
    output=dict(complete=True,model=args.model,isolated_process=True,
        scope='65 identical original message contracts: 33 Brown eligible English excerpts + 32 Spanish publisher-day messages selected by ID',
        timing_scope='preprocessing + representation + classifier; excludes ingestion, network checks and full OriginTrace',
        load_wall_seconds=load_seconds,load_scope='model/runtime loading; excludes interpreter startup and common research imports',
        before_load=before,after_load=after_load,after_warm_inference=memory(),
        first_pass=first_pass,warm_predictions=traces,input_hashes=hashes,
        warm_timing=dict(rows=len(traces),median_ms=float(np.median(timings)),mean_ms=float(np.mean(timings)),
                         p95_ms=float(np.quantile(timings,.95)),p99_ms=float(np.quantile(timings,.99))),
        approved_for_deployment=False)
    save(directory/('profile_'+args.model+'.json'),output)
    print(json.dumps(dict(model=args.model,warm_timing=output['warm_timing'],memory=output['after_warm_inference'])),flush=True)


if __name__=='__main__':
    main()
