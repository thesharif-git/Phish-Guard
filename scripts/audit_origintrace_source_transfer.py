"""Supplementary source-transfer diagnostic; produces research results only.

Uses already-inspected legacy data. This is NOT a fresh external benchmark.
Trains only temporary in-memory logistic baselines; no production artifacts changed.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
import sys
import time
from urllib.parse import urlparse
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'benchmarks'/'root_cause_audit_20261002'
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
from threadpoolctl import threadpool_limits
from training.train_url_and_domain_models import extract_domain_features
from utils.feature_extractor import extract_features
from utils.text_sanitizer import clean_and_mask_email_text

warnings.filterwarnings('ignore',message='X does not have valid feature names')


def metrics(y,p,t=.5):
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=t,labels=[0,1]).ravel())
    return {'n':len(y),'tn':tn,'fp':fp,'fn':fn,'tp':tp,
            'recall':tp/(tp+fn) if tp+fn else None,'fpr':fp/(tn+fp) if tn+fp else None,
            'accuracy':(tn+tp)/len(y),'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'threshold':float(t) if np.isfinite(t) else None}


def sha(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return d.hexdigest()


result={'purpose':'Source-transfer development diagnostic, not new final evaluation',
        'input_sha256':{k:sha(ROOT/f'data/clean_{k}_corpus.csv') for k in ['url','email']},
        'script_sha256':sha(__file__),'experiments':{}}
with threadpool_limits(limits=1):
    df=pd.read_csv(ROOT/'data/clean_url_corpus.csv')
    df['publisher']=df.source.map(lambda s:'MaliciousPhish' if s.startswith('MaliciousPhish') else s)
    tr=df[df.split=='train'];te=df[df.split=='test']
    priors=tr.groupby('publisher').label.mean()
    result['publisher_only_baseline']=metrics(te.label.values,te.publisher.map(priors).values)
    # Compare identical basic lexical baseline under mixed-publisher and publisher-held-out training.
    for mode in ['mixed_publishers','maliciousphish_only']:
        train=df[(df.split=='train') & ((df.publisher=='MaliciousPhish') if mode.endswith('_only') else True)].copy()
        val=df[(df.split=='val') & ((df.publisher=='MaliciousPhish') if mode.endswith('_only') else True)].copy()
        excluded=set(train.psl_domain)|set(val.psl_domain)
        test=df[(df.split=='test') & ~df.psl_domain.isin(excluded)].copy()
        vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=12000,min_df=2,sublinear_tf=True)
        # Scheme removal is identical in both experiments; paths and host characters remain.
        text=lambda x:x.str.replace(r'^https?://','',regex=True)
        x=vectorizer.fit_transform(text(train.url)); xv=vectorizer.transform(text(val.url));xt=vectorizer.transform(text(test.url))
        model=LogisticRegression(C=1.,max_iter=500,solver='liblinear',random_state=42)
        start=time.perf_counter();model.fit(x,train.label)
        pv=model.predict_proba(xv)[:,1];pt=model.predict_proba(xt)[:,1]
        fpr,tpr,thresholds=roc_curve(val.label,pv,drop_intermediate=False)
        eligible=np.flatnonzero((fpr<=.005)&np.isfinite(thresholds))
        threshold=float(thresholds[eligible[np.argmax(tpr[eligible])]]) if len(eligible) else float('inf')
        entry={'training_n':len(train),'validation_n':len(val),'fit_seconds':time.perf_counter()-start,
               'test_at_05':metrics(test.label.values,pt),'test_at_validation_fpr_005':metrics(test.label.values,pt,threshold),
               'by_publisher':{p:{'test_at_05':metrics(test.loc[test.publisher==p,'label'].values,pt[test.publisher.values==p]),
                                  'test_at_validation_fpr_005':metrics(test.loc[test.publisher==p,'label'].values,pt[test.publisher.values==p],threshold)} for p in sorted(test.publisher.unique())}}
        result['experiments'][mode]=entry
        print(mode,json.dumps(entry['test_at_05']),flush=True)
        (OUT/'source_transfer.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    # Actual label admission rule contradicts the source registry: all input rows called malicious.
    raw=Path(r'C:\Users\shari\Desktop\3_malicious_domains.csv')
    rows=list(csv.DictReader(raw.open(encoding='utf-8-sig',newline='')))
    result['domain_reputation_admission']={'raw_file_sha256':sha(raw),'n':len(rows),
        'zero_malicious_votes':sum(float(r['Malicious_Votes'] or 0)==0 for r in rows),
        'positive_reputation_rows':sum(float(r['Reputation'] or 0)>0 for r in rows),
        'negative_reputation_rows':sum(float(r['Reputation'] or 0)<0 for r in rows),
        'note':'These fields do not themselves establish phishing ground truth. Curator assigns label=1 to every admitted row without interpreting them.'}
    # Root URLs are routed away from the URL specialist in utils.url_scanner.scan_url.
    test=df[df.split=='test'].copy(); y=test.label.values
    model=joblib.load(ROOT/'models/domain_model.pkl')
    for c in model.calibrated_classifiers_: c.estimator.set_params(n_jobs=1)
    host_features=pd.DataFrame([extract_domain_features(u) for u in test.url])
    root_features=pd.DataFrame([extract_domain_features(u) for u in test.psl_domain])
    ph=model.predict_proba(host_features)[:,1]; pr=model.predict_proba(root_features)[:,1]
    parsed=[urlparse(u if u.startswith('http') else 'http://'+u) for u in test.url]
    no_path=np.array([p.path in ('','/') and not p.query for p in parsed])
    saved=np.load(OUT/'url_predictions.npz')
    expected=test.index.values
    url_probs=saved['probability'][expected]
    result['root_url_routing']={'n_root_urls':int(no_path.sum()),
        'url_specialist_at_05':metrics(y[no_path],url_probs[no_path]),
        'domain_specialist_on_url_host_at_05':metrics(y[no_path],ph[no_path]),
        'scope':'Model-stage routing comparison on URL-labelled development test. Does not include full online engine.'}
    result['domain_train_serve_scope_shift']={'n_test_urls':len(test),
        'changed_05_decisions':int(((ph>=.5)!=(pr>=.5)).sum()),
        'mean_absolute_probability_change':float(np.abs(ph-pr).mean()),
        'p95_absolute_probability_change':float(np.percentile(np.abs(ph-pr),95)),
        'max_absolute_probability_change':float(np.abs(ph-pr).max())}
    print('Root routing and domain input scope measured.',flush=True)
    (OUT/'source_transfer.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    # Preserve the exact calibration mathematics while avoiding sklearn metadata reconstruction.
    # Research comparison only; private calibration attributes are not a production API contract.
    emails=pd.read_csv(ROOT/'data/clean_email_corpus.csv'); emails=emails[emails.split=='test'].head(10)
    email_model=joblib.load(ROOT/'models/phish_model.pkl');vec=joblib.load(ROOT/'models/vectorizer.pkl')
    for c in email_model.calibrated_classifiers_: c.estimator.set_params(n_jobs=1)
    matrices=[hstack([vec.transform([clean_and_mask_email_text(r.raw_text)]),csr_matrix(extract_features(r.raw_text))]).tocsr() for _,r in emails.iterrows()]
    def direct_calibrated(x):
        return float(np.mean([c.calibrators[0].predict(c.estimator.booster_.predict(x,num_threads=1))[0] for c in email_model.calibrated_classifiers_]))
    agreement=max(abs(float(email_model.predict_proba(x)[0,1])-direct_calibrated(x)) for x in matrices)
    timings={}
    for name,func in [('sklearn_calibrated_wrapper',lambda x:float(email_model.predict_proba(x)[0,1])),('direct_booster_with_same_sigmoid_calibrators',direct_calibrated)]:
        wall=[];cpu=[]
        for x in matrices: func(x)
        for _ in range(2):
            for x in matrices:
                t0=time.perf_counter(); c0=time.process_time();func(x)
                wall.append((time.perf_counter()-t0)*1000);cpu.append((time.process_time()-c0)*1000)
        timings[name]={'n':len(wall),'wall_median_ms':float(np.median(wall)),'wall_p95_ms':float(np.percentile(wall,95)),
                       'process_cpu_median_ms':float(np.median(cpu)),'process_cpu_p95_ms':float(np.percentile(cpu,95))}
    result['email_prediction_wrapper_ablation']={'max_probability_difference_10_inputs':agreement,'timings':timings,
        'scope':'Warm model prediction only, 20 calls per path, one CPU thread. Excludes preprocessing, remote IO, and production concurrency.'}
result['completed']=True
result['inputs_changed']=[k for k,h in result['input_sha256'].items() if sha(ROOT/f'data/clean_{k}_corpus.csv')!=h]
(OUT/'source_transfer.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print('Source-transfer diagnostic complete.',flush=True)
