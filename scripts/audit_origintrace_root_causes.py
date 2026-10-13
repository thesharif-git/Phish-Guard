"""Offline, reproducible diagnostics. Never replaces production models or calls networks.

Run with compatible Python and the project's site-packages on sys.path.
Outputs development diagnostics, NOT a fresh final benchmark.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import types
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'benchmarks' / 'root_cause_audit_20261002'
OUT.mkdir(parents=True, exist_ok=True)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from utils.feature_extractor import extract_features
from utils.text_sanitizer import clean_and_mask_email_text

SOURCES = [
    'scripts/clean_and_curate_master_datasets.py', 'scripts/train_production_models.py',
    'training/train_url_and_domain_models.py', 'utils/text_sanitizer.py',
    'utils/feature_extractor.py', 'utils/url_scanner.py', 'services/scan_engine.py',
    'services/origintrace_pillars.py', 'services/bec_detector.py',
    'services/origintrace_inference_pipeline.py', 'routers/api.py',
    'scripts/verify_production_engine.py', 'scripts/deep_pipeline_diagnostic.py',
    'data/clean_email_corpus.csv', 'data/clean_url_corpus.csv', 'data/clean_domain_corpus.csv',
    'models/phish_model.pkl', 'models/vectorizer.pkl', 'models/url_model.pkl', 'models/domain_model.pkl',
]


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pure_namespace(path, funcs=(), constants=(), classes=()):
    """Extract named pure functions/classes from the actual code, avoiding import side effects."""
    text = (ROOT / path).read_text(encoding='utf-8-sig')
    ns = dict(re=re, math=math, np=np, pd=pd, os=os, Counter=Counter,
              defaultdict=defaultdict, urlparse=urlparse, csr_matrix=csr_matrix,
              hstack=hstack, Dict=Dict, Any=Any, List=List, Optional=Optional,
              Tuple=Tuple, hashlib=hashlib, __file__=str(ROOT/path))
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in funcs:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/path), 'exec'), ns)
    return ns


def metrics(y, probs, threshold=.5):
    y=np.asarray(y, dtype=int); probs=np.asarray(probs, dtype=float)
    tn,fp,fn,tp=map(int,confusion_matrix(y, probs>=threshold, labels=[0,1]).ravel())
    result=dict(n=len(y), threshold=float(threshold) if np.isfinite(threshold) else None,
                tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/len(y),
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(tn+fp) if tn+fp else None,
                precision=tp/(tp+fp) if tp+fp else None,
                brier=float(brier_score_loss(y,probs)),
                log_loss=float(log_loss(y,probs,labels=[0,1])))
    if len(set(y))==2:
        result.update(roc_auc=float(roc_auc_score(y,probs)),
                      average_precision=float(average_precision_score(y,probs)))
    else:
        result.update(roc_auc=None,average_precision=None)
    bins=[]; ece=0.
    for i in range(10):
        mask=(probs>=i/10)&((probs<(i+1)/10) if i<9 else probs<=1)
        if mask.any():
            observed=float(y[mask].mean()); predicted=float(probs[mask].mean())
            ece+=float(mask.mean())*abs(observed-predicted)
            bins.append(dict(n=int(mask.sum()),predicted=predicted,observed=observed))
    result.update(ece_10_equal_width=ece,reliability_bins=bins)
    return result


def threshold_on_validation(y,p,budget=.005):
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    eligible=np.flatnonzero((fpr<=budget)&np.isfinite(thresholds))
    if not len(eligible): return float('inf')
    return float(thresholds[eligible[np.argmax(tpr[eligible])]])


def pin(model):
    estimators=[c.estimator for c in model.calibrated_classifiers_] if hasattr(model,'calibrated_classifiers_') else [model]
    for est in estimators:
        if hasattr(est,'set_params'): est.set_params(n_jobs=1)
    return model


def save():
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


def checkpoint(message):
    print(message,flush=True); save()


started=time.time()
report={'purpose':'Root cause development diagnosis; no fresh holdout and no deployment changes',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                       'lightgbm':lgb.__version__,'threads':1},
        'input_sha256':{p:sha(ROOT/p) for p in SOURCES},
        'audit_script_sha256':sha(__file__),'results':{}}
frames={kind:pd.read_csv(ROOT/f'data/clean_{kind}_corpus.csv') for kind in ['email','url','domain']}

train_ns=pure_namespace('training/train_url_and_domain_models.py',
    funcs=['extract_url_features','extract_domain_features','_entropy','_tld_risk',
           '_char_continuation_rate','_unique_char_ratio','_dga_score','_vowel_ratio',
           '_max_consonant_run','_homoglyph_flag'],
    constants=['TLD_RISK','KNOWN_SHORTENERS','BRAND_KEYWORDS','OFFICIAL_DOMAINS',
               'SECURITY_KEYWORDS_RE','_BIGRAM_FREQ','KNOWN_TLD_TIERS'])
live_ns=pure_namespace('services/scan_engine.py',
    funcs=['extract_url_features','extract_domain_features','entropy','_tld_risk_score',
           '_char_continuation_rate','_unique_char_ratio','_dga_score','_vowel_ratio',
           '_max_consonant_run','_homoglyph_flag_domain','score_content'],
    constants=['_TLD_RISK','_KNOWN_SHORTENERS_RE','_BRAND_KEYWORDS_RE','_OFFICIAL_DOMAINS_RE',
               '_SECURITY_KEYWORDS_RE','_BIGRAM_FREQ','_KNOWN_TLD_TIERS','PHISHING_KEYWORDS'])
pillar_ns=pure_namespace('services/origintrace_pillars.py',classes=['SemanticDeception','EvidentialFusion'])
Semantic=pillar_ns['SemanticDeception']; Fusion=pillar_ns['EvidentialFusion']

with threadpool_limits(limits=1):
    # Dataset identity and label/source structure.
    audit={}
    for kind,frame in frames.items():
        group='campaign_cluster_id' if kind=='email' else 'psl_domain'
        sets={sp:set(frame.loc[frame.split==sp,group]) for sp in ['train','val','test']}
        entry={'n':len(frame),'splits':frame.split.value_counts().to_dict(),
               'group_overlap':{a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
               'duplicate_group_rows':int(frame[group].duplicated().sum())}
        if 'source' in frame:
            entry['source_label_counts']={f'{s}|{label}':int(n) for (s,label),n in frame.groupby(['source','label']).size().items()}
            # Source-only baseline uses source metadata, not input text; proves potential confounding only.
            tr=frame[frame.split=='train']; te=frame[frame.split=='test']
            priors=tr.groupby('source').label.mean().to_dict()
            entry['source_only_test_baseline']=metrics(te.label,te.source.map(priors).fillna(tr.label.mean()))
        if kind=='email':
            entry['preprocessing_mismatches']=sum(clean_and_mask_email_text(r)!=m for r,m in zip(frame.raw_text,frame.masked_text))
            entry['length_exclusions_are_not_verified_corruption']=True
            entry['masked_exact_duplicates']=int(frame.masked_text.duplicated().sum())
        audit[kind]=entry
    url_test=set(frames['url'].loc[frames['url'].split=='test','psl_domain'])
    dom_train=set(frames['domain'].loc[frames['domain'].split=='train','psl_domain'])
    url_train=set(frames['url'].loc[frames['url'].split=='train','psl_domain'])
    dom_test=set(frames['domain'].loc[frames['domain'].split=='test','psl_domain'])
    audit['cross_model_overlap']={'url_test_domains_in_domain_training':len(url_test&dom_train),
                                'domain_test_domains_in_url_training':len(dom_test&url_train)}
    # Independently check all pairs satisfying the implemented 64-bit SimHash rule,
    # including pairs skipped by earlier large-bucket heuristics. Store row IDs only.
    sim_ns=pure_namespace('scripts/clean_and_curate_master_datasets.py',funcs=['compute_simhash_fast'])
    fingerprints=[sim_ns['compute_simhash_fast'](t) for t in frames['email'].masked_text]
    by_hash=defaultdict(list)
    for idx,h in enumerate(fingerprints): by_hash[h].append(idx)
    bucket=[defaultdict(list) for _ in range(8)]
    for h in by_hash:
        if h:
            for chunk in range(8): bucket[chunk][(h>>(8*chunk))&255].append(h)
    pairs=set()
    for part in bucket:
        for hashes in part.values():
            array=np.array(hashes,dtype=np.uint64)
            for i,h in enumerate(hashes[:-1]):
                close=np.flatnonzero(np.bitwise_count(array[i+1:]^np.uint64(h))<=6)
                for j in close:
                    other=hashes[i+1+int(j)];pairs.add(tuple(sorted((h,other))))
    cross_split=[]; assigned_cluster_misses=0; email_frame=frames['email']
    candidate_pairs=[(h,h) for h,ids in by_hash.items() if h and len(ids)>1]+list(pairs)
    for h1,h2 in candidate_pairs:
        for i in by_hash[h1]:
            for j in by_hash[h2]:
                if i>=j and h1==h2: continue
                if email_frame.iloc[i].campaign_cluster_id!=email_frame.iloc[j].campaign_cluster_id:
                    assigned_cluster_misses+=1
                if email_frame.iloc[i].split!=email_frame.iloc[j].split:
                    cross_split.append({'row_a':i,'row_b':j,'hamming_distance':(h1^h2).bit_count()})
    audit['email']['independent_simhash_audit']={'fingerprint_rule':'first 80 whitespace tokens longer than 2 characters; distance <= 6',
        'different_nonzero_fingerprint_pairs_within_distance':len(pairs),'assigned_cluster_missed_pairs':assigned_cluster_misses,
        'cross_split_near_duplicate_pairs':len(cross_split),'cross_split_pair_examples':cross_split[:20],
        'zero_fingerprint_rows':len(by_hash.get(0,[])),
        'scope':'Exhaustive for this SimHash criterion only; does not establish semantic campaign independence.'}
    report['results']['data']=audit
    checkpoint('Dataset counts, source/label confounding and preprocessing checked.')

    # Exact feature parity on all exported URLs, and inputs used by domain training vs serving.
    parity={}; Xs={}
    for kind,frame in [('url',frames['url']),('domain',frames['domain'])]:
        column='url' if kind=='url' else 'psl_domain'
        train_func=train_ns['extract_'+kind+'_features']; live_func=live_ns['extract_'+kind+'_features']
        values=frame[column].astype(str).tolist()
        expected=pd.DataFrame([train_func(s) for s in values]); actual=pd.DataFrame([live_func(s) for s in values])
        mismatches=(expected!=actual)
        parity[kind]={'same_column_order':list(expected.columns)==list(actual.columns),
                      'rows_with_mismatch':int(mismatches.any(axis=1).sum()),
                      'different_features':{k:int(v) for k,v in mismatches.sum().items() if v},
                      'constant_features':list(expected.columns[expected.nunique()==1])}
        Xs[kind]=expected
    domain_url=pd.DataFrame([train_ns['extract_domain_features'](s) for s in frames['url'].url])
    domain_root=pd.DataFrame([train_ns['extract_domain_features'](s) for s in frames['url'].psl_domain])
    parity['domain_input_scope']={'url_vs_registered_domain_different_rows':int((domain_url!=domain_root).any(axis=1).sum()),
                                'n_urls':len(domain_url),
                                'interpretation':'Model trained on registrable domain strings; live engine often passes complete URL host.'}
    report['results']['feature_parity']=parity
    checkpoint('Feature parity and domain input scope checked.')

    # Actual saved specialist results. No metrics copied from previous reports.
    models={k:pin(joblib.load(ROOT/f'models/{"phish" if k=="email" else k}_model.pkl')) for k in frames}
    vec=joblib.load(ROOT/'models/vectorizer.pkl')
    email=frames['email']
    Xs['email']=hstack([vec.transform(email.masked_text),csr_matrix([extract_features(s)[0] for s in email.raw_text])]).tocsr()
    all_probs={k:models[k].predict_proba(Xs[k])[:,1] for k in frames}
    measured={}
    for kind,frame in frames.items():
        tr=frame.split.values=='train'; va=frame.split.values=='val'; te=frame.split.values=='test'; y=frame.label.values
        threshold=threshold_on_validation(y[va],all_probs[kind][va])
        measured[kind]={'train_at_05':metrics(y[tr],all_probs[kind][tr]),
                        'validation_at_05':metrics(y[va],all_probs[kind][va]),
                        'test_at_05':metrics(y[te],all_probs[kind][te]),
                        'validation_selected_fpr_005':metrics(y[va],all_probs[kind][va],threshold),
                        'test_at_validation_selected_fpr_005':metrics(y[te],all_probs[kind][te],threshold),
                        'n_calibrated_estimators':len(models[kind].calibrated_classifiers_)}
        base_estimators=[c.estimator for c in models[kind].calibrated_classifiers_]
        measured[kind]['effective_bagging_freq']=[e.booster_.params.get('bagging_freq',e.booster_.params.get('subsample_freq',0)) for e in base_estimators]
        if 'source' in frame:
            measured[kind]['test_by_source']={s:metrics(y[te&(frame.source.values==s)],all_probs[kind][te&(frame.source.values==s)]) for s in sorted(frame.source.unique()) if (te&(frame.source.values==s)).any()}
        if kind=='url':
            measured[kind]['test_by_depth_type']={s:metrics(y[te&(frame.depth_type.values==s)],all_probs[kind][te&(frame.depth_type.values==s)]) for s in sorted(frame.depth_type.unique()) if (te&(frame.depth_type.values==s)).any()}
        raw=np.mean([c.estimator.predict_proba(Xs[kind])[:,1] for c in models[kind].calibrated_classifiers_],axis=0)
        raw_tau=threshold_on_validation(y[va],raw[va])
        measured[kind]['raw_base_ensemble_test_at_05']=metrics(y[te],raw[te])
        measured[kind]['raw_base_ensemble_test_at_validation_fpr_005']=metrics(y[te],raw[te],raw_tau)
        if kind=='email':
            measured[kind]['calibrated_test_threshold_curve']={str(t):metrics(y[te],all_probs[kind][te],t) for t in [.5,.8,.9,.95,.98,.99]}
        np.savez_compressed(OUT/f'{kind}_predictions.npz',row=np.arange(len(frame)),label=y,probability=all_probs[kind],raw_probability=raw,split=frame.split.values.astype(str))
    report['results']['saved_models']=measured
    checkpoint('Recomputed saved-model metrics, calibration and matched-FPR thresholds.')

    # Two-by-two domain experiment separates representation from model family.
    dom=frames['domain']; tr=dom.split.values=='train'; va=dom.split.values=='val'; te=dom.split.values=='test'; y=dom.label.values
    char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=10000,min_df=2,sublinear_tf=True)
    chars=char.fit_transform(dom.loc[tr,'psl_domain']); chars_val=char.transform(dom.loc[va,'psl_domain']); chars_test=char.transform(dom.loc[te,'psl_domain'])
    features=Xs['domain'].values
    scaler=StandardScaler().fit(features[tr]); struct_train=scaler.transform(features[tr]); struct_val=scaler.transform(features[va]); struct_test=scaler.transform(features[te])
    experiments={}
    for representation,xtr,xval,xte in [('structural',struct_train,struct_val,struct_test),('character_ngrams',chars,chars_val,chars_test)]:
        for family in ['logistic_regression','lightgbm']:
            t0=time.perf_counter()
            candidate=(LogisticRegression(C=1.,max_iter=500,solver='liblinear',random_state=42) if family=='logistic_regression' else lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=31,reg_alpha=.5,reg_lambda=1.,random_state=42,n_jobs=1,verbosity=-1))
            candidate.fit(xtr,y[tr]); pv=candidate.predict_proba(xval)[:,1]; pt=candidate.predict_proba(xte)[:,1]
            tau=threshold_on_validation(y[va],pv)
            experiments[representation+'__'+family]={'fit_seconds':time.perf_counter()-t0,'test_at_05':metrics(y[te],pt),'test_at_validation_fpr_005':metrics(y[te],pt,tau)}
            checkpoint('Domain experiment completed: '+representation+' / '+family)
    report['results']['domain_representation_x_model']=experiments
    report['results']['domain_feature_collisions']={'different_strings_with_identical_features':len(dom)-len(Xs['domain'].drop_duplicates())}
    checkpoint('Domain two-by-two representation/model comparison complete.')

    # Model-to-engine ablation with missing modalities neutral. No remote scans/DB/RAG/vision/BEC.
    # Inject the already computed probabilities to avoid repeated per-row ensemble inference.
    live_ns.update(MODEL_LOADED=True,TFIDF_VECTORIZER=types.SimpleNamespace(transform=lambda rows:csr_matrix((1,0))),
                   clean_and_mask_email_text=clean_and_mask_email_text,extract_features_for_inference=extract_features,
                   logger=types.SimpleNamespace(debug=lambda *a:None,error=lambda *a:None,warning=lambda *a:None))
    live_ns['ML_MODEL']=types.SimpleNamespace(predict_proba=lambda x:np.array([[1-live_ns['_audit_p'],live_ns['_audit_p']]]))
    content_scores=[]; semantic_scores=[]; fused_scores=[]; urgency_flags=[]
    neutral={'confidence':0.,'threat_prob':0.}
    for text,p in zip(email.raw_text,all_probs['email']):
        live_ns['_audit_p']=float(p)
        c=live_ns['score_content'](text)
        s=Semantic.analyze({'is_bec':False,'bec_score':0},c,{'has_match':False},{'has_quishing':False})
        f=Fusion.synthesize(neutral,neutral,s)
        content_scores.append(c['score']/100);semantic_scores.append(s['threat_prob']);fused_scores.append(f['final_score']/100);urgency_flags.append(c.get('has_urgency_language',False))
    va=email.split.values=='val';te=email.split.values=='test';y=email.label.values;urgency=np.array(urgency_flags)
    stages={}
    for name,p in [('specialist_probability',all_probs['email']),('content_score',content_scores),('semantic_without_other_detectors',semantic_scores),('fusion_missing_other_modalities',fused_scores)]:
        p=np.array(p); tau=threshold_on_validation(y[va],p[va])
        stages[name]={'test_at_05':metrics(y[te],p[te]),'test_at_075':metrics(y[te],p[te],.75),'test_at_validation_fpr_005':metrics(y[te],p[te],tau)}
    strong=(all_probs['email']>=.9)&te&(y==1)
    report['results']['email_engine_ablation']={'scope':'Controlled missing-modality ablation. Other detectors disabled to isolate ML-to-content-to-semantic-to-fusion path; NOT full online API accuracy.',
        'stages':stages,'high_confidence_phishing_test_without_urgency_flag':int((strong&~urgency).sum()),
        'high_confidence_phishing_test_total':int(strong.sum()),
        'semantic_probe_probability_099_no_urgency':Semantic.analyze({'is_bec':False,'bec_score':0},{'score':79,'ml_probability':.99,'has_urgency_language':False,'is_english':True},{'has_match':False},{'has_quishing':False})}
    checkpoint('Measured ML-to-engine stage ablation with neutral missing evidence.')

    # Latency includes transform + batch-one inference, warm CPU only (no remote services).
    timings={}
    for kind in frames:
        frame=frames[kind]; samples=frame[frame.split=='test'].head(10); ts=[]
        for rep in range(4):
            for _,row in samples.iterrows():
                t0=time.perf_counter()
                if kind=='email': x=hstack([vec.transform([clean_and_mask_email_text(row.raw_text)]),csr_matrix(extract_features(row.raw_text))]).tocsr()
                else: x=pd.DataFrame([train_ns['extract_'+kind+'_features'](row.url if kind=='url' else row.psl_domain)])
                models[kind].predict_proba(x)
                if rep: ts.append((time.perf_counter()-t0)*1000)
        timings[kind]={'n':len(ts),'median_ms':float(np.median(ts)),'p95_ms':float(np.percentile(ts,95)),
                       'scope':'30 warm batch-one CPU calls, 10 inputs; descriptive latency only, not scalability or production SLA'}
    report['results']['warm_cpu_latency']=timings
    checkpoint('Warm CPU latency measured; no network or application database calls.')

report['elapsed_seconds']=time.time()-started
report['inputs_changed_during_audit']=[p for p,h in report['input_sha256'].items() if sha(ROOT/p)!=h]
report['audit_completed']=True
save()
print(json.dumps({'completed':True,'elapsed_seconds':report['elapsed_seconds'],'changed_inputs':report['inputs_changed_during_audit'],'output':str(OUT/'results.json')}),flush=True)
