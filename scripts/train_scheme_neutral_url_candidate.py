"""Remove protocol-label shortcut without relabelling data or changing live models."""
from pathlib import Path
import hashlib
import json
import sys
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from training.train_url_and_domain_models import extract_url_features
from services.security_pipeline import numeric_url_frame
from services.domain_scan_policy import domain_scan_input
from scripts.diagnose_domain_representation import metrics,threshold

def features(urls):
    frame=pd.DataFrame([extract_url_features(u) for u in urls])
    frame['is_https']=0.0
    return frame.astype(float)

def https_homepage(raw):
    try:return domain_scan_input(urlsplit(raw if '://' in raw else 'https://'+raw).hostname or '')[1]
    except (ValueError,UnicodeError):return None

def main():
    path=ROOT/'data/clean_url_corpus.csv';data_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    df=pd.read_csv(path);parts={s:df[df.split==s].copy() for s in ('train','val','test')}
    tr=parts['train'];X=features(tr.url)
    folds=list(GroupKFold(n_splits=5).split(X,tr.label,groups=tr.psl_domain))
    base=lgb.LGBMClassifier(objective='binary',n_estimators=500,learning_rate=.03,
         num_leaves=31,subsample=.8,subsample_freq=1,colsample_bytree=.8,
         reg_alpha=.5,reg_lambda=1.,random_state=42,n_jobs=1,verbose=-1)
    model=CalibratedClassifierCV(estimator=base,method='sigmoid',cv=folds,n_jobs=1)
    print('Training grouped scheme-neutral numeric URL candidate',flush=True)
    model.fit(X,tr.label)
    for c in model.calibrated_classifiers_:
        names=c.estimator.feature_name_;importance=c.estimator.feature_importances_
        assert importance[names.index('is_https')]==0
    pval=model.predict_proba(features(parts['val'].url))[:,1]
    tau=threshold(parts['val'].label,pval)
    results={};root_predictions={}
    for split in ('val','test'):
        part=parts[split];p=model.predict_proba(features(part.url))[:,1]
        roots=part[part.path_query.fillna('').eq('')].copy()
        roots['homepage']=roots.url.map(https_homepage);roots=roots.dropna(subset=['homepage'])
        raw_features=pd.DataFrame([extract_url_features(u) for u in roots.homepage]).astype(float)
        hp=model.predict_proba(numeric_url_frame(model,raw_features.to_dict('records')))[:,1]
        swapped=raw_features.copy();swapped['is_https']=1-swapped.is_https
        assert np.allclose(hp,model.predict_proba(swapped)[:,1],rtol=0,atol=1e-12)
        results[split]=dict(all_urls=metrics(part.label.to_numpy(),p,tau),
                            https_homepages_at_current_cutoff=metrics(roots.label.to_numpy(),hp,.80))
        root_predictions[split]=(roots,hp)
    # Also report a separately validation-selected root operating point; do not deploy it here.
    rv,rp=root_predictions['val'];rt,tp=root_predictions['test']
    root_tau=threshold(rv.label,rp)
    results['test']['https_homepages_validation_budget']=metrics(rt.label.to_numpy(),tp,root_tau)
    assert data_hash==hashlib.sha256(path.read_bytes()).hexdigest()
    out=ROOT/'benchmarks/domain_scan_correction';out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/'scheme_neutral_url_candidate.pkl',compress=3)
    r=dict(complete=True,dataset_sha256=data_hash,production_models_changed=False,
           scheme_invariant_verified=True,scope='existing development splits; no ownership/fresh-holdout claim',results=results)
    (out/'scheme_neutral_candidate_results.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
