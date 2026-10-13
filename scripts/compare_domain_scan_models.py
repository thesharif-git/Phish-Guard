"""Paired URL-task comparison: full URL specialist vs registrable-domain model.

Existing development splits only; neither is a domain-ownership classifier.
No training or deployed artifact writes.
"""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import pandas as pd
from services.security_pipeline import domain_model_input,numeric_url_frame
from training.train_url_and_domain_models import extract_url_features,extract_domain_features
from scripts.diagnose_domain_representation import metrics,threshold

def main():
    files=[ROOT/'data/clean_url_corpus.csv',ROOT/'models/url_model.pkl',ROOT/'models/domain_model.pkl']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    df=pd.read_csv(files[0]);df=df[df.split.isin(['val','test'])].copy()
    models={name:joblib.load(p) for name,p in zip(['url','domain'],files[1:])}
    for name,model in models.items():
        for c in model.calibrated_classifiers_:c.estimator.set_params(n_jobs=1)
        print('Scoring '+name+' specialist on identical URL rows',flush=True)
        f=extract_url_features if name=='url' else lambda u:extract_domain_features(domain_model_input(u))
        df[name+'_probability']=model.predict_proba(numeric_url_frame(model,[f(u) for u in df.url]))[:,1]
    results={}
    for scope,mask in [('all_urls',df.index==df.index),('root_urls',df.path_query.fillna('').eq(''))]:
        part=df[mask];val=part[part.split=='val'];test=part[part.split=='test']
        results[scope]={name:metrics(test.label.to_numpy(),test[name+'_probability'].to_numpy(),
                              threshold(val.label,val[name+'_probability'])) for name in models}
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out=ROOT/'benchmarks/domain_scan_correction';out.mkdir(parents=True,exist_ok=True)
    r=dict(complete=True,input_hashes=hashes,scope='Existing URL task development split; model stages only, not final online policy or ownership labels',
           operating_point='Each model threshold selected separately on validation at <=1% empirical benign FPR',results=results)
    (out/'model_comparison.json').write_text(json.dumps(r,indent=2))
    df.to_csv(out/'paired_predictions.csv',index=False)
    print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
