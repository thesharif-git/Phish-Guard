"""Measure the actual HTTPS homepage input used for bare-host domain scans."""
from pathlib import Path
import json
import sys
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import pandas as pd
from training.train_url_and_domain_models import extract_url_features
from services.domain_scan_policy import domain_scan_input
from services.security_pipeline import numeric_url_frame
from scripts.diagnose_domain_representation import metrics,threshold

def main():
    out=ROOT/'benchmarks/domain_scan_correction'
    df=pd.read_csv(out/'paired_predictions.csv')
    df=df[df.path_query.fillna('').eq('')].copy()
    def homepage(raw):
        try:
            host=urlsplit(raw if '://' in raw else 'https://'+raw).hostname or ''
            return domain_scan_input(host)[1]
        except (ValueError,UnicodeError):return None
    df['submitted_homepage']=df.url.map(homepage)
    excluded=int(df.submitted_homepage.isna().sum());df=df.dropna(subset=['submitted_homepage'])
    model=joblib.load(ROOT/'models/url_model.pkl')
    for c in model.calibrated_classifiers_:c.estimator.set_params(n_jobs=1)
    print('Scoring exact HTTPS homepage inputs',flush=True)
    df['https_probability']=model.predict_proba(numeric_url_frame(model,[extract_url_features(u) for u in df.submitted_homepage]))[:,1]
    val=df[df.split=='val'];test=df[df.split=='test']
    r=dict(scope='Same root-URL development rows, full hostname retained; bare-host API HTTPS synthesis; not ownership labels',
           excluded_invalid_input_rows=excluded,validation_rows=len(val),test_rows=len(test),
           comparisons={p:metrics(test.label.to_numpy(),test[p].to_numpy(),threshold(val.label,val[p]))
                        for p in ['url_probability','https_probability','domain_probability']},
           https_at_current_web_ml_cutoff=metrics(test.label.to_numpy(),test.https_probability.to_numpy(),.80))
    (out/'homepage_input_comparison.json').write_text(json.dumps(r,indent=2))
    df.to_csv(out/'homepage_predictions.csv',index=False)
    print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
