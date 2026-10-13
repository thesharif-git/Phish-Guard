"""Development-only protocol-independent URL experiment; never deploys models."""
from pathlib import Path
import sys, json, hashlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from training.url_representation import URLSurfaceView
from scripts.train_scheme_neutral_url_candidate import https_homepage
from scripts.diagnose_domain_representation import metrics, threshold

def main():
    path = ROOT/'data/clean_url_corpus.csv'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    df = pd.read_csv(path)
    parts = {s: df[df.split == s].copy() for s in ('train', 'val', 'test')}
    tr = parts['train']
    fit, cal = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(tr, groups=tr.psl_domain))
    roots = {}
    for s in ('val','test'):
        r = parts[s][parts[s].path_query.fillna('').eq('')].copy()
        r['homepage'] = r.url.map(https_homepage)
        roots[s] = r.dropna(subset=['homepage'])
    candidates, models = [], []
    for c in (.1, 1., 10.):
        print(f'Training scheme-neutral text candidate C={c}', flush=True)
        pipe = Pipeline([('surface',URLSurfaceView()), ('text',TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=50000,sublinear_tf=True)), ('classifier',LogisticRegression(C=c,solver='liblinear',max_iter=500,random_state=42))])
        pipe.fit(tr.iloc[fit].url, tr.iloc[fit].label)
        model = CalibratedClassifierCV(FrozenEstimator(pipe), method='sigmoid')
        model.fit(tr.iloc[cal].url,tr.iloc[cal].label)
        rv = roots['val']; p = model.predict_proba(rv.homepage)[:,1]
        candidates.append(dict(C=c, validation_homepages=metrics(rv.label.to_numpy(),p,threshold(rv.label,p))))
        models.append(model)
    selected = max(range(len(models)),key=lambda i:(candidates[i]['validation_homepages']['recall'],candidates[i]['validation_homepages']['auc']))
    model = models[selected]; rt=roots['test']; p=model.predict_proba(rt.homepage)[:,1]
    swapped = rt.homepage.str.replace('https://','http://',regex=False)
    assert np.allclose(p,model.predict_proba(swapped)[:,1],rtol=0,atol=1e-12)
    fullp = model.predict_proba(parts['test'].url)[:,1]
    vp = model.predict_proba(parts['val'].url)[:,1]
    result = dict(complete=True, production_models_changed=False,dataset_sha256=before,scope='Existing development labels and splits; no fresh holdout or ownership validation', candidates=candidates, selected=selected,scheme_invariant_verified=True,test_homepages=metrics(rt.label.to_numpy(),p,candidates[selected]['validation_homepages']['threshold']),test_all_urls=metrics(parts['test'].label.to_numpy(),fullp,threshold(parts['val'].label,vp)))
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'benchmarks/domain_scan_correction/text_candidate_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
