"""Read-only production audit. Writes only a new benchmark evidence directory.

No fitting, calibration, network scans, live policy claims or automatic relabels.
Sampling is seeded within label/decision strata, not selected for a narrative.
"""
from pathlib import Path
import sys, json, hashlib, subprocess, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from services import scan_engine as engine
from services.security_pipeline import security_pipeline_id,numeric_url_frame
from services.conformal_evidential_engine import ConformalPredictor
from utils.text_sanitizer import clean_and_mask_email_text
from training.train_url_and_domain_models import extract_url_features,extract_domain_features
from scripts.diagnose_domain_representation import threshold

OUT=ROOT/'benchmarks/frozen_failure_baseline_20261003'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def files():
    return sorted(set([p for folder in ('services','utils','training','routers','core','tests','scripts') for p in (ROOT/folder).rglob('*.py')]+[ROOT/'data'/f'clean_{m}_corpus.csv' for m in ('email','url','domain')]+list((ROOT/'models').glob('*.pkl'))+[ROOT/'models/conformal_params.json',ROOT/'models/conformal_evaluation_results.json']))
def snapshots(): return {p.relative_to(ROOT).as_posix():digest(p) for p in files()}
def pin(model):
    for c in getattr(model,'calibrated_classifiers_',[]):
        if hasattr(c.estimator,'set_params'): c.estimator.set_params(n_jobs=1)
def sample(frame,modality,columns):
    selected=[]
    for (label,decision),part in frame.groupby(['label','decision'],sort=True):
        take=part.sample(n=min(8,len(part)),random_state=42).sort_index()
        for index,row in take.iterrows():
            rec={k:(None if pd.isna(row[k]) else row[k]) for k in columns}
            rec.update(modality=modality,corpus_row_index=int(row['_corpus_row_index']),prediction_row_index=int(index),label=int(label),decision=str(decision),probability=float(row.probability),label_review_status='UNVERIFIED',failure_cause='NOT_ADJUDICATED')
            selected.append(rec)
    return selected
def counts(frame):
    return [{'label':int(label),'decision':str(action),'n':int(len(part))} for (label,action),part in frame.groupby(['label','decision'])]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=snapshots(); identity=security_pipeline_id()
    (OUT/'manifest.json').write_text(json.dumps({'complete':False,'pipeline_id':identity,'files':frozen,'scope':'offline development diagnosis; full live policy not evaluated'},indent=2))
    diff=subprocess.run(['git','diff','--stat'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'tracked_change_inventory.txt').write_text(diff.stdout+diff.stderr,encoding='utf-8')
    engine.load_ml_model()
    for model in (engine.ML_MODEL,engine.URL_MODEL,engine.DOMAIN_MODEL): pin(model)
    predictor=ConformalPredictor(parameter_path=ROOT/'models/conformal_params.json')
    assert predictor.pipeline_id==identity and predictor.is_calibrated
    email=pd.read_csv(ROOT/'data/clean_email_corpus.csv')
    et=email[email.split=='test'].copy()
    et['_corpus_row_index']=et.index
    started=time.perf_counter()
    print('Scoring current shared email probability branch (no live sensors)',flush=True)
    et['probability']=engine.predict_email_probabilities([str(t)+' ' for t in et.raw_text])
    et['decision']=[predictor.predict_set(float(p),pipeline_id=identity)[2] for p in et.probability]
    et['current_masked']=et.raw_text.map(clean_and_mask_email_text)
    mismatch=int((email.masked_text.fillna('')!=email.raw_text.map(clean_and_mask_email_text)).sum())
    vocabulary=engine.TFIDF_VECTORIZER.vocabulary_
    analyzer=engine.TFIDF_VECTORIZER.build_analyzer()
    def oov(text):
        tokens=analyzer(text)
        return sum(t not in vocabulary for t in tokens)/len(tokens) if tokens else None
    et['oov_fraction']=et.current_masked.map(oov)
    elapsed=time.perf_counter()-started
    # Existing predictions are usable only when dataset/model checksums match.
    comparison=json.loads((ROOT/'benchmarks/domain_scan_correction/model_comparison.json').read_text())
    for filename,sha in comparison['input_hashes'].items(): assert digest(Path(filename))==sha
    urls=pd.read_csv(ROOT/'benchmarks/domain_scan_correction/paired_predictions.csv')
    url_corpus=pd.read_csv(ROOT/'data/clean_url_corpus.csv')
    assert not url_corpus.url.duplicated().any()
    url_indices=dict(zip(url_corpus.url,url_corpus.index))
    urls['_corpus_row_index']=urls.url.map(url_indices)
    assert urls['_corpus_row_index'].notna().all()
    uv=urls[urls.split=='val'];ut=urls[urls.split=='test'].copy()
    tau=threshold(uv.label,uv.url_probability)
    ut['probability']=ut.url_probability
    ut['decision']=np.where(ut.probability>=tau,'FLAGGED_MODEL_STAGE','NOT_FLAGGED_MODEL_STAGE')
    provenance=pd.read_csv(ROOT/'benchmarks/domain_error_review/domain_predictions_and_provenance.csv')
    dh=json.loads((ROOT/'benchmarks/domain_error_review/results.json').read_text())
    for filename in ('data/clean_domain_corpus.csv','models/domain_model.pkl'):
        path=ROOT/filename;assert digest(path)==dh['input_hashes'][str(path)]
    dt=provenance[provenance.split=='test'].copy()
    domain_corpus=pd.read_csv(ROOT/'data/clean_domain_corpus.csv')
    assert not domain_corpus.psl_domain.duplicated().any()
    domain_indices=dict(zip(domain_corpus.psl_domain,domain_corpus.index))
    dt['_corpus_row_index']=dt.psl_domain.map(domain_indices)
    assert dt['_corpus_row_index'].notna().all()
    dt['decision']=np.where(dt.probability>=dh['threshold'],'FLAGGED_LEGACY_DIAGNOSTIC','NOT_FLAGGED_LEGACY_DIAGNOSTIC')
    reviews=sample(et,'email',['raw_text','current_masked','source','campaign_cluster_id','oov_fraction'])+sample(ut,'url',['url','psl_domain','source','scheme','path_query'])+sample(dt,'domain',['psl_domain','origin','sources'])
    # Replay selected cached probabilities with actual current feature/schema functions.
    for kind,part,model,extract,col in [('url',ut,engine.URL_MODEL,extract_url_features,'url'),('domain',dt,engine.DOMAIN_MODEL,extract_domain_features,'psl_domain')]:
        indices=[r['prediction_row_index'] for r in reviews if r['modality']==kind]
        subset=part.loc[indices]
        actual=model.predict_proba(numeric_url_frame(model,[extract(v) for v in subset[col]]))[:,1]
        assert np.allclose(actual,subset.probability,atol=1e-12,rtol=0),f'{kind} cached probabilities stale'
    (OUT/'review_queue.json').write_text(json.dumps(reviews,indent=2,ensure_ascii=False),encoding='utf-8')
    et.drop(columns=['raw_text','masked_text','current_masked']).to_csv(OUT/'email_decisions.csv',index=True,index_label='corpus_row_index')
    source_counts=[]
    for source,part in et.groupby('source'):
        source_counts.append({'source':source,'n':len(part),'decisions':counts(part),'auc':float(roc_auc_score(part.label,part.probability)) if part.label.nunique()==2 else None})
    masked= email.raw_text.map(clean_and_mask_email_text)
    duplicates=[]
    for s in ('val','test'):
        train=set(masked[email.split=='train']);target=masked[email.split==s]
        duplicates.append({'split':s,'exact_masked_overlap_with_training_rows':int(target.isin(train).sum())})
    result={'complete':True,'pipeline_id':identity,'email':{'auc':float(roc_auc_score(et.label,et.probability)),'brier':float(brier_score_loss(et.label,et.probability)),'decisions':counts(et),'by_source':source_counts,'stored_vs_current_mask_mismatch_rows':mismatch,'masked_overlap':duplicates,'batch_scoring_and_diagnostics_seconds':elapsed},'url':{'operating_point':'validation selected <=1% empirical FPR, model stage only','threshold':tau,'decisions':counts(ut)},'domain':{'operating_point':'legacy classifier diagnostic only; no production blocking claim','threshold':dh['threshold'],'decisions':counts(dt)},'review_sampling':'up to 8 per label/decision cell, seed 42; disproportionate sampling, not prevalence estimate','review_records':len(reviews),'cached_predictions_replayed':True,'limitations':['All labels unverified until adjudication','Test splits previously inspected during development','No full live policy benchmark','No new training, calibration or network checks']}
    assert snapshots()==frozen,'Inputs changed during audit'
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({'complete':True,'pipeline_id':identity,'files':frozen,'unchanged_at_end':True,'output_sha256':{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}},indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
