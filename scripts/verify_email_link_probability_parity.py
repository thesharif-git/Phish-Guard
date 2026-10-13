"""Verify unchanged statistical scorer; rebind existing parameters without fitting.

Transport/link policy changes do not acquire conformal guarantees from this check.
No quantiles, probabilities, trained artifacts or datasets are refitted.
"""
from pathlib import Path
import sys,ast,json,hashlib,warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from services import scan_engine as engine
from services.security_pipeline import security_pipeline_id
from services.conformal_evidential_engine import ConformalPredictor
from scripts.calibrate_conformal_engine import evaluation_metrics
OUT=ROOT/'benchmarks/email_link_preservation'

def function_node(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    before=(OUT/'scan_engine.before.py').read_text(encoding='utf-8')
    current=(ROOT/'services/scan_engine.py').read_text(encoding='utf-8')
    for name in ('predict_email_probabilities','score_content'):
        assert ast.dump(function_node(before,name))==ast.dump(function_node(current,name)),name+' changed'
    original_manifest=json.loads((ROOT/'benchmarks/frozen_failure_baseline_20261003/manifest.json').read_text(encoding='utf-8'))
    preserved={n:h for n,h in original_manifest['files'].items() if n.startswith('data/clean_') or (n.startswith('models/') and n.endswith('.pkl'))}
    assert all(digest(ROOT/n)==h for n,h in preserved.items())
    old_params=json.loads((OUT/'conformal_params.before.json').read_text(encoding='utf-8'))
    old_eval=json.loads((OUT/'conformal_evaluation_results.before.json').read_text(encoding='utf-8'))
    engine.load_ml_model()
    for model in (engine.ML_MODEL,engine.URL_MODEL,engine.DOMAIN_MODEL):
        for c in getattr(model,'calibrated_classifiers_',[]):c.estimator.set_params(n_jobs=1)
    original_scope=dict(vars(engine))
    exec(compile(ast.Module(body=[function_node(before,'predict_email_probabilities')],type_ignores=[]),'before_link_fix','exec'),original_scope)
    df=pd.read_csv(ROOT/'data/clean_email_corpus.csv')
    original_predictor=ConformalPredictor(parameter_path=OUT/'conformal_params.before.json')
    test_predictions=[];test_sets=[];test_actions=[];checked=0
    identity=security_pipeline_id()
    for split in ('val','test'):
        part=df[df.split==split]
        for offset in range(0,len(part),256):
            batch=part.iloc[offset:offset+256];texts=[str(t)+' ' for t in batch.raw_text]
            old=original_scope['predict_email_probabilities'](texts)
            new=engine.predict_email_probabilities(texts)
            assert np.array_equal(old,new),'Statistical probabilities changed'
            checked+=len(batch)
            if split=='test':
                for p in new:
                    ps,_,action=original_predictor.predict_set(float(p),pipeline_id=old_params['pipeline_id'])
                    test_predictions.append(float(p));test_sets.append(ps);test_actions.append(action)
        print(f'Verified before/after probability parity: {split}, {len(part)} rows',flush=True)
    test=df[df.split=='test']
    recomputed=evaluation_metrics(test.label,np.array(test_predictions),test_sets,np.array(test_actions))
    assert recomputed==old_eval['test_metrics'],'Saved evaluation not reproduced'
    baseline=pd.read_csv(ROOT/'benchmarks/frozen_failure_baseline_20261003/email_decisions.csv')
    assert np.allclose(test_predictions,baseline.probability,rtol=0,atol=1e-12)
    assert identity==security_pipeline_id()
    assert all(digest(ROOT/n)==h for n,h in preserved.items())
    # Preserve existing order statistics exactly; this is version rebinding, not calibration.
    new_params=dict(old_params,pipeline_id=identity)
    new_eval=dict(old_eval,pipeline_id=identity)
    new_eval['identity_rebinding']='Verified unchanged probability function and exact before/after validation/test probability parity; link review policies remain outside statistical evaluation'
    proof={'complete':True,'pipeline_id':identity,'before_pipeline_id':old_params['pipeline_id'],'rows_compared':checked,'before_after_probabilities_exactly_equal':True,'saved_test_metrics_reproduced':True,'quantiles_refitted':False,'trained_models_and_datasets_unchanged':True,'scope':'Statistical branch only; recovered-link operational decisions are separately regression-tested','preserved_hashes':preserved}
    (OUT/'probability_parity.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    for name,value in [('conformal_params.json',new_params),('conformal_evaluation_results.json',new_eval)]:
        target=ROOT/'models'/name;pending=target.with_suffix('.link_rebind_pending')
        pending.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');pending.replace(target)
    print(json.dumps({k:v for k,v in proof.items() if k!='preserved_hashes'},indent=2),flush=True)
if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    main()
