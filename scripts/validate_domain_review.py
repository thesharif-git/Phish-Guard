"""Recompute the domain review's arithmetic, sampling, and live feature parity."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Dict, Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from training.train_url_and_domain_models import extract_domain_features
from services.security_pipeline import domain_model_input, numeric_url_frame
from scripts.diagnose_domain_representation import metrics, threshold

OUT = ROOT/'benchmarks/domain_error_review'

def main():
    r = json.loads((OUT/'results.json').read_text())
    for path, expected in r['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    original = pd.read_csv(ROOT/'data/clean_domain_corpus.csv')
    saved = pd.read_csv(OUT/'domain_predictions_and_provenance.csv')
    pd.testing.assert_frame_equal(original, saved[original.columns])
    assert original.psl_domain.is_unique
    assert len(saved) == 32000 and saved.reconstruction_matches.all()
    # Read the actual production function without importing DB/network services.
    funcs = {'extract_domain_features','entropy','_tld_risk_score','_unique_char_ratio',
             '_dga_score','_vowel_ratio','_max_consonant_run','_homoglyph_flag_domain'}
    constants = {'_TLD_RISK','_BRAND_KEYWORDS_RE','_OFFICIAL_DOMAINS_RE',
                 '_SECURITY_KEYWORDS_RE','_BIGRAM_FREQ','_KNOWN_TLD_TIERS'}
    source = ROOT/'services/scan_engine.py'
    nodes = []
    for n in ast.parse(source.read_text(encoding='utf-8-sig')).body:
        if isinstance(n,ast.FunctionDef) and n.name in funcs:
            nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets):
            nodes.append(n)
    ns = dict(re=re,math=math,Counter=Counter,urlparse=urlparse,
              domain_model_input=domain_model_input,Dict=Dict,Any=Any)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    train_feats = [extract_domain_features(d) for d in original.psl_domain]
    live_feats = [ns['extract_domain_features'](d) for d in original.psl_domain]
    assert train_feats == live_feats, 'Training/live feature mismatch'
    model = joblib.load(ROOT/'models/domain_model.pkl')
    for c in model.calibrated_classifiers_:
        c.estimator.set_params(n_jobs=1)
    probabilities = model.predict_proba(numeric_url_frame(model, live_feats))[:,1]
    assert np.allclose(probabilities,saved.probability,rtol=0,atol=1e-12)
    val = saved[saved.split=='val']
    tau = threshold(val.label,val.probability)
    assert tau == r['threshold']
    assert np.array_equal(saved.decision_strict, probabilities>=tau)
    for split,g in saved.groupby('split'):
        actual = metrics(g.label.to_numpy(),g.probability.to_numpy(),tau)
        for key,value in actual.items():
            assert abs(value-r['split_metrics'][split][key])<1e-12, (split,key)
    assert dict(Counter(saved.origin)) == r['domain_origins']
    frames = pd.DataFrame(train_feats)
    cols = frames.columns.tolist()
    frames['label'] = saved.label
    groups = frames.groupby(cols,dropna=False).label.agg(['size','sum'])
    mixed = groups[(groups['sum']>0)&(groups['sum']<groups['size'])]
    c = r['numeric_feature_collision']
    assert len(mixed)==c['mixed_groups']
    assert int(mixed['size'].sum())==c['rows_in_mixed_groups']
    assert int(np.minimum(groups['sum'],groups['size']-groups['sum']).sum())==c['minimum_errors_for_any_deterministic_classifier_on_these_features']
    samples = json.loads((OUT/'review_samples.json').read_text())
    observed = {s['domain'] for s in samples}
    expected = set()
    test = saved[saved.split=='test']
    for y,decision in ((1,False),(0,True),(1,True),(0,False)):
        g=test[(test.label==y)&(test.decision_strict==decision)]
        expected.update(g.sample(n=min(12,len(g)),random_state=42).psl_domain)
    assert expected==observed and len(observed)==48
    assert all(s['independently_verified'] is False for s in samples)
    result = dict(complete=True,checks_passed=[
        'All recorded input checksums unchanged', 'Original labels/order reproduced exactly',
        'Unique registrable-domain identities across splits',
        'All 32,000 training and live feature vectors match',
        'All 32,000 live-model probabilities reproduce saved values',
        'Validation-only threshold and every split metric reproduced',
        'Origin and feature-collision counts reproduced',
        '48 review examples reproduce seeded selection without manual cherry-picking'],
        limitations=['Underlying labels not independently verified',
                     'Balanced legacy sources and reused development test are not deployment-representative',
                     'Equal-count error/control sampling cannot estimate population label error',
                     'Path/source associations do not establish causal contributions'])
    (OUT/'evidence_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
