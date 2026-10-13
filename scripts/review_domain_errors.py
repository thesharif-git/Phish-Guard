"""Offline domain error/provenance review. Never changes labels or live models."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.metrics import roc_auc_score, brier_score_loss
from training.train_url_and_domain_models import extract_domain_features
from services.security_pipeline import numeric_url_frame
from scripts.clean_and_curate_master_datasets import canonicalize_url_clean
from scripts.diagnose_domain_representation import threshold, metrics

OUT = ROOT / 'benchmarks/domain_error_review'
DESKTOP = Path.home() / 'Desktop'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def reconstruct():
    # Match the curator's selection, order, conflicts, and duplicate handling.
    # Use bundled PSL only; make no requests to live domains.
    tldextract.extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
    top = pd.read_csv(DESKTOP / 'top-1m.csv', header=None, nrows=50000)
    rows = [('https://www.' + str(d), 0, 'Tranco_Top50k') for d in top[1].iloc[1000:16000]]
    mp = pd.read_csv(DESKTOP / 'malicious_phish.csv', nrows=250000).dropna(subset=['url'])
    mp.url = mp.url.astype(str).str.strip()
    rows += [(u, 0, 'MaliciousPhish_Deep') for u in mp[mp.type == 'benign'].url.drop_duplicates().iloc[:15000]]
    phi = pd.read_csv(DESKTOP / 'PhiUSIIL_Phishing_URL_Dataset.csv', nrows=50000,
                      usecols=['URL', 'label']).dropna(subset=['URL'])
    rows += [(u, 0, 'PhiUSIIL') for u in phi[phi.label == 1].URL.drop_duplicates().iloc[:10000]]
    phishing = mp[mp.type == 'phishing'].url.drop_duplicates().tolist()
    rows += [(u, 1, 'MaliciousPhish_Root') for u in [u for u in phishing if '/' not in u.split('://')[-1]][:15000]]
    rows += [(u, 1, 'MaliciousPhish_Deep') for u in [u for u in phishing if '/' in u.split('://')[-1]][:15000]]
    rows += [(u, 1, 'PhiUSIIL') for u in phi[phi.label == 0].URL.drop_duplicates().iloc[:10000]]
    result = []
    for u, label, source in rows:
        clean, _, domain, path = canonicalize_url_clean(u)
        if domain != 'unknown' and len(domain) >= 3 and 4 <= len(clean) <= 2048:
            result.append(dict(url=u, label=label, source=source, clean_url=clean,
                               psl_domain=domain, path_query=path))
    pool = pd.DataFrame(result)
    conflicts = pool.groupby('clean_url').label.nunique()
    pool = pool[~pool.clean_url.isin(conflicts[conflicts > 1].index)].drop_duplicates('clean_url')
    return pool

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT/'data/clean_domain_corpus.csv', ROOT/'models/domain_model.pkl',
              DESKTOP/'top-1m.csv', DESKTOP/'malicious_phish.csv',
              DESKTOP/'PhiUSIIL_Phishing_URL_Dataset.csv', DESKTOP/'3_malicious_domains.csv',
              ROOT/'scripts/clean_and_curate_master_datasets.py']
    before = {str(p): sha(p) for p in inputs}
    df = pd.read_csv(inputs[0])
    model = joblib.load(inputs[1])
    for c in model.calibrated_classifiers_:
        c.estimator.set_params(n_jobs=1)
    feats = numeric_url_frame(model, [extract_domain_features(d) for d in df.psl_domain])
    df['probability'] = model.predict_proba(feats)[:, 1]
    val = df[df.split == 'val']
    tau = threshold(val.label, val.probability)
    print('Reconstructing domain label provenance from original local source files', flush=True)
    pool = reconstruct()
    grouped = {d: g for d, g in pool.groupby('psl_domain')}
    standalone = pd.read_csv(DESKTOP/'3_malicious_domains.csv').dropna()
    standalone['psl_domain'] = [tldextract.extract(str(d)).top_domain_under_public_suffix
                                for d in standalone.iloc[:, 0]]
    standalone_groups = {d: g for d, g in standalone.groupby('psl_domain')}
    provenance = {}
    for row in df.itertuples():
        g = grouped.get(row.psl_domain)
        if g is not None:
            labels = sorted(map(int, g.label.unique()))
            origin = 'URL-label aggregation'
            match = labels == [row.label] and len(g) == row.count
            urls = g[['url', 'source', 'label', 'path_query']].head(3).to_dict('records')
            sources = sorted(g.source.unique())
        else:
            labels, urls, sources = [], [], []
            origin = 'Standalone unconditional positive' if row.psl_domain in standalone_groups else 'Unrecovered'
            match = origin == 'Standalone unconditional positive' and row.label == 1 and row.count == 1
        provenance[row.psl_domain] = dict(origin=origin, reconstruction_matches=bool(match),
                                         sources=sources, source_labels=labels, example_urls=urls)
    df['origin'] = df.psl_domain.map(lambda d: provenance[d]['origin'])
    df['sources'] = df.psl_domain.map(lambda d: '+'.join(provenance[d]['sources']))
    df['reconstruction_matches'] = df.psl_domain.map(lambda d: provenance[d]['reconstruction_matches'])
    df['decision_strict'] = df.probability >= tau
    test = df[df.split == 'test']
    split_scores = {s: metrics(g.label.to_numpy(), g.probability.to_numpy(), tau)
                    for s, g in df.groupby('split')}
    # Exact numeric-feature collisions quantify observable ambiguity, not Bayes error.
    collision = feats.copy()
    collision['label'] = df.label.to_numpy()
    groups = collision.groupby(list(feats.columns), dropna=False).label.agg(['size', 'sum'])
    mixed = groups[(groups['sum'] > 0) & (groups['sum'] < groups['size'])]
    unavoidable_training_errors = int(np.minimum(groups['sum'], groups['size']-groups['sum']).sum())
    strata = []
    for keys, g in test.groupby(['origin', 'sources', 'label']):
        strata.append(dict(origin=keys[0], sources=keys[1], label=int(keys[2]), n=len(g),
                           flagged=int(g.decision_strict.sum()),
                           mean_probability=float(g.probability.mean())))
    samples = []
    for label, predicted, name in [(1, False, 'false_negative'), (0, True, 'false_positive'),
                                    (1, True, 'true_positive'), (0, False, 'true_negative')]:
        group = test[(test.label == label) & (test.decision_strict == predicted)]
        chosen = group.sample(n=min(12,len(group)), random_state=42)
        for row in chosen.itertuples():
            samples.append(dict(domain=row.psl_domain, label=int(label), decision_group=name,
                                probability=row.probability, **provenance[row.psl_domain],
                                independently_verified=False,
                                review_status='needs time-stamped URL/ownership evidence; name alone cannot verify label'))
    extra = []
    for d, g in standalone_groups.items():
        if d in provenance:
            extra.append(dict(domain=d, corpus_label=int(df[df.psl_domain==d].label.iloc[0]),
                              origin=provenance[d]['origin'], records=g.drop(columns='psl_domain').to_dict('records')))
    result = dict(complete=True, input_hashes=before, no_network_requests=True,
                  development_only=True, production_unchanged=True, threshold=tau,
                  split_metrics=split_scores, domain_origins=dict(Counter(df.origin)),
                  reconstructed_matching_rows=int(df.reconstruction_matches.sum()),
                  unreconciled_rows=int((~df.reconstruction_matches).sum()), test_strata=strata,
                  numeric_feature_collision=dict(mixed_groups=len(mixed),
                       rows_in_mixed_groups=int(mixed['size'].sum()),
                       minimum_errors_for_any_deterministic_classifier_on_these_features=unavoidable_training_errors,
                       scope='all existing rows; descriptive representation ambiguity, not unseen error bound'),
                  standalone_source_corpus_overlap=extra)
    if before != {str(p):sha(p) for p in inputs}:
        raise RuntimeError('Inputs changed during audit')
    df.to_csv(OUT/'domain_predictions_and_provenance.csv',index=False)
    (OUT/'review_samples.json').write_text(json.dumps(samples, indent=2),encoding='utf-8')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','standalone_source_corpus_overlap','test_strata')},indent=2),flush=True)

if __name__ == '__main__':
    main()
