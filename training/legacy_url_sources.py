"""Preserve legacy URL source selection while retaining evidence provenance.

These are URL assertions for development/review, never domain annotations.
"""
import hashlib
from pathlib import Path
import pandas as pd

def load_url_candidates(canonicalize, desktop=None):
    desktop = Path(desktop or Path.home()/'Desktop')
    paths = [desktop/'top-1m.csv', desktop/'malicious_phish.csv',
             desktop/'PhiUSIIL_Phishing_URL_Dataset.csv']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    rows = []
    def add(frame, column, label, source, depth, path):
        for index,value in frame[column].items():
            rows.append(dict(url=str(value),label=label,source=source,depth_type=depth,
                             original_source_input=str(frame.loc[index,'original_source_input']),
                             source_snapshot_path=str(path),source_snapshot_sha256=hashes[str(path)],
                             source_row_number=int(index)+ (1 if column==1 else 2)))
    top = pd.read_csv(paths[0],header=None,nrows=50000)
    top['original_source_input'] = top[1].astype(str)
    top[1] = 'https://www.' + top[1].astype(str)
    add(top.iloc[1000:16000],1,0,'Tranco_Top50k','root_homepage',paths[0])
    mp = pd.read_csv(paths[1],nrows=250000).dropna(subset=['url'])
    mp['original_source_input'] = mp.url.astype(str)
    mp.url = mp.url.astype(str).str.strip()
    add(mp[mp.type=='benign'].drop_duplicates('url').iloc[:15000],
        'url',0,'MaliciousPhish_Deep','deep_path',paths[1])
    phi = None
    if paths[2].exists():
        phi = pd.read_csv(paths[2],nrows=50000,usecols=['URL','label']).dropna(subset=['URL'])
        phi['original_source_input'] = phi.URL.astype(str)
        add(phi[phi.label==1].drop_duplicates('URL').iloc[:10000],
            'URL',0,'PhiUSIIL','academic_legit',paths[2])
    phishing = mp[mp.type=='phishing'].drop_duplicates('url')
    deep = phishing.url.map(lambda u:'/' in u.split('://')[-1])
    add(phishing[~deep].iloc[:15000],'url',1,'MaliciousPhish_Root','root_domain',paths[1])
    add(phishing[deep].iloc[:15000],'url',1,'MaliciousPhish_Deep','deep_path',paths[1])
    if phi is not None:
        add(phi[phi.label==0].drop_duplicates('URL').iloc[:10000],
            'URL',1,'PhiUSIIL','academic_phish',paths[2])
    frame = pd.DataFrame(rows)
    parsed = [canonicalize(u) for u in frame.url]
    for i,col in enumerate(('clean_url','scheme','psl_domain','path_query')):
        frame[col] = [p[i] for p in parsed]
    # Raw candidates are returned for review before any rejection/deduplication.
    for p,h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:
            raise RuntimeError('URL source changed during loading')
    return frame
