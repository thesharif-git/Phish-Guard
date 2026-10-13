"""Download public source snapshots only; never execute mail or follow mail URLs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/email_root_audit_sources_20261003'
SOURCES = {
    'spamassassin_readme.html': 'https://spamassassin.apache.org/old/publiccorpus/readme.html',
    'spamassassin_spam.tar.bz2': 'https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2',
    'spamassassin_spam2.tar.bz2': 'https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2',
    'phisherman_tree.json': 'https://api.github.com/repos/ChamberZ1/phisherman/git/trees/main?recursive=1',
    'brown_phish_bowl.html': 'https://it.brown.edu/alerts/phish-bowl',
    'brown_phish_bowl_alternative.html': 'https://it.brown.edu/dsf_phishing-no-js',
}


def fetch(url, limit=12*1024*1024):
    with urlopen(Request(url, headers={'User-Agent': 'OriginTrace-Email-Source-Audit/1.0'}), timeout=25) as response:
        data = response.read(limit+1)
        if len(data) > limit:
            raise ValueError('Public snapshot exceeds bounded acquisition limit')
        return data, response.geturl()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'acquisition.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for filename, url in SOURCES.items():
        path = OUT / filename
        if path.exists():
            print('preserved existing ' + filename, flush=True)
            continue
        data, final_url = fetch(url)
        path.write_bytes(data)
        manifest.append({'path': filename, 'url': url, 'final_url': final_url,
                         'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                         'acquired_utc': datetime.now(timezone.utc).isoformat(),
                         'role': 'Provenance investigation/review candidates, not approved training or gold evaluation'})
        (OUT / 'acquisition.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(filename + ' ' + str(len(data)), flush=True)
    # Selection is by publisher label/date, before model predictions. Download pages, not email destinations.
    index = OUT / 'brown_index.json'
    if index.exists():
        rows = json.loads(index.read_text(encoding='utf-8'))
        selected = [r for r in rows if r['label'] == 'Legitimate Email']
        selected += [r for r in rows if r['label'].startswith('Phishing Email')][:45]
        for row in selected:
            filename = 'brown_pages/' + hashlib.sha256(row['url'].encode()).hexdigest()[:20] + '.html'
            path = OUT / filename
            if path.exists():
                continue
            data, final_url = fetch(row['url'], 3*1024*1024)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest.append(dict(row, path=filename, final_url=final_url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                acquired_utc=datetime.now(timezone.utc).isoformat(), role='Publisher adjudication evidence; excerpt extraction still needs review; no model-selected sampling'))
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(filename + ' ' + row['label'], flush=True)


if __name__ == '__main__':
    main()
