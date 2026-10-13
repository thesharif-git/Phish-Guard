"""Export existing URL/reputation evidence without changing models or corpora."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.domain_evidence import review_record, write_review_queue

def export(output=None, url_path=None, reputation_path=None):
    output = Path(output or ROOT/'data/domain_evidence_review.jsonl')
    sources = [(Path(url_path or ROOT/'data/clean_url_corpus.csv'),'url'),
               (Path(reputation_path or Path.home()/'Desktop/3_malicious_domains.csv'),'reputation')]
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p,_ in sources}
    def records():
        for path,kind in sources:
            with path.open(encoding='utf-8-sig',newline='') as stream:
                for number,row in enumerate(csv.DictReader(stream),2):
                    raw = row['url'] if kind=='url' else row['Domain']
                    yield review_record(raw,source=row.get('source') or path.name,
                        source_label=row.get('label') if kind=='url' else None,
                        source_label_scope='url_phishing' if kind=='url' else 'unverified_reputation',
                        snapshot_path=path,snapshot_sha256=before[str(path)],source_row_number=number,
                        observation_date=row.get('Last_Analysis_Date') or None,
                        observation_date_kind='source_reported_last_analysis' if kind=='reputation' else None,
                        metadata=row)
        # Validate before committing the atomic export.
        for path,_ in sources:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=before[str(path)]:
                raise RuntimeError('Source snapshot changed during export')
    result=write_review_queue(output,records())
    result.update(input_sha256=before,scope='all rows of existing URL export and local reputation CSV; no inferred labels')
    output.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    print(json.dumps(export(),indent=2),flush=True)
