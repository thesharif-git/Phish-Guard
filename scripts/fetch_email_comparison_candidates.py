"""Acquire published comparison candidates for inspection, not gold evaluation.

Only original seeds and author notes; no synthetic expansion, repository code,
message destinations, model scoring, or training. Source license remains a gate.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/external_email_validation/phishfuzzer_original_seeds'
FILES = {
    'original_seeds.json': 'https://github.com/DataPhish/PhishFuzzer/raw/refs/heads/main/PhishFuzzer_emails_original_seed_v1.json',
    'ReadME.md': 'https://github.com/DataPhish/PhishFuzzer/raw/refs/heads/main/ReadME.md',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Source snapshot already complete; refusing overwrite')
    files = []
    for name, url in FILES.items():
        target = OUT / name
        if target.exists():
            # An interrupted acquisition may already have completed this file.
            files.append({'path': target.relative_to(ROOT).as_posix(), 'source_url': url,
                          'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                          'bytes': target.stat().st_size, 'retrieved_during_previous_attempt': True})
            continue
        pending = OUT / (name + '.pending')
        request = Request(url, headers={'User-Agent': 'OriginTrace-email-validation/1.0'})
        size = 0
        with urlopen(request, timeout=45) as response, pending.open('wb') as stream:
            final_url = response.geturl()
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 16 * 1024 * 1024:
                    raise RuntimeError('Source exceeded bounded acquisition size')
                stream.write(chunk)
        pending.replace(target)
        files.append({'path': target.relative_to(ROOT).as_posix(), 'source_url': url,
                      'resolved_url': final_url, 'bytes': size,
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        print(f'Downloaded {name}: {size} bytes', flush=True)
    value = {'complete': True, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
             'source': 'DataPhish/PhishFuzzer original seeds',
             'publication_url': 'https://arxiv.org/abs/2511.21448', 'files': files,
             'license_status': 'No explicit dataset reuse license established from inspected author pages',
             'admitted_to_training': False, 'approved_independent_benchmark': False,
             'model_predictions_computed': False,
             'limitations': ['Mixed old public and privately collected original seeds',
                             'Public-seed metadata includes LLM enrichment',
                             'Private claimed years are not independently verified timestamps',
                             'Labels, source fields, near-overlap and reuse terms require review',
                             'Structured JSON is not original MIME or verified receiver authentication']}
    (OUT / 'manifest.json').write_text(json.dumps(value, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
