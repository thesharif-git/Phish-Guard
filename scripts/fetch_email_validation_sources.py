"""Acquire immutable candidate email snapshots; never promote them to training."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/external_email_validation/nazario_2024_2025'
FILES = {
    'README.txt': 'https://monkey.org/~jose/phishing/README.txt',
    'LICENSE.txt': 'https://monkey.org/~jose/phishing/LICENSE.txt',
    'phishing-2024.mbox': 'https://monkey.org/~jose/phishing/phishing-2024',
    'phishing-2025.mbox': 'https://monkey.org/~jose/phishing/phishing-2025',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for name, url in FILES.items():
        target = OUT / name
        if target.exists():
            raise RuntimeError('Refusing to overwrite a source snapshot: ' + name)
        pending = OUT / (name + '.pending')
        total = 0
        sha = hashlib.sha256()
        request = Request(url, headers={'User-Agent': 'OriginTrace-email-validation/1.0'})
        with urlopen(request, timeout=45) as response, pending.open('wb') as stream:
            last_modified = response.headers.get('Last-Modified')
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 64 * 1024 * 1024:
                    raise RuntimeError('Candidate source exceeded download budget')
                sha.update(chunk)
                stream.write(chunk)
        pending.replace(target)
        files.append({'path': target.relative_to(ROOT).as_posix(), 'source_url': url,
                      'bytes': total, 'sha256': sha.hexdigest(), 'last_modified': last_modified})
        print('Downloaded ' + name + ': ' + str(total) + ' bytes', flush=True)
    manifest = {'complete': True, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_author': 'Jose Nazario', 'source_label': 'phishing',
                'label_status': 'source_hand_classified_not_independently_adjudicated',
                'license': 'CC-BY-4.0', 'files': files, 'purpose': 'candidate_external_email_evaluation',
                'admitted_to_training': False, 'approved_independent_benchmark': False,
                'limitations': ['Single collector inbox and positive-only corpus',
                    'Maintainer acknowledges possible classification errors',
                    'Mailbox year and Date headers are not verified reception timestamps',
                    'Requires overlap/campaign screening and independent review',
                    'Not sufficient for FPR, specificity, AUC or precision evaluation alone']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
