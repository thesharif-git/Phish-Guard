"""Acquire one versioned, licensed research source; never approve training.

SpaPhish is Spanish. It cannot substitute for an English generalization test.
Original bytes are preserved and downloads are bounded/hash-checked.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/email_evidence_sources/spaphish_v4'
BASE = 'https://data.mendeley.com/public-files/datasets/hz2d6gz7pc/files/'
FILES = [
    ('corpus.csv', '6002fe09-36b2-4c66-980c-a3da45776dee', 5096525,
     'fdd74842d0a19fd4332bd91f90b0bcb06e045ceb2b599051b4598b65055a9cc5'),
    ('schema.json', 'eb378c47-8c13-44d2-afef-328cb1a03a43', 17798,
     '51d501037e05f2dd117f8d46c4131d08971fa1f21b220bcc401670741605cdba'),
    ('README.txt', '6c446cb1-0094-43dd-85cc-09b9860f2d5e', 3212, None),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acquired = []
    for name, identifier, expected_bytes, expected_hash in FILES:
        path = OUT / name
        url = BASE + identifier + '/file_downloaded'
        if path.exists():
            payload = path.read_bytes()
        else:
            request = Request(url, headers={'User-Agent': 'OriginTrace-email-corpus-research/1.0'})
            with urlopen(request, timeout=45) as response:
                payload = response.read(expected_bytes + 1)
            if len(payload) != expected_bytes:
                raise ValueError('Unexpected size for ' + name)
        sha = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_bytes or (expected_hash and sha != expected_hash):
            raise ValueError('Source bytes/hash mismatch for ' + name)
        if not path.exists():
            pending = path.with_suffix(path.suffix + '.pending')
            pending.write_bytes(payload)
            pending.replace(path)
        acquired.append(dict(path=name, url=url, bytes=len(payload), sha256=sha))
        print(name + ' verified ' + sha, flush=True)
    manifest = dict(source='SpaPhish', version=4,
        source_url='https://data.mendeley.com/datasets/hz2d6gz7pc/4',
        acquired_at_utc=datetime.now(timezone.utc).isoformat(),
        declared_license='CC BY 4.0', language='Spanish', files=acquired,
        purpose='Evidence-preserving source inspection and review; not approved deployment data',
        approved_for_training=False, english_generalization_evidence=False,
        original_rfc822_available=False, trusted_receiver_authentication_available=False)
    (OUT / 'acquisition.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
