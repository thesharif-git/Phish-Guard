"""Acquire and inspect a small English source; never admit labels or fit models.

Preserve all original rows. Repetition is reported, not silently removed.
The publisher mixes real and artificial mail without per-message origin proof.
"""
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/email_roadmap_v1_20261006/source_candidate_zenodo_13474746'
URL = 'https://zenodo.org/records/13474746/files/Phishing_validation_emails.csv?download=1'
PAGE = 'https://zenodo.org/records/13474746'
PUBLISHER_MD5 = '1bf8ec0fe3f67e12dd275ce5b2b91b69'


def inspect_csv(payload):
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    rows = list(reader)
    if not reader.fieldnames or not {'Email Text', 'Email Type'}.issubset(reader.fieldnames):
        raise ValueError('Unexpected publisher CSV schema')
    groups = defaultdict(list)
    malformed, empty, labels = [], [], Counter()
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            malformed.append(index)
        body = row.get('Email Text') or ''
        label = row.get('Email Type') or ''
        labels[label] += 1
        normalized = re.sub(r'\s+', ' ', body).strip().casefold()
        if not normalized:
            empty.append(index)
        else:
            key = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            groups[key].append(dict(source_row_index=index, publisher_label=label))
    repeats = [dict(normalized_body_sha256=key, rows=members) for key, members in groups.items() if len(members) > 1]
    conflicts = [group for group in repeats if len({r['publisher_label'] for r in group['rows']}) > 1]
    return dict(rows=len(rows), columns=reader.fieldnames, publisher_label_counts=dict(labels),
        empty_body_row_indices=empty, malformed_row_indices=malformed,
        normalized_unique_nonempty_bodies=len(groups), repeated_body_groups=len(repeats),
        repeat_excess_rows=sum(len(group['rows']) - 1 for group in repeats),
        conflicting_label_body_groups=len(conflicts), repeated_bodies=repeats,
        unknown_label_counts={label: count for label, count in labels.items()
                              if label not in ('Safe Email', 'Phishing Email')},
        normalization='whitespace collapsed, casefold; no words, URLs or emails masked',
        approved_for_training=False, approved_independent_benchmark=False,
        limitations=['Publisher describes mixed real-world/artificial messages',
            'Per-row native/real/synthetic provenance and adjudication not established',
            'CSV is not original MIME or trusted mailbox authentication evidence',
            'No labels independently reviewed by this project; no accuracy evaluated'])


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main():
    # Retry a failed network attempt in place; never overwrite acquired evidence.
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'corpus.csv').exists() or (OUT / 'results.json').exists():
        raise FileExistsError('Refusing to overwrite acquired candidate evidence')
    save(OUT / 'run_status.json', dict(complete=False, stage='acquiring_candidate'))
    try:
        request = Request(URL, headers={'User-Agent': 'OriginTrace-source-audit/1.0'})
        with urlopen(request, timeout=30) as response:
            payload = response.read(2 * 1024 * 1024 + 1)
            final_url = response.url
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError('Candidate exceeds the 2 MiB inspection budget')
        md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        if md5 != PUBLISHER_MD5:
            raise ValueError('Downloaded CSV differs from published release checksum')
        (OUT / 'corpus.csv').write_bytes(payload)
        results = inspect_csv(payload)
        save(OUT / 'acquisition.json', dict(source_url=PAGE, download_url=URL, final_url=final_url,
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(), bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(), md5=md5, publisher_checksum_verified=True,
            purpose='source_integrity_inspection_only', training_reuse_permission_established=False,
            admitted_to_training=False, approved_independent_benchmark=False))
        save(OUT / 'results.json', results)
        save(OUT / 'run_status.json', dict(complete=True, stage='inspection_complete',
            training_started=False, calibration_fitted=False, candidate_admitted=False))
        print(json.dumps({k: v for k, v in results.items() if k != 'repeated_bodies'}), flush=True)
    except Exception as error:
        save(OUT / 'run_status.json', dict(complete=False, stage='failed',
            error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
