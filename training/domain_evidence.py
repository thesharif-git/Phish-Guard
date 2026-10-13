"""Review-only domain evidence. Source assertions never become training labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import tldextract

PSL = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

def reject_legacy_domain_training():
    raise ValueError(
        'Legacy domain training is disabled: URL labels, popularity ranks and '
        'reputation-file membership are not verified domain labels. Use the '
        'audited corpus workflow with independently verified domain annotations.'
    )

def review_record(raw_input, *, source, source_label=None, source_label_scope,
                  snapshot_path=None, snapshot_sha256=None, source_row_number=None,
                  observation_date=None, observation_date_kind=None, metadata=None):
    """Preserve exact input and source metadata, including conflicting assertions.

    Dates remain source assertions, not verified timestamps. Never use a file
    modification time, export time or registration date as attack observation.
    """
    raw = str(raw_input)
    if observation_date is not None and str(observation_date).strip().casefold() in {
        '', 'unknown', 'none', 'null', 'nan', 'nat', 'n/a'
    }:
        observation_date = None
    host = registrable = None
    error = None
    try:
        parts = urlsplit(raw if '://' in raw else 'http://' + raw)
        host = parts.hostname
        if not host:
            raise ValueError('missing_hostname')
        host = host.encode('idna').decode('ascii').lower().rstrip('.')
        registrable = PSL(host).top_domain_under_public_suffix or None
    except (ValueError, UnicodeError) as exc:
        error = str(exc)
    item = dict(schema_version=1, raw_input=raw, full_hostname=host,
                registrable_domain=registrable, source=source,
                source_label=source_label, source_label_scope=source_label_scope,
                domain_label=None, review_status='unverified', training_eligible=False,
                source_snapshot_path=str(snapshot_path) if snapshot_path else None,
                source_snapshot_sha256=snapshot_sha256,
                source_row_number=source_row_number, observation_date=observation_date,
                observation_date_kind=observation_date_kind if observation_date is not None else None,
                observation_date_verified=False, parsing_error=error,
                source_metadata=metadata or {})
    item['evidence_id'] = hashlib.sha256(json.dumps(item,sort_keys=True,ensure_ascii=False,
                                                   allow_nan=False).encode()).hexdigest()
    return item

def write_review_queue(path, records):
    """Atomic, lossless JSONL export; never writes a domain training corpus."""
    path = Path(path)
    if path.suffix != '.jsonl':
        raise ValueError('Review exports must be JSONL, separate from training CSVs')
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.jsonl.pending')
    count = 0
    try:
        with pending.open('w',encoding='utf-8',newline='\n') as stream:
            for record in records:
                if record.get('training_eligible') is not False or record.get('domain_label') is not None:
                    raise ValueError('Review evidence cannot contain an admitted domain label')
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                count += 1
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return dict(path=str(path),records=count,training_eligible_records=0,
                exported_at=datetime.now(timezone.utc).isoformat(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
