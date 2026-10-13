"""LF-delimited JSON records, preserving Unicode characters inside email text."""
import json
from pathlib import Path


def read_jsonl(path):
    """Read records without str.splitlines(), which also splits U+0085/U+2028.

    A malformed or nonfinite record raises with the actual physical line number;
    no row is silently discarded or repaired into a different message.
    """
    def reject_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    records = []
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line, parse_constant=reject_constant))
            except ValueError as error:
                raise ValueError(f'{path}: JSONL record {number}: {error}') from error
    return records
