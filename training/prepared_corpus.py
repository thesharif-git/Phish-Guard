"""Verified consumer of the corpus preparer's explicit export manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_prepared(directory: Path) -> tuple[dict, dict]:
    directory = directory.resolve()
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256((directory / "manifest.jsonl").read_bytes()).hexdigest()
    if manifest_hash != audit["manifest_sha256"]:
        raise ValueError("Corpus manifest checksum mismatch")
    datasets = {}
    groups = {}
    ids = set()
    counts = {}
    for name, export in audit["export_files"].items():
        path = (directory / name).resolve()
        if path.parent != directory or path.suffix != ".jsonl":
            raise ValueError("Export path escapes corpus directory")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != export["sha256"]:
            raise ValueError("Export checksum mismatch: " + name)
        records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                group, split, key = record["group_id"], record["split"], record["id"]
                if key in ids:
                    raise ValueError("Repeated identity in prepared exports")
                ids.add(key)
                if group in groups and groups[group] != split:
                    raise ValueError("Group leaks between splits")
                groups[group] = split
                if record["label"] not in (0, 1) or record["label_scope"] != record["modality"] + "_phishing":
                    raise ValueError("Invalid task label")
                expected = record["modality"] + "_" + split + ".jsonl"
                if name != expected:
                    raise ValueError("Record disagrees with export split")
                count_key = "/".join([record["modality"], split, str(record["label"])])
                counts[count_key] = counts.get(count_key, 0) + 1
                records.append(record)
        datasets[name.removesuffix(".jsonl")] = records
    if counts != audit["selected_counts"]:
        raise ValueError("Export counts disagree with audit")
    return datasets, audit
