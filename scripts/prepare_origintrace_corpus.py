"""Audit local corpora and export grouped development/regression splits.

No downloads, label invention, live URL visits, or model replacement. Public-corpus
text stays in ignored local data artifacts; the report contains aggregate counts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.resources
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.corpus_integrity import digest, email_identity, url_identity, split_for_group, validate_source

csv.field_size_limit(32 * 1024 * 1024)


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def prepare(registry: Path, output: Path, desktop: Path, max_per_class: int = 20000) -> dict:
    specs = json.loads(registry.read_text(encoding="utf-8"))["sources"]
    ids = [spec["id"] for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("Source IDs must be unique")
    for spec in specs:
        validate_source(spec)
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "corpus.sqlite3"
    # Rebuilding this explicitly named output is safe; all original corpora stay intact.
    db = sqlite3.connect(db_path)
    # Derived artifacts can be rebuilt; WAL avoids copying large rollback journals
    # for every batch while retaining transactional commits.
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-131072")
    db.executescript("""
      DROP TABLE IF EXISTS records;
      DROP TABLE IF EXISTS origins;
      CREATE TABLE records (
        id TEXT PRIMARY KEY, modality TEXT, raw_input TEXT, canonical TEXT,
        group_id TEXT, label INTEGER, conflict INTEGER DEFAULT 0,
        has_development INTEGER, has_regression INTEGER, has_final INTEGER
      );
      CREATE TABLE origins (id TEXT, source TEXT, row_number INTEGER,
                            UNIQUE(id, source, row_number));
    """)
    reports = []
    review_queue = []
    for spec in specs:
        path = Path(spec["path"].replace("{desktop}", str(desktop)))
        if not path.is_absolute():
            path = ROOT / path
        counts = Counter()
        stats = {"id": spec["id"], "path": str(path), "role": spec["role"],
                 "label_scope": spec["label_scope"], "collection_period": spec.get("collection_period"),
                 "provenance": spec.get("provenance"), "limitations": spec.get("limitations", [])}
        if not path.exists():
            stats["status"] = "missing"
            reports.append(stats)
            continue
        print("Auditing " + spec["id"], flush=True)
        stats.update(status="audited", sha256=file_hash(path), size_bytes=path.stat().st_size)
        records = []
        origins = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, fieldnames=spec.get("columns"))
            required = [spec["input_column"]]
            if spec.get("label_column"):
                required.append(spec["label_column"])
            if set(required) - set(reader.fieldnames or []):
                raise ValueError("Missing columns in " + spec["id"])
            for row_no, row in enumerate(reader, 1):
                counts["rows"] += 1
                if row_no % 100000 == 0:
                    print(f"  {spec['id']}: {row_no:,} rows checked", flush=True)
                raw = (row.get(spec["input_column"]) or "").strip()
                label_text = (row.get(spec.get("label_column", "")) or "").strip()
                counts["original_label:" + label_text] += 1
                if not raw:
                    counts["missing_input"] += 1
                    continue
                if spec["role"] == "review_only":
                    counts["quarantined_unverified_label_scope"] += 1
                    if counts["quarantined_unverified_label_scope"] <= 100:
                        review_queue.append({"source": spec["id"], "row_number": row_no,
                                             "input_sha256": digest(raw), "source_label": label_text,
                                             "reason": spec["label_scope"]})
                    continue
                label = spec["label_map"].get(label_text)
                if label is None:
                    counts["excluded_other_label"] += 1
                    continue
                modality = spec["modality"]
                if modality == "email" and not 50 <= len(raw) <= 25000:
                    counts["excluded_length"] += 1
                    continue
                try:
                    canonical, group = (email_identity(raw) if modality == "email"
                                        else url_identity(raw, modality == "domain"))
                except ValueError as exc:
                    counts["excluded_invalid:" + str(exc)] += 1
                    continue
                key = digest(modality + ":" + canonical)
                counts["accepted_label:" + str(label)] += 1
                roles = [int(spec["role"] == role) for role in ("development", "regression", "final_holdout")]
                records.append((key, modality, raw, canonical, group, label, *roles))
                origins.append((key, spec["id"], row_no))
                if len(records) >= (5000 if modality == "email" else 25000):
                    insert(db, records, origins)
                    records.clear()
                    origins.clear()
        insert(db, records, origins)
        stats["counts"] = dict(counts)
        reports.append(stats)
    db.executescript("""
      CREATE INDEX idx_group ON records(group_id);
      CREATE TEMP TABLE reserved_final_groups AS
          SELECT DISTINCT group_id FROM records WHERE has_final=1;
      CREATE INDEX idx_final_group ON reserved_final_groups(group_id);
      CREATE TEMP TABLE development_groups AS
          SELECT DISTINCT group_id FROM records WHERE has_development=1;
      CREATE INDEX idx_dev_group ON development_groups(group_id);
    """)
    db.commit()
    eligible = """SELECT r.* FROM records r NOT INDEXED WHERE conflict=0
       AND NOT EXISTS (SELECT 1 FROM reserved_final_groups f WHERE f.group_id=r.group_id)
       AND has_development=1 ORDER BY id"""
    exports = {}
    selections = Counter()
    group_splits = {}
    manifests = []
    files = {}
    try:
        def emit(row, split):
            key, modality, raw, canonical, group, label, *_ = row
            selection = (modality, split, label)
            if selections[selection] >= max_per_class:
                return
            selections[selection] += 1
            if group in group_splits and group_splits[group] != split:
                raise AssertionError("Split overlap: " + group)
            group_splits[group] = split
            name = modality + "_" + split + ".jsonl"
            if name not in files:
                files[name] = (output / name).open("w", encoding="utf-8", newline="\n")
            source_ids = sorted({o[0] for o in db.execute("SELECT source FROM origins WHERE id=?", (key,))})
            item = {"id": key, "modality": modality, "raw_input": raw, "label": label,
                    "group_id": group, "split": split, "sources": source_ids,
                    "label_scope": modality + "_phishing"}
            files[name].write(json.dumps(item, ensure_ascii=False) + "\n")
            manifests.append({k: v for k, v in item.items() if k != "raw_input"})
        for row in db.execute(eligible):
            emit(row, split_for_group(row[4]))
        regression = """SELECT r.* FROM records r NOT INDEXED WHERE conflict=0 AND has_regression=1
           AND NOT EXISTS (SELECT 1 FROM development_groups d WHERE d.group_id=r.group_id)
           AND NOT EXISTS (SELECT 1 FROM reserved_final_groups f WHERE f.group_id=r.group_id)
           ORDER BY id"""
        for row in db.execute(regression):
            emit(row, "regression")
        final = "SELECT * FROM records NOT INDEXED WHERE conflict=0 AND has_final=1 ORDER BY id"
        for row in db.execute(final):
            emit(row, "final_holdout")
    finally:
        for handle in files.values():
            handle.close()
    for name in files:
        exports[name] = {"sha256": file_hash(output / name), "path": str(output / name)}
    # Prevent stale split files from a prior run being consumed as current output.
    for modality in ("email", "url", "domain"):
        for split in ("train", "calibration", "validation", "development_test", "regression", "final_holdout"):
            old = output / (modality + "_" + split + ".jsonl")
            if old.name not in exports and old.exists():
                old.unlink()
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for item in manifests:
            stream.write(json.dumps(item) + "\n")
    summary = {
        "schema_version": 1,
        "max_records_per_class_per_split": max_per_class,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": file_hash(registry),
        "grouping": {"tldextract_version": importlib.metadata.version("tldextract"),
                     "psl_snapshot_sha256": hashlib.sha256(importlib.resources.files("tldextract").joinpath(".tld_set_snapshot").read_bytes()).hexdigest(),
                     "include_private_suffixes": True,
                     "split_seed": "origintrace-corpus-v1"},
        "sources": reports,
        "unique_accepted_records": db.execute("SELECT COUNT(*) FROM records").fetchone()[0],
        "conflicting_canonical_labels_quarantined": db.execute("SELECT COUNT(*) FROM records WHERE conflict=1").fetchone()[0],
        "regression_records_excluded_by_development_group": db.execute("""SELECT COUNT(*) FROM records r WHERE has_regression=1
          AND EXISTS (SELECT 1 FROM development_groups d WHERE d.group_id=r.group_id)""").fetchone()[0],
        "selected_counts": {"/".join(map(str, k)): v for k, v in sorted(selections.items())},
        "export_files": exports,
        "manifest_sha256": file_hash(output / "manifest.jsonl"),
        "review_queue": review_queue,
        "checks": {"exported_group_overlap": 0, "domain_labels_inherited_from_urls": False,
                   "fresh_final_holdout_available": any(k[1] == "final_holdout" for k in selections),
                   "verified_domain_training_available": any(k[0] == "domain" and k[1] == "train" for k in selections)},
        "limitations": [
            "Exports are capped per class per split; precision, accuracy and average precision do not estimate deployment prevalence.",
            "Email template grouping catches normalized templates; it is not exhaustive semantic near-duplicate detection.",
            "Unknown per-record collection dates prevent temporal validation of the legacy corpora.",
            "All previously inspected corpora are development/regression data, not fresh final holdouts.",
            "No corpus licensing or label quality is inferred from the file name.",
            "New source admission requires independent label/provenance review; available local files do not establish 2026 coverage."
        ]
    }
    db.close()
    (output / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "AUDIT.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def insert(db, records, origins):
    db.executemany("""INSERT INTO records(id,modality,raw_input,canonical,group_id,label,
        has_development,has_regression,has_final) VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          conflict=MAX(records.conflict, records.label != excluded.label),
          has_development=MAX(records.has_development,excluded.has_development),
          has_regression=MAX(records.has_regression,excluded.has_regression),
          has_final=MAX(records.has_final,excluded.has_final)""", records)
    db.executemany("INSERT OR IGNORE INTO origins VALUES(?,?,?)", origins)
    db.commit()


def render_report(summary):
    lines = ["# OriginTrace local corpus audit", "", "This run prepares legacy development and regression data. It does not certify production readiness.", "",
             "| Source | Role | Rows | Accepted before deduplication |", "|---|---|---:|---:|"]
    for source in summary["sources"]:
        c = source.get("counts", {})
        accepted = c.get("accepted_label:0", 0) + c.get("accepted_label:1", 0)
        lines.append(f"| {source['id']} | {source['role']} | {c.get('rows', 0):,} | {accepted:,} |")
    lines += ["", f"Conflicting canonical labels quarantined: {summary['conflicting_canonical_labels_quarantined']:,}.",
              f"Regression rows excluded for development group overlap: {summary['regression_records_excluded_by_development_group']:,}.",
              "", "## Exported split counts", "", "| Modality / Split / Label (1 = phishing) | Count |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in summary["selected_counts"].items()]
    lines += ["", "## Release blockers", "",
              "- Fresh, independently labeled final holdout available: " + str(summary["checks"]["fresh_final_holdout_available"]),
              "- Verified domain-specific training data available: " + str(summary["checks"]["verified_domain_training_available"]),
              "- Sources with missing dates/annotation history require review before any recent-state-of-the-art claim.",
              "", "## Method", "",
              "Split by a stable group hash: 60% train, 10% calibration, 10% threshold validation, 20% development test. Percentages apply to groups, so row counts differ. Exact identities are merged across sources; conflicting exact labels are quarantined. Email template equivalents share a split. URL hosts use the bundled PSL with private suffixes, and URL/domain records share the same host grouping. No popularity or URL labels are converted into domain phishing labels.",
              "", "## Limitations", ""]
    lines += ["- " + item for item in summary["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "data/origintrace_sources.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/origintrace_prepared")
    parser.add_argument("--desktop", type=Path, default=Path.home() / "Desktop")
    parser.add_argument("--max-per-class", type=int, default=20000)
    args = parser.parse_args()
    if args.max_per_class <= 0:
        parser.error("--max-per-class must be positive")
    summary = prepare(args.registry, args.output, args.desktop, args.max_per_class)
    print(json.dumps({"report": str(args.output / "AUDIT.md"), "checks": summary["checks"],
                      "selected_counts": summary["selected_counts"]}, indent=2))


if __name__ == "__main__":
    main()
