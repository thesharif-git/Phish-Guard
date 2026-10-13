import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_origintrace_corpus import prepare
from training.corpus_integrity import email_identity, split_for_group, url_identity, validate_source
from training.prepared_corpus import load_prepared


class CorpusIntegrityTests(unittest.TestCase):
    def test_url_identity_preserves_meaning_sensitive_components(self):
        canonical, group = url_identity("HTTPS://Example.CO.UK/Login%2fAdmin?q=A%2Bb#step")
        self.assertEqual(canonical, "https://example.co.uk/Login%2FAdmin?q=A%2Bb#step")
        self.assertEqual(group, "host:example.co.uk")
        self.assertNotEqual(url_identity("https://example.co.uk/login")[0],
                            url_identity("https://example.co.uk/Login")[0])

    def test_host_groups_prevent_domain_and_template_leakage(self):
        self.assertEqual(url_identity("https://shop.example.co.uk/a")[1],
                         url_identity("example.co.uk", domain_only=True)[1])
        self.assertNotEqual(url_identity("a.github.io")[1], url_identity("b.github.io")[1])
        a = email_identity("Invoice 123: send to first@example.com at https://example.com/1")[1]
        b = email_identity("Invoice 999: send to second@example.com at https://example.com/2")[1]
        self.assertEqual(a, b)
        self.assertEqual(split_for_group(a), split_for_group(b))

    def test_invalid_hosts_and_url_as_domain_are_rejected(self):
        for value in ("http://%70aypal.com", "http://example.com:bogus", "https://a b.com"):
            with self.assertRaises(ValueError):
                url_identity(value)
        with self.assertRaises(ValueError):
            url_identity("https://example.com/phish", domain_only=True)

    def test_scope_and_final_holdout_admission(self):
        base = {"id": "test", "modality": "domain", "role": "development",
                "label_scope": "url_phishing", "label_map": {"bad": 1}}
        with self.assertRaises(ValueError):
            validate_source(base)
        base.update(modality="url", role="final_holdout")
        with self.assertRaises(ValueError):
            validate_source(base)

    def test_exports_quarantine_conflicts_and_exclude_regression_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = []
            corpora = {
                "dev": [("https://example.com/a", "good"), ("https://example.com/a", "bad"),
                        ("https://second.org/a", "good"), ("https://third.net/a", "bad")],
                "reg": [("https://example.com/b", "bad"), ("https://other.org/a", "good")]
            }
            for name, rows in corpora.items():
                path = root / (name + ".csv")
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["url", "label"])
                    writer.writerows(rows)
                specs.append({"id": name, "path": str(path), "modality": "url",
                              "input_column": "url", "label_column": "label",
                              "label_map": {"good": 0, "bad": 1},
                              "label_scope": "url_phishing",
                              "role": "development" if name == "dev" else "regression"})
            registry = root / "registry.json"
            registry.write_text(json.dumps({"sources": specs}), encoding="utf-8")
            first = prepare(registry, root / "out", root)
            self.assertEqual(first["conflicting_canonical_labels_quarantined"], 1)
            self.assertEqual(first["regression_records_excluded_by_development_group"], 1)
            self.assertFalse(first["checks"]["verified_domain_training_available"])
            reg = [json.loads(line) for line in (root / "out/url_regression.jsonl").read_text().splitlines()]
            self.assertEqual([item["group_id"] for item in reg], ["host:other.org"])
            second = prepare(registry, root / "out", root)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(first["export_files"], second["export_files"])
            loaded, _ = load_prepared(root / "out")
            self.assertEqual(len(loaded["url_regression"]), 1)
            with (root / "out/url_regression.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(root / "out")


if __name__ == "__main__":
    unittest.main()
