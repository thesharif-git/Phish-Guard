# OriginTrace corpus repair and source admission

The local legacy datasets are baseline and regression resources. They do not establish 2026 collection coverage, independent labels, or production readiness. This work does not replace the deployed models or change application routes.

## Run

From the repository root with the project dependencies installed:

```powershell
python scripts/prepare_origintrace_corpus.py
python -m unittest discover -s tests -p test_corpus_integrity.py
python scripts/train_audited_candidates.py
```

The preparer uses `data/origintrace_sources.json`. Override `--desktop` for a different data directory. Output goes to the ignored `data/origintrace_prepared/` directory. The candidate trainer consumes only checksum-verified exports named by `audit.json`, and stores separate development models under `models/audited_candidates/`. Neither command overwrites `models/frozen_pipeline/` or the deployed `.pkl` models.

The current `.venv` launcher points at a missing Python312 executable on this machine. A compatible Python 3.12 interpreter with the existing site-packages can run the scripts; repair that launcher before relying on the short commands above.

The verified fallback on this machine, run from the repository root, is:

```powershell
$originTracePython = 'C:\Users\shari\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $originTracePython -u -c "import sys,runpy;sys.path.insert(0,r'.venv\Lib\site-packages');runpy.run_path('scripts/prepare_origintrace_corpus.py',run_name='__main__')"
& $originTracePython -u -c "import sys,runpy;sys.path.insert(0,r'.venv\Lib\site-packages');runpy.run_path('scripts/train_audited_candidates.py',run_name='__main__')"
```

## Repairs

### Domain admission correction

The older `clean_and_curate_master_datasets.py` now preserves selected original
URL/hostname/ranking/reputation records in `data/domain_candidate_review.jsonl`.
It does not derive domain labels from uniform sampled URL labels, popularity,
or reputation-file membership, and does not overwrite the legacy domain CSV.
Conflicting URL assertions remain in this review queue before URL filtering.
Original source input, full hostname, source file checksum, source row number,
and available source-reported analysis dates are retained. Missing observation
dates remain null; registration or export times are not substituted.

To export currently available evidence without rebuilding a corpus, run
`python scripts/export_domain_evidence_review.py`. This produces
`data/domain_evidence_review.jsonl` and a checksum manifest, covering all rows of
the existing URL export and local reputation CSV. Every record is unverified,
has `domain_label: null`, and has `training_eligible: false`. The original source
label is preserved as a claim; no record is silently deleted or relabelled.

Legacy combined production trainers refuse domain training before training or
saving artifacts. Use the audited corpus/candidate workflow after independently
verified domain-specific annotations are available. Existing live models are
unchanged. This is a training-admission repair, not a demonstrated model upgrade.

Verification and scope: `benchmarks/domain_admission_fix/REPORT.md`.

The default cap is 20,000 records per class per split, selected by stable identity order. These capped evaluation sets do not reproduce deployment prevalence; precision, accuracy and average precision must not be presented as estimates of real-world traffic performance.

- Audit every row rather than assuming the first portion of a CSV is representative.
- Record source hashes, explicit label maps, label scope, collection-period uncertainty, and rejection reasons.
- Merge identical inputs across files and quarantine conflicting labels.
- Group email template equivalents across splits. This catches whitespace and variable URL/address/number substitutions; it does not guarantee detection of paraphrases or every near duplicate.
- Group URLs and verified domains by the bundled offline PSL, including private hosting suffixes. Preserve path case, reserved percent escapes, query order, and fragments when matching identities.
- Split groups into 60% training, 10% probability calibration, 10% threshold validation, and 20% development testing. Keep exported regression sources domain-disjoint from all development data.
- Never transfer a URL's phishing label to its host or registrable domain. Domain models require independently verified domain-specific annotations.
- Keep fraud labels, popularity ranks, generic good/bad URL labels, and unverified reputation metadata in a review queue until their task meaning and provenance are established.
- Keep previously inspected PhiUSIIL as regression data. Row order and a tail slice do not establish independence from randomly sampled training data.

## New data admission

Add sources to the registry after documenting their publisher, original source identities, acquisition dates, usage terms, annotation procedure, and task-specific label scope. A local CSV should provide raw input and explicit labels, with acquisition timestamps and evidence references when available. Do not rely on file names or model predictions for ground truth.

For a fresh final holdout, use `role: final_holdout`, verified annotations, a documented collection period, and `previously_used_for_development: false`. Its complete groups are reserved from development exports. The candidate trainer deliberately does not score final-holdout data; freeze model selection first. A previously used source cannot be made fresh merely by changing the registry flag.

For domains, add `label_scope: domain_phishing` and `annotation_verified: true` only after review establishes that the verdict describes the domain at a specified time. Track legitimate shared hosting and compromised pages separately from attacker-controlled domains.

## Contemporary sources examined

- [PhiShark2026 paper](https://arxiv.org/html/2608.23199v1) describes timestamped web and infrastructure evidence, hosting-aware attribution, and controlled research access. No data from this corpus has been obtained or added. Request authorized access through its published procedure before evaluating it for ingestion.
- [CIC-Trap4Phish2025](https://www.unb.ca/cic/datasets/trap4phish2025.html) focuses on attachments and QR phishing. It is not a drop-in email-body or domain-label replacement.
- [PhiUSIIL documentation](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset) establishes the label direction: 1 is legitimate, 0 is phishing. The local copy remains regression data, not new temporal validation.

Fresh, independently labeled 2026 email and domain corpora are not currently available locally. Keep that limitation in every performance report until it is resolved.

## Completed baseline diagnosis

The raw URL baseline produced 100% false positives on the domain-disjoint PhiUSIIL regression export at its strict threshold. A controlled scheme-independent representation experiment still produced 96.91% false positives and lower recall. Protocol-format confounding is therefore insufficient to explain or repair the cross-source failure. Neither URL candidate is suitable for promotion.

Both experiments use the same grouped data, model family, calibration split and validation threshold procedure. The representation experiment was motivated by an inspected regression result, so its result is a development diagnostic, not a fresh independent evaluation. Full metrics are retained under `benchmarks/corpus_repair/`.

To reproduce the raw representation, pass `--url-view raw --output models/audited_candidates/raw`. The default representation is scheme-independent and writes to `models/audited_candidates/normalized/`. These are research outputs only. Keep raw URLs for infrastructure and security evidence; scheme removal applies only to the experimental lexical view.

Next, establish label comparability across sources, include independently reviewed contemporary legitimate examples and attack families in development, reserve a new source/campaign/time-disjoint final test, and then compare modern representations and architectures. Do not tune a model to the existing regression set and relabel that result as independent validation.
