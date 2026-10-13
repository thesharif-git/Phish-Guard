"""Assemble measured evidence and closure checks. No fitting or deployment."""
import json
from datetime import datetime, timezone
from pathlib import Path
from scripts.run_email_retraining_experiment import ROOT, OUT, sha
from services.security_pipeline import security_pipeline_id


def main():
    experiment = json.loads((OUT / 'results.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    diagnostic = json.loads((OUT / 'generalization_diagnostics.json').read_text())
    replay = json.loads((OUT / 'operational_replay_with_source_links/results.json').read_text())
    challenge = json.loads((OUT / 'raw_challenge_results.json').read_text())
    integration = json.loads((OUT / 'integration_checks.json').read_text())
    calibration = json.loads((ROOT / 'models/conformal_evaluation_results.json').read_text())
    parameters = json.loads((ROOT / 'models/conformal_params.json').read_text())
    identity = security_pipeline_id()
    assert experiment['complete'] and replay['complete'] and challenge['complete'] and diagnostic['complete']
    assert identity == replay['pipeline_id'] == calibration['pipeline_id'] == parameters['pipeline_id'] == integration['current_pipeline_id']
    protected = {p: sha(ROOT / p) == h for p, h in protocol['protected_hashes'].items() if p not in ('models/conformal_params.json', 'models/conformal_evaluation_results.json')}
    assert all(protected.values())
    closure = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'pipeline_id': identity,
               'calibration_current': True, 'calibration_schema': parameters['schema_version'],
               'original_models_and_corpus_preserved': protected,
               'candidate_promoted': False, 'email_reliability_gate_passed': False,
               'email_technical_experiment_complete': True, 'move_to_next_scanner': False,
               'release_gate_reasons': ['External candidate benign quarantine FPR exceeds the 1% operating budget',
                                       'External discrimination remains weak', 'Task labels/campaign independence not independently adjudicated',
                                       'Auxiliary review policy has excessive valid-message warnings'],
               'allowed_artifact_changes': ['Corrected Email source code', 'research candidate and evaluation artifacts',
                                          'authorized current-pipeline development calibration'],
               'source_snapshots_unchanged': {
                    'phishfuzzer': sha(ROOT / 'data/external_email_validation/phishfuzzer_original_seeds/original_seeds.json') == protocol['source_sha256'],
                    'nazario_2024': sha(ROOT / 'data/external_email_validation/nazario_2024_2025/phishing-2024.mbox') == challenge['source_sha256']}}
    assert all(closure['source_snapshots_unchanged'].values())
    (OUT / 'closure_checks.json').write_text(json.dumps(closure, indent=2), encoding='utf-8')
    table = []
    for split, container in [('Existing test', experiment['legacy']['test']), ('Separate author labels', experiment['external_author_label_comparison'])]:
        for name in ('baseline', 'candidate'):
            entry = container[name]
            m = entry['supported_metrics']
            table.append(f"| {split} | {name} | {m['rows']} | {m['accuracy']:.2%} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.4f} | {entry['quarantine_recall']:.2%} | {entry['benign_quarantine_fpr']:.2%} |")
    actions = []
    for cls in ('Phishing', 'Spam', 'Valid'):
        data = replay['per_source_class'][cls]
        a = data['actions']
        actions.append(f"| {cls} | {data['rows']} | {a.get('AUTO_QUARANTINE', 0)} | {a.get('WARNING_BANNER', 0)} | {a.get('SAFE_DELIVERY', 0)} |")
    latency = replay['latency_ms']
    report = f"""# Email Scanner — measured validation and release decision

**Calibration, controlled retraining and the evaluated integration checks completed. Email reliability did not pass the release gate.** The corrected candidate stays separate and unapproved; the existing trained model remains active. Standalone URL/Domain work remains paused. Restart the application to load current calibration into existing processes.

## Changes actually made

- Retrained one 5-fold LightGBM candidate using the same hyperparameters, with vocabulary/IDF fitted inside each training fold. This fixes inner-calibration preprocessing leakage rather than adding features. Fit time: {experiment['fit_seconds']:.2f} seconds.
- A no-vocabulary input now requests review in both legacy and fold-owned model paths. One external message has no probability; it remains in the action denominator.
- Bound calibration to feature-transform source, model files and scientific-library versions. Current identity: `{identity}`.
- Calibrated the **deployed** content-probability branch on 2,464 existing validation rows and evaluated all 2,544 existing test rows. Probability construction is shared with live inference. No FDR guarantee or learned uncertainty claim is introduced.
- Isolated Email inference to one CPU thread per tree model. Single-message structural features avoid a DataFrame construction while preserving all 16 values exactly across 12,804 corpus rows. 24 live/batch probabilities matched exactly.
- Added explicit supplementary URL evidence without modifying model text or inventing original MIME. The external adapter passes author URL metadata separately and reports attachment names without payloads as uninspected.
- Corrected a false “alignment verified” statement when sender authentication was unavailable.

The earlier MIME/Gmail/HTML/QR, model-failure, persistence and presentation repairs are documented in `../email_ingestion_repairs/REPORT.md`; their unchanged source-byte preservation tests are not phishing-accuracy evidence.

## Controlled content-model comparison

| Data | Model | Probability-supported rows | Accuracy at 0.5 | F1 at 0.5 | ROC-AUC | Brier (lower better) | Quarantine recall, all rows | Non-phishing quarantine FPR, all rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

The separate collection contains 292 original author-reported Manual/Human seeds: 101 Phishing, 92 Spam and 99 Valid. One Phishing row lacks recognized vocabulary in both models; ROC-AUC, Brier and threshold metrics disclose the remaining 291-row denominator. Action counts include all 292. Spam is **not** relabelled as phishing. These author labels and claimed dates have not been independently adjudicated by this project.

Candidate quarantine false positives are 21/191 non-phishing rows versus baseline 16/191. Its old-test improvement does not justify promotion. External AUC 95% bootstrap intervals are {diagnostic['external_auc_bootstrap_95_intervals']['baseline']} for the baseline and {diagnostic['external_auc_bootstrap_95_intervals']['candidate']} for the candidate; both include 0.5. The bootstrap assumes independent rows within class, which unverified campaign relationships may violate. See `results.json` for exact binomial quarantine intervals and supported-set coverage.

## OriginTrace offline operational replay with preserved links

| Author class | Rows | Quarantine recommendation | Warning | Delivery recommendation |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(actions)}

These are recommendations, not executed inbox actions or demonstrated attack prevention. 74/101 phishing-source rows contain HTTP URLs in separate metadata outside model text. Preserving this evidence changes review recommendations, not content probabilities. The earlier body/HTML-only replay is retained under `operational_replay/`; it omitted the separate source URL field and must not be represented as complete source-evidence evaluation.

The corrected replay uses actual local content/embedded-link models, cached BEC NLI, and local OriginTrace logic. Remote redirects, VirusTotal, receiver-authenticated SPF/DKIM/DMARC, original MIME/attachment payloads, and independently verified campaign memory are unavailable and expressly excluded. Network calls and campaign-memory writes are disabled. URL/Domain trained files are unchanged; embedded links are tested only as Email evidence.

Warning volume is high: 83/99 author-valid messages require warning and 9/99 get quarantine recommendations. That is not a usable low-false-alarm result. A warning is not counted as quarantine or prevention. Cached BEC was available for 292 rows, but 199 inputs were truncated to 500 characters; this specialist is not independently calibrated as a fraud probability.

Offline replay median: **{latency['median']:.1f} ms**; p95: **{latency['p95']:.1f} ms**; p99: **{latency['p99']:.1f} ms**. First-message cold load/inference: {latency['first_message']/1000:.1f} seconds. Cached BEC median alone: {replay['stage_latency_ms']['bec_intent']['median']:.1f} ms. These are observations on this laptop with other work running, not isolated hardware-certified throughput or an under-8-ms claim. Diagnostic 100-run single-message model timings are recorded separately and exclude auxiliary/network work.

## Original RFC/MIME challenge

All 403 original 2024 collector messages were parsed using the corrected upload parser. Exact model-text repeats/old-corpus matches were excluded reproducibly, leaving 360 positive-source candidates. No hard or malformed message was deleted merely for a low score. Eight retained messages have incomplete text and require review.

- Baseline: 134/360 have probability at least 0.5; 48 quarantine, 116 warning, 196 model-branch delivery recommendations.
- Corrected candidate: 129/360 have probability at least 0.5; 48 quarantine, 122 warning, 190 model-branch delivery recommendations.

This is a positive-source content challenge, not a two-class accuracy benchmark. Its collector labels are not independent project adjudications. FPR, precision and ROC-AUC cannot be estimated without benign examples. Separate original 2025 messages remain unscored by this experiment.

## What the evidence establishes

1. Fixing the inner-fold preprocessing problem is necessary but **does not resolve generalization**. Old-test AUC is essentially unchanged and external discrimination remains weak. This does not establish that every training example was memorized, or prove LightGBM's capacity is the primary limitation.
2. The classifier is validated mainly against historical body-only source labels, while current phishing/non-phishing decisions often require destination, attachment or identity context. Its old distribution does not establish the desired phishing/BEC task. All 6,402 old positives originate from one named dataset; 2,262 Enron rows are exclusively negative. This is a measurable source/task confound, not proof that every label is wrong.
3. The loss also occurs with original MIME, so it cannot be attributed solely to JSON ingestion. Missing separate link evidence explained part of the operational review loss and has been corrected.
4. Broad auxiliary rules create many warnings but do not repair the content model. Their false-alarm behavior and cached NLI latency are additional measured operational bottlenecks.
5. Conformal calibration does not repair bad ranking, invalid task labels or distribution shift. The 1% validation budget does not establish 1% FPR on this separate traffic, and never establishes FDR control. Code correctness and calibration version matching are separate from deployment accuracy.

## Data integrity and next required change

The original 12,804-row corpus passed exact masked-text uniqueness, label-conflict, missing-field and recorded cross-partition group/text checks. Recorded group IDs are not independently verified campaigns. The source snapshots and all deployed trained-model files were preserved byte-for-byte. External provenance/exclusion records and encoding/length/near-overlap flags remain auditable; difficult messages and length outliers are not silently discarded. No independently reviewed label or commercial-data admission attestation was fabricated.

The next training corpus must distinguish phishing/BEC, ordinary spam and valid messages; include legitimate account alerts/verification codes/invoices as hard negatives; preserve raw MIME, links, and trustworthy provenance; adjudicate ambiguous cases with evidence and record reviewer disagreements. Split by independently established campaign/source/time, fit transforms only on training, use a separate calibration partition, and keep a new final holdout uninspected. The current 292 rows and 2024 challenge are now development evidence, not a fresh future holdout.

These requirements are not resolved by downloading an arbitrary larger dataset, deleting errors, relabelling according to this model, raising thresholds after inspection, or adding more heuristic features. The existing body-only corpus cannot supply missing authentication/destination ground truth through a code change. Until task-valid corpus and fresh traffic evidence establish the declared false-alarm budget and useful recall, neither this candidate nor the current engine is certified for automatic quarantine. Standalone scanner work should remain paused.

## Auditable artifacts

- `protocol.json`: pre-training frozen data/experiment choices and original protected hashes.
- `candidate_email_model.pkl`: trained, unapproved fold-owned candidate.
- `results.json` and `decisions.csv`: baseline/candidate measurements and every decision.
- `generalization_diagnostics.json`: source/context audit, latency and error-review queue; labels unchanged.
- `operational_replay_with_source_links/`: frozen protocol, every local-sensor decision and timing.
- `raw_challenge_results.json` / `raw_challenge_decisions.jsonl`: original-message challenge decisions.
- `integration_checks.json`: feature parity, live/batch parity and current calibration match.
- `closure_checks.json`: actual implementation completion versus failed reliability gate.
- `../../models/conformal_evaluation_results.json`: current deployed-branch development calibration results.

The application must be restarted to load the saved calibration in processes that already imported the prior singleton. This report is not a claim that Email reliability is complete.
"""
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(closure, indent=2))


if __name__ == '__main__':
    main()
