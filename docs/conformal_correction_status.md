# Model usage correction — verified implementation

**Current status (2026-10-04, after the generalization root-cause audit):** saved
calibration is historical development evidence. It no longer matches the scorer
after evidence-truthfulness repairs, so the current engine requests review.
No retraining or recalibration was performed in this audit: those steps remain
gated on reliable source/time generalization. A
separate fold-owned LightGBM candidate has been trained and evaluated, but is not
promoted: separate author-labeled data shows weak discrimination and high false
alarms. The metrics below remain body-only development measurements, not proof
of current-traffic phishing accuracy. See
`benchmarks/email_retraining_20261003/REPORT.md` for the controlled comparison,
original-MIME challenge and operational replay. See
`benchmarks/email_generalization_root_20261003/REPORT.md` for the newer causal
probes, upstream label trace, overlap checks and repair verification.

During the earlier calibration correction, datasets and trained model artifacts
were not changed, and parameters were fitted on the existing validation split.
Those saved parameters are now stale; this root-cause audit did not refit them.
The existing test split remains
ordinary development validation, not a new untouched research holdout.

## Changes
- Email scoring preserves the complete, unrounded specialist probability.
  Confidence priors and auxiliary risk scores do not rescale it.
- BEC, URL, contradiction, QR and threat findings remain separate evidence.
  They can request review but do not invent or raise a statistical probability.
- Missing optional evidence cannot downgrade a model-based quarantine.
  Missing core model output requests review rather than generating a probability.
- Uncertainty and belief estimates are null/not_estimated rather than arbitrary
  numerical claims. No learned fusion or Bayesian uncertainty claim is made.
- Root URLs use the URL specialist. Domain inference receives registrable domains,
  with named numeric feature schemas matched to saved models.
- Batch calibration and live inference share input construction. Two actual
  model examples matched to absolute tolerance 1e-12.
- Calibrated statistical decisions are separate from operational heuristic warnings.
  Calibration is bound to current source/model fingerprints; ties and empty sets
  are handled conservatively.

## Recomputed probability-branch metrics
All 2,544 existing test rows were evaluated after validation calibration:
- AUC: 0.9964559678
- Brier: 0.0232382972
- Empirical prediction-set coverage: 95.1651%
- Quarantine: 1,050 labelled phishing, 7 benign; recall 86.7052%; benign FPR 0.5251%.
- Empirical quarantine FDR: 0.6623%; this is not a theoretical guarantee.
- Model-branch warnings: 146 labelled phishing, 65 benign.
- Model-branch delivery: 15 labelled phishing, 1,261 benign.
- Empty prediction sets: 78; all route to warning, never safe delivery.

These are statistical-branch decisions. Additional live heuristic warnings are
not included in this table. The previous full-fusion probabilities had AUC 0.9890
and Brier 0.0700; retaining the specialist probability avoids that distortion.
No overall live policy performance or attack-prevention claim is implied.

## Checks and limits
20 targeted regression tests passed, including root-URL routing, domain scope,
probability preservation, auxiliary review policies, absent/stale calibration,
conservative ties and empty sets. Final model/code and dataset fingerprints match
the saved calibration/results at that earlier run, not the subsequently repaired
scorer. Current verification is reported separately in the generalization audit.

The domain specialist itself remains weak; input normalization does not change
its available information or make it a stronger classifier. BEC inference remains
expensive; selectively skipping it requires independent benefit/recall evaluation.
An independently reviewed, source/time-separated holdout is still needed before
claiming generalization. Coverage/Type I interpretations require independent
exchangeable observations; grouped splitting alone does not establish those
assumptions. Restart the application to reload code and calibration.
