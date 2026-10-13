# Email-only validation and release protocol

The controlled experiment `run_email_retraining_experiment.py` trains a candidate
to reproduce the existing source-labelled binary task with corrected fold-local
preprocessing. It does not attest independently reviewed phishing/BEC labels.
It saves the model and calibration separately and cannot deploy it automatically.
Research evaluation on published original comparison seeds is separate from
training and from commercial data admission. Spam is mapped to non-phishing for
that comparison and remains separately reported.

Source and model hashes, input contracts, split overlap, exact repeated model
texts, missing fields and outlier/recovery flags must be recorded before scoring.
Never select a replacement based only on high AUC in the repeatedly used legacy
test partition. Group IDs are declared group IDs, not proof of campaign truth.

Probabilities, prediction sets and automatic action recommendations must be
evaluated separately from auxiliary warnings. Report FPR, precision/empirical
FDR, recall, PR-AUC, ROC-AUC, Brier, uncertainty intervals, unsupported inputs and
warning volume. Keep source-reported and independently reviewed labels distinct.

Benchmark preprocessing, core model and optional sensors separately, with warm
single-message CPU median/p95/p99 and cold loading cost. Network lookups have a
separate budget and cannot be claimed part of an 8ms model-only benchmark.

The final production calibration uses the exact deployed statistical scorer and
binds to its source/model/runtime identity. Its corpus provenance and evaluation
scope remain explicit. Mathematical coverage/Type-I interpretations remain
conditional on independent exchangeable calibration observations. Marginal set
coverage does not establish FDR control or safe-delivery purity.

Release requires task-valid labels and evidence that the candidate satisfies the
declared operating budget on the intended traffic. A failed release gate keeps
the existing model and records the failure; it does not authorize deleting hard
cases, relabelling outcomes, changing thresholds after test inspection, or
describing an experiment as a production-certified model.
