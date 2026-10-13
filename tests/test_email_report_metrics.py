"""Undefined/error metrics cannot be converted into fabricated AUC results."""
import numpy as np
import pytest
from scripts.train_production_models import evaluate_metrics


def test_single_class_auc_is_unavailable_not_invented_half():
    result = evaluate_metrics([0, 0], [.1, .8])
    assert result['roc_auc'] is None
    assert result['fnr'] is None
    assert result['tn'] == 1 and result['fp'] == 1


@pytest.mark.parametrize('p', [[float('nan'), .2], [-.1, .5], [.1, 1.1]])
def test_invalid_probabilities_fail_instead_of_becoming_metrics(p):
    with pytest.raises(ValueError, match='finite probabilities'):
        evaluate_metrics([0, 1], p)


def test_confusion_counts_and_error_rates_are_computed_from_predictions():
    result = evaluate_metrics([0, 0, 1, 1], [.2, .9, .1, .8])
    assert result['cm'].tolist() == [[1, 1], [1, 1]]
    assert result['fpr'] == .5 and result['fnr'] == .5


def test_failed_generalization_cannot_be_bypassed_by_legacy_production_calibration(monkeypatch):
    from scripts import calibrate_conformal_engine as calibration
    monkeypatch.setattr(calibration.pd, 'read_csv', lambda *a, **k: pytest.fail('Must reject before dataset access or calibration writes'))
    with pytest.raises(ValueError, match='generalization'):
        calibration.run_calibration_and_evaluation()


def test_development_calibration_cannot_target_production_models_directory():
    from scripts import calibrate_conformal_engine as calibration
    with pytest.raises(ValueError, match='benchmarks'):
        calibration.run_calibration_and_evaluation(development_only=True, output_directory=calibration.ROOT / 'models')
