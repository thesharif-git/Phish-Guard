"""Encoder correctness tests; synthetic examples are not accuracy evidence."""
import numpy as np
import pytest

from training.email_semantic import content_text, masked_mean, token_windows
from utils.email_input_contract import build_email_input
from scripts.run_email_semantic_comparison import evaluate, source_weights


def test_pooling_excludes_padding_and_normalizes():
    hidden = np.asarray([[[3., 0.], [0., 4.], [999., 999.]]])
    vector = masked_mean(hidden, np.asarray([[1, 1, 0]]))[0]
    np.testing.assert_allclose(vector, [.6, .8], atol=1e-6)


@pytest.mark.parametrize('hidden,mask', [
    (np.zeros((1, 2, 3)), np.ones((1, 2))),
    (np.ones((1, 2, 3)), np.zeros((1, 2))),
    (np.full((1, 2, 3), np.nan), np.ones((1, 2))),
    (np.ones((1, 2, 3)), np.ones((1, 3))),
])
def test_invalid_encoder_output_is_unavailable_not_fabricated(hidden, mask):
    with pytest.raises(ValueError):
        masked_mean(hidden, mask)


def test_long_message_windows_preserve_every_content_token_once():
    payload = list(range(10, 1250))
    windows = token_windows([0]+payload+[2],bos=0,eos=2,max_tokens=512)
    assert all(len(window)<=512 and window[0]==0 and window[-1]==2 for window in windows)
    assert [token for window in windows for token in window[1:-1]] == payload


def test_unexpected_special_token_contract_is_rejected():
    with pytest.raises(ValueError):
        token_windows([7,8,9],bos=0,eos=2)


def test_content_representation_uses_observed_text_not_labels_or_source_metadata():
    record = build_email_input('Please inspect the attached invoice.',subject='Payment')
    text = content_text(record)
    assert 'invoice' in text and 'payment' in text
    assert 'phishing' not in text and 'legitimate' not in text
    html = build_email_input('',html_body='<p>مرحبا</p><script>hidden()</script>')
    assert 'مرحبا' in content_text(html) and 'hidden' not in content_text(html)


def test_semantic_representation_binds_current_normalizer_and_keeps_boundaries():
    from training.email_semantic import SEMANTIC_TEXT_VERSION
    from training.email_evidence_model import _text
    from training.email_text_candidates import CandidateEmailFeatures
    from utils.email_candidate_text import EMAIL_CANDIDATE_TEXT_VERSION
    record = build_email_input('wire\x0btransfer', subject='security\x0cnotice')
    assert content_text(record) == 'wire transfer security notice'
    assert _text(record) == content_text(record)
    assert CandidateEmailFeatures(repaired_text=True).canonicalize(['wire\x0btransfer']) == ['wire transfer']
    assert SEMANTIC_TEXT_VERSION.endswith(':' + EMAIL_CANDIDATE_TEXT_VERSION)


def test_train_weights_equalize_source_class_cells_and_reject_missing_classes():
    rows=[dict(source='a',label=0)]*3+[dict(source='a',label=1)]+[
        dict(source='b',label=0),dict(source='b',label=1)]
    weights=source_weights(rows)
    totals=[sum(w for r,w in zip(rows,weights) if (r['source'],r['label'])==(s,y))
            for s in ('a','b') for y in (0,1)]
    np.testing.assert_allclose(totals,[1.5]*4)
    with pytest.raises(ValueError):
        source_weights([dict(source='a',label=0)])


def test_quality_exclusions_and_unavailable_predictions_keep_denominators():
    rows=[dict(label=0,eligible_for_scoring=True),dict(label=1,eligible_for_scoring=True),
          dict(label=1,eligible_for_scoring=False,exclusion_reasons=['unscanned_image'])]
    result=evaluate(rows,[.9,np.nan,np.nan],.95)
    assert result['total_rows']==3 and result['eligible_rows']==2 and result['quality_excluded_rows']==1
    assert result['at_0_5']['scored_rows']==1 and result['at_0_5']['unsupported_by_class']['1']==1
    assert result['quality_exclusion_counts']=={'unscanned_image':1}


def test_baseline_identity_includes_cohort_and_keeps_footer_probe_separate(monkeypatch):
    from scripts import run_email_semantic_comparison as experiment
    original = dict(model='deployed_baseline', split='brown_English_source_test',
                    row_id='message', input_contract_sha256='original', probability=.1)
    probe = dict(original, split='brown_neutral_footer_probe',
                 input_contract_sha256='modified-footer', probability=.9)
    validation = dict(original, split='val', input_contract_sha256='validation', probability=.3)
    temporal = dict(original, input_contract_sha256='temporal', probability=.4)
    monkeypatch.setattr(experiment, 'read_jsonl', lambda path:
        [temporal] if 'matched_temporal' in str(path) else [original, probe, validation])
    scores = experiment.baseline_lookup()['deployed_baseline']
    assert scores[('external_english','message')]['input_contract_sha256']=='original'
    assert scores[('external_english','message')]['probability']==.1
    assert scores[('val','message')]['probability']==.3
    assert scores[('published_day_test','message')]['probability']==.4
    assert len(scores)==3


def test_duplicate_baseline_identity_fails_instead_of_silently_overwriting(monkeypatch):
    from scripts import run_email_semantic_comparison as experiment
    record=dict(model='deployed_baseline',split='brown_English_source_test',row_id='message')
    monkeypatch.setattr(experiment,'read_jsonl',lambda path:
        [] if 'matched_temporal' in str(path) else [record,dict(record)])
    with pytest.raises(ValueError,match='Duplicate baseline'):
        experiment.baseline_lookup()
