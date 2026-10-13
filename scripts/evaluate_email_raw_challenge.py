"""2024 original MIME positive-source challenge. Leaves the 2025 mailbox unscored.
There is no benign class, so neither specificity nor AUC is identifiable here.
"""
import hashlib
import json
import mailbox
import joblib
import numpy as np
import pandas as pd
from scripts.run_email_retraining_experiment import ROOT, OUT, sha
from services import scan_engine as engine
from services.security_pipeline import security_pipeline_id
from services.conformal_evidential_engine import ConformalPredictor
from utils.email_parser import parse_eml_content
from utils.text_sanitizer import clean_and_mask_email_text


def main():
    source = ROOT / 'data/external_email_validation/nazario_2024_2025/phishing-2024.mbox'
    acquisition = json.loads((source.parent / 'manifest.json').read_text())
    item = next(r for r in acquisition['files'] if r['path'].endswith('phishing-2024.mbox'))
    assert sha(source) == item['sha256']
    old = set(pd.read_csv(ROOT / 'data/clean_email_corpus.csv', keep_default_na=False).raw_text.map(clean_and_mask_email_text))
    seen, records = set(), []
    box = mailbox.mbox(source, create=False)
    for key in box.iterkeys():
        raw = box.get_bytes(key, from_=False)
        message = parse_eml_content(raw)
        text = (message.get('body') or '') + ' ' + (message.get('subject') or '')
        masked = clean_and_mask_email_text(text)
        reasons = ['exact_old_corpus_overlap'] if masked in old else ['exact_masked_challenge_repeat'] if masked in seen else []
        seen.add(masked)
        records.append({'source_key': int(key), 'raw_sha256': hashlib.sha256(raw).hexdigest(), 'model_text': text,
                        'parsing': message.get('parsing') or {}, 'exclusions': reasons, 'source_reported_label': 'phishing'})
    engine.load_ml_model()
    artifact = joblib.load(OUT / 'candidate_email_model.pkl')
    original_model, original_kind = engine.ML_MODEL, engine.EMAIL_INPUT_KIND
    model_ids = {'baseline': security_pipeline_id(), 'candidate': 'research-email-candidate:' + sha(OUT / 'candidate_email_model.pkl')}
    result = {'source_sha256': sha(source), 'source_year': 2024, 'raw_rows': len(records),
              'unique_no_exact_overlap_rows': sum(not r['exclusions'] for r in records),
              'label_authority': 'Collector phishing-source label; not independent project adjudication',
              'limitations': ['Positive source only: FPR, ROC-AUC, precision and deployment coverage cannot be estimated',
                              'Exact deduplication is not verified campaign independence', 'Content-model challenge, not full sensor policy'],
              'year_2025_original_mailbox_scored': False, 'models': {}}
    for name, model, kind in [('baseline', original_model, original_kind), ('candidate', artifact['model'], 'raw_email_model_text')]:
        engine.ML_MODEL, engine.EMAIL_INPUT_KIND = model, kind
        if name == 'candidate':
            predictor = ConformalPredictor(parameter_path=OUT / 'candidate_conformal_params.json')
        else:
            predictor = ConformalPredictor()
        selected = [r for r in records if not r['exclusions']]
        texts = [r['model_text'] for r in selected]
        support = engine.email_vocabulary_supported(texts, model=model if name == 'candidate' else None,
                                                    vectorizer=engine.TFIDF_VECTORIZER if name == 'baseline' else None)
        support &= np.array([len(clean_and_mask_email_text(t)) > 5 for t in texts])
        eligible = np.flatnonzero(support)
        probabilities = engine.predict_email_probabilities([texts[i] for i in eligible])
        for r in records:
            r[name] = {'probability': None, 'action': 'WARNING_BANNER', 'model_supported': False}
        for i, p in zip(eligible, probabilities):
            action = predictor.predict_set(float(p), pipeline_id=model_ids[name])[2]
            if selected[i]['parsing'].get('text_complete') is False:
                action = 'WARNING_BANNER'
            selected[i][name] = {'probability': float(p), 'action': action, 'model_supported': True}
        result['models'][name] = {'eligible_unique_rows': len(selected), 'probability_supported_rows': len(eligible),
                'unsupported_rows': len(selected)-len(eligible), 'partial_text_rows': sum(r['parsing'].get('text_complete') is False for r in selected),
                'score_0_5_positive_rows': int((probabilities >= .5).sum()),
                'actions': dict(__import__('collections').Counter(r[name]['action'] for r in selected))}
    engine.ML_MODEL, engine.EMAIL_INPUT_KIND = original_model, original_kind
    for r in records:
        del r['model_text']
    result['complete'] = True
    (OUT / 'raw_challenge_results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'raw_challenge_decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    main()
