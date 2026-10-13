"""Reproduce frozen candidate errors and their exact linear score terms.

Contributions explain a fitted score, not the actual cause of an attack.
This does not change labels, thresholds, features, weights or deployment.
"""
import argparse
import json
from pathlib import Path
import sys
import gc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scipy.special import expit
from scripts.run_published_email_experiment import assert_frozen, digest, save, save_jsonl
from scripts.train_email_evidence_candidate import validate_output
from training.email_evidence_model import NUMERIC_NAMES
from utils.jsonl import read_jsonl


def feature_names(features):
    names = ['word:'+name for name in features.word_vectorizer_.get_feature_names_out()]
    if features.character_features:
        names += ['character:'+name for name in features.character_vectorizer_.get_feature_names_out()]
    if features.evidence_features:
        if features.identity_vectorizer_ is not None:
            names += ['identity:'+name for name in features.identity_vectorizer_.get_feature_names_out()]
        names += ['observed_fact:'+name for name in NUMERIC_NAMES]
    return np.asarray(names)


def score_terms(model, record, expected_probability):
    features = model.named_steps['features']
    if not features.supported([record])[0]:
        if expected_probability is not None:
            raise ValueError('Frozen score incorrectly exists for unsupported input')
        return dict(state='UNAVAILABLE_REVIEW_REQUIRED')
    x = features.transform([record]).tocsr()
    classifier = model.named_steps['classifier']
    names = feature_names(features)
    coefficients = classifier.coef_[0]
    if len(names) != len(coefficients) or x.shape[1] != len(names):
        raise ValueError('Linear feature names differ from actual columns')
    terms = x.data * coefficients[x.indices]
    intercept = float(classifier.intercept_[0])
    logit = float(terms.sum()+intercept)
    probability = float(expit(logit))
    if expected_probability is None or not np.isclose(probability, expected_probability, atol=1e-12, rtol=1e-12):
        raise ValueError('Frozen per-message probability does not reproduce')
    channels = {}
    for name,term in zip(names[x.indices],terms):
        channel = name.split(':',1)[0]
        channels[channel] = channels.get(channel,0.)+float(term)
    def strongest(positive):
        order = np.argsort(-terms if positive else terms)
        return [dict(feature=str(names[x.indices[index]]),
                     observed_value=float(x.data[index]), coefficient=float(coefficients[x.indices[index]]),
                     log_odds_contribution=float(terms[index]))
                for index in order if (terms[index]>0 if positive else terms[index]<0)][:12]
    return dict(state='SCORED', probability=probability, intercept=intercept,
                log_odds=logit, channel_log_odds_terms=channels,
                toward_phishing=strongest(True), toward_legitimate=strongest(False))


def run(experiment, output):
    directory, output = Path(experiment).resolve(), validate_output(output)
    protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    results = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    if results.get('complete') is not True or results['protocol_sha256'] != digest(directory/'protocol.json'):
        raise ValueError('Research experiment is incomplete or unbound')
    assert_frozen(directory,protocol)
    rows = read_jsonl(protocol['Brown_records_path'])
    rows = [row for row in rows if row['eligible_for_scoring']]
    saved = {}
    for prediction in read_jsonl(directory/'predictions.jsonl'):
        if prediction['split'] != 'brown_English_source_test':
            continue
        key=(prediction['model'],prediction['row_id'])
        if key in saved:
            raise ValueError('Duplicate frozen Brown probability')
        saved[key]=prediction
    output.mkdir(parents=True,exist_ok=False)
    cases, summary = [], {}
    for name, metadata in results['models'].items():
        if metadata.get('fitted_in_this_run') is False:
            continue
        model_path=directory/(name+'.pkl')
        if digest(model_path)!=metadata['model_sha256']:
            raise ValueError('Candidate artifact changed after freeze')
        model=joblib.load(model_path)['model']
        errors={'false_positive':0,'false_negative':0,'unavailable':0,'reviewed_rows':len(rows)}
        for row in rows:
            prediction=saved[(name,row['id'])]
            if prediction['input_contract_sha256']!=row['input_contract_sha256'] or prediction['label']!=row['label']:
                raise ValueError('Error review input or source label differs')
            probability=prediction['probability']
            terms=score_terms(model,row['email_input'],probability)
            kind='unavailable' if probability is None else (
                'false_positive' if row['label']==0 and probability>=.5 else
                'false_negative' if row['label']==1 and probability<.5 else 'correct')
            if kind!='correct':errors[kind]+=1
            cases.append(dict(model=name,row_id=row['id'],source_label=row['task_label'],
                label_authority=row['label_authority_kind'],original_reference=row['original_reference'],
                input_contract_sha256=row['input_contract_sha256'],
                sender=row['email_input']['sender'],subject=row['email_input']['subject'],
                body_excerpt=row['email_input']['body'][:800],error_at_0_5=kind,score_terms=terms,
                comment='Exact fitted log-odds terms, not causal proof of phishing or verified identity'))
        summary[name]=errors
        del model
        gc.collect()
    assert_frozen(directory,protocol)
    save_jsonl(output/'error_review.jsonl',cases)
    save(output/'verification.json',dict(complete=True,all_candidate_eligible_probabilities_reproduced=True,
        no_labels_changed=True,no_models_changed=True,no_thresholds_changed=True,
        experiment_protocol_sha256=digest(directory/'protocol.json'),
        source_predictions_sha256=digest(directory/'predictions.jsonl'),
        review_code_sha256=digest(__file__),summary=summary,
        limitations='Explaining coefficients establishes computation, not causal security semantics; small published excerpts do not establish inbox reliability'))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.experiment,args.output)
