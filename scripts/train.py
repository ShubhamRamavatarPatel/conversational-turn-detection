"""Train on the training split, tune threshold on validation only, evaluate held-out test."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, accuracy_score
from turn_detector.model import ROOT, build_pipeline
from scripts.prepare_data import main as prepare


def read(name):
    return [json.loads(x) for x in (ROOT/'data'/f'{name}.jsonl').read_text().splitlines() if x.strip()]


def score(y, probs, threshold):
    y = np.asarray(y,dtype=int)
    predictions = (np.asarray(probs)>=threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0,1]).ravel()
    return {'n':len(y), 'accuracy':round(float(accuracy_score(y,predictions)),4),
            'precision_complete':round(float(precision),4), 'recall_complete':round(float(recall),4),
            'f1_complete':round(float(f1),4), 'false_early_rate':round(float(fp / (fp+tn)),4),
            'missed_completion_rate':round(float(fn / (fn+tp)),4),
            'confusion':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
            'roc_auc':round(float(roc_auc_score(y,probs)),4)}


def main():
    prepare()
    train, val, test = (read(x) for x in ('train','val','test'))
    model=build_pipeline()
    model.fit([x['text'] for x in train], [x['label'] for x in train])
    val_y = [x['label'] for x in val]
    val_probs = model.predict_proba([x['text'] for x in val])[:,1]
    # Rank high-turn-interruption-risk false positives 1.5x versus false negatives.
    candidates = np.arange(.25, .901, .01)
    best = min(candidates, key=lambda t: (
        1.5*sum((val_probs>=t) & (np.array(val_y)==0)) +
        sum((val_probs<t) & (np.array(val_y)==1))) / len(val_y))
    best=float(round(float(best),2))
    artifact={'schema_version':1,'model_version':'charword-logreg-v1', 'pipeline':model,
              'validation_threshold':best, 'trained_at':datetime.now(timezone.utc).isoformat(),
              'training_examples':len(train), 'class_definition':'1=linguistically complete;0=continuation'}
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT/'artifacts'/'turn_model.joblib', compress=3)
    test_probs = model.predict_proba([x['text'] for x in test])[:,1]
    challenge=read('challenge')
    chall_probs=model.predict_proba([x['text'] for x in challenge])[:,1]
    report={'note':'Synthetic/hand-authored English transcripts; NOT actual production ASR evaluation.',
            'n_train':len(train), 'n_validation':len(val),'n_test':len(test),
            'threshold_tuned_on_validation':best,
            'validation':score(val_y,val_probs,best),
            'heldout_test':score([x['label'] for x in test],test_probs,best),
            'hand_authored_unseen_challenge':score([x['label'] for x in challenge],chall_probs,best)}
    (ROOT/'artifacts'/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    errors=[]
    for x,p in zip(challenge,chall_probs):
        if (p>=best) != bool(x['label']):
            errors.append({'text':x['text'],'expected_complete':bool(x['label']),
                           'p_complete':round(float(p),3),'reason':x.get('reason')})
    (ROOT/'artifacts'/'challenge_errors.json').write_text(json.dumps(errors, indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
