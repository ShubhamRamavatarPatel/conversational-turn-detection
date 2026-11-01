"""Create weakly labeled examples and split by source utterance, never by example.

Class 1 = utterance is linguistically complete. Class 0 = continuation expected.
Partial negatives are weak labels: a truncated phrase is sometimes a valid turn.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from turn_detector.model import normalize, ROOT


def read_lines(path: Path):
    return [normalize(x) for x in path.read_text(encoding='utf8').splitlines()
            if x.strip() and not x.lstrip().startswith('#')]


def prefixes(text):
    tokens = text.split()
    if len(tokens) < 7:
        return []
    picks = [max(3, int(len(tokens) * .40)), max(4, int(len(tokens) * .70))]
    # Leave >=2 words for meaningful continuation; do not label near-full as partial.
    return list(dict.fromkeys(' '.join(tokens[:min(p, len(tokens)-2)]) for p in picks))


def build(seed=42):
    full = read_lines(ROOT / 'data/source_utterances.txt')
    partial = read_lines(ROOT / 'data/explicit_partial.txt')
    if len(set(full)) != len(full):
        raise ValueError('Duplicate complete phrases: fix before training')
    groups = [(f'full_{i}', t, 1) for i,t in enumerate(full)] + [(f'partial_{i}',t,0) for i,t in enumerate(partial)]
    all_ids = np.arange(len(groups))
    train_idx, other_idx = train_test_split(all_ids, test_size=.30, random_state=seed, shuffle=True,
                                            stratify=[x[2] for x in groups])
    val_idx, test_idx = train_test_split(other_idx, test_size=.5, random_state=seed,
                                         stratify=[groups[i][2] for i in other_idx])
    result = {}
    for name, idxs in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        examples=[]
        for i in idxs:
            group, transcript, label = groups[i]
            examples.append({'text':transcript, 'label':label, 'group':group, 'source':'human_seed'})
            if label==1:
                for prefix in prefixes(transcript):
                    examples.append({'text':prefix, 'label':0, 'group':group, 'source':'generated_prefix'})
        result[name] = examples
    return result


def main():
    splits = build()
    for name, examples in splits.items():
        path = ROOT / 'data' / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(x) + '\n' for x in examples), encoding='utf8')
        print(name, len(examples), 'positives', sum(e['label'] for e in examples),
              'negatives', sum(not e['label'] for e in examples))


if __name__=='__main__':
    main()
