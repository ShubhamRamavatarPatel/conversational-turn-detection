"""Feature extraction and serialized ML inference: no network or hosted model required."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'artifacts' / 'turn_model.joblib'


def normalize(text: str) -> str:
    """Normalize punctuation to approximate imperfect, unpunctuated ASR output."""
    text = text.lower().replace('’', "'")
    text = re.sub(r"[^\w\s']", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """Hand-picked generalizable transcript end cues; no external grammar parser needed."""
    STOP_END = frozenset('a an the and or but if because since unless although while when where why what which whose who that to for from with of at on in into by as than'.split())
    AUX_END = frozenset("i we you he she they it is are was were have has had would could should can do did does will been being".split())
    DISCOURSE_END = frozenset('so um uh well and but'.split())
    ACKS = frozenset('yes yeah yep no okay ok sure correct absolutely definitely understood thanks thank goodbye fine great perfect right'.split())

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            toks = normalize(text).split()
            last = toks[-1] if toks else ''
            first = toks[0] if toks else ''
            rows.append([
                int(last in self.STOP_END), int(last in self.AUX_END),
                int(last in self.DISCOURSE_END), int(last in self.ACKS),
                int(first in {'and', 'but', 'because', 'if', 'when', 'although', 'before'}),
                int(len(toks) <= 2), int(len(toks) <= 4),
                min(len(toks), 30) / 30.,
                int(last in {'number','address','order','delivery','invoice','meeting','call','payment'}),
            ])
        return csr_matrix(np.array(rows, dtype='float64'))


def build_pipeline() -> Pipeline:
    return Pipeline([
        ('features', FeatureUnion([
            ('words', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=18000)),
            ('chars', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True, min_df=1, max_features=25000)),
            ('linguistic', LinguisticFeatures()),
        ])),
        ('classifier', LogisticRegression(C=1.5, class_weight='balanced', max_iter=1200, random_state=42)),
    ])


@lru_cache(maxsize=2)
def load_artifact(path: str = str(DEFAULT_MODEL)) -> dict:
    artifact = joblib.load(path)
    if artifact.get('schema_version') != 1:
        raise ValueError('Unsupported model artifact')
    return artifact


def probability(text: str, artifact: dict) -> float:
    normalized = normalize(text)
    if not normalized:
        return 0.0
    return float(artifact['pipeline'].predict_proba([normalized])[0, 1])
