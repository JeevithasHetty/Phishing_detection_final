"""Leakage-free stacking: the meta-model is trained on OUT-OF-FOLD base predictions (group-aware folds)."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from .zoo import UrlModel

def _logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def oof_predictions(kinds, urls, y, groups, lex, strip_www, seed, n_folds=3, **kw):
    """Returns array (n, len(kinds)) of OOF probabilities; each row predicted by models that never saw its group."""
    urls = np.asarray(urls, dtype=object); y = np.asarray(y)
    out = np.zeros((len(y), len(kinds)))
    for tr, te in GroupKFold(n_splits=n_folds).split(urls, y, groups):
        for j, k in enumerate(kinds):
            m = UrlModel(k, strip_www=strip_www, seed=seed, **kw).fit(urls[tr], y[tr], lex=lex[tr])
            out[te, j] = m.predict_proba(urls[te], lex=lex[te])
    return out

class Stacker:
    def __init__(self, C=1.0): self.C = C
    def fit(self, oof, y):
        self.meta = LogisticRegression(C=self.C, max_iter=500).fit(_logit(oof), y); return self
    def predict_proba(self, base_probs):
        return self.meta.predict_proba(_logit(base_probs))[:, 1]
