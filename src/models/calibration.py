"""Post-hoc calibration fitted on VALIDATION data only (never on test)."""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def _logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

class Calibrator:
    def __init__(self, method="platt"): self.method = method
    def fit(self, p, y):
        if self.method == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(np.asarray(p)).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        elif self.method == "none":
            pass
        else: raise ValueError(self.method)
        return self
    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "platt": return self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic": return self.m.predict(p)
        return p
