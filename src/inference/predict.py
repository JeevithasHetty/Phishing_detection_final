"""Offline single-URL prediction. Never performs network access; the URL string is only parsed/featurised locally.
LIMITS: the model sees only the URL string (no page content, WHOIS, reputation, or time information). Probabilities are
calibrated on the project's validation split and are not guaranteed on other distributions (see distribution-shift report).
The action uses the Bayes expected-cost rule over ALLOW/BLOCK/INSPECT with ASSUMED costs; inspection capacity is NOT
modelled for a single URL (INSPECT means 'would be sent to an analyst if capacity were available')."""
from __future__ import annotations
import joblib
import numpy as np
from ..decision.costs import CostModel, ACTION_NAMES, ALLOW, BLOCK, INSPECT

class Detector:
    def __init__(self, model, calibrator, cm: CostModel, meta: dict):
        self.model, self.calibrator, self.cm, self.meta = model, calibrator, cm, meta

    @classmethod
    def load(cls, path):
        b = joblib.load(path); return cls(b["model"], b["calibrator"], b["cost_model"], b["meta"])

    def save(self, path):
        joblib.dump({"model": self.model, "calibrator": self.calibrator, "cost_model": self.cm, "meta": self.meta}, path, compress=3)

    def score(self, urls):
        raw = self.model.predict_proba(list(urls)); return np.clip(self.calibrator.transform(raw), 0, 1)

    def decide_p(self, p):
        cm = self.cm; e_a, e_b = cm.fn * p, cm.fp * (1 - p)
        if cm.inspect < min(e_a, e_b): return INSPECT
        return BLOCK if e_b < e_a else ALLOW

    def predict(self, urls):
        if isinstance(urls, str): urls = [urls]
        p = self.score(urls)
        return [{"url": u, "predicted_class": "phishing" if pi >= 0.5 else "legitimate", "risk_score": float(pi), "action": ACTION_NAMES[self.decide_p(float(pi))]} for u, pi in zip(urls, p)]
