"""Feedback-gated model adaptation with versioning, rollback and an audit log (OFFLINE SIMULATION; not a deployed system)."""
from __future__ import annotations
import copy, hashlib, json, time
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import roc_auc_score

@dataclass
class GateConfig:
    min_feedback: int = 300            # minimum labelled feedback items
    min_per_class: int = 30
    max_confident_disagreement: float = 0.30   # suspicious-feedback check
    confident_margin: float = 0.05             # "confident" = p<margin or p>1-margin
    min_gain: float = 0.0                      # required improvement of candidate over incumbent on fresh feedback-validation (cost/1k)
    max_reference_regression: float = 0.02     # max allowed drop of reference AUC (trusted clean data)
    max_reference_cost_increase: float = 0.25  # max relative increase in reference cost per 1k
    rollback_cost_margin: float = 0.10         # live-window relative cost increase that triggers rollback

def cost_per_1k(y, p, thr, fn_c, fp_c):
    y = np.asarray(y); pred = np.asarray(p) >= thr
    return 1000 * (fn_c * ((~pred) & (y == 1)).sum() + fp_c * (pred & (y == 0)).sum()) / max(1, len(y))

class ModelRegistry:
    """Versioned model store: commit / rollback with a JSONL-style in-memory audit log (also dumpable to disk)."""
    def __init__(self):
        self.versions = []; self.active = -1; self.log = []
    def commit(self, model, meta):
        self.versions.append({"model": model, "meta": meta}); self.active = len(self.versions) - 1
        self._log("commit", version=self.active, **meta); return self.active
    def rollback(self, reason):
        if self.active <= 0: self._log("rollback_refused", reason=reason); return self.active
        frm = self.active; self.active -= 1; self._log("rollback", from_version=frm, to_version=self.active, reason=reason); return self.active
    @property
    def model(self): return self.versions[self.active]["model"]
    def _log(self, event, **kw): self.log.append({"event": event, **{k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)) for k, v in kw.items()}})

def evaluate_candidate(gate: GateConfig, incumbent, candidate, X_fb, y_fb, X_fbval, y_fbval, X_ref, y_ref, thr, fn_c, fp_c, score):
    """Return (accept: bool, reasons: list, diagnostics: dict). `score(model, X)` -> P(phishing)."""
    reasons, d = [], {}
    n = len(y_fb); n1 = int(np.sum(y_fb)); n0 = n - n1
    d.update(n_feedback=n, n_pos=n1, n_neg=n0)
    if n < gate.min_feedback: reasons.append(f"insufficient_feedback({n}<{gate.min_feedback})")
    if min(n0, n1) < gate.min_per_class: reasons.append(f"insufficient_class_support(min={min(n0, n1)})")
    p_inc = score(incumbent, X_fb)
    conf = (p_inc < gate.confident_margin) | (p_inc > 1 - gate.confident_margin)
    dis = float(np.mean(((p_inc > 0.5).astype(int) != np.asarray(y_fb))[conf])) if conf.any() else 0.0
    d["confident_disagreement"] = dis; d["n_confident"] = int(conf.sum())
    if dis > gate.max_confident_disagreement: reasons.append(f"suspicious_feedback(confident_disagreement={dis:.2f}>{gate.max_confident_disagreement})")
    if reasons: return False, reasons, d   # do not even train-evaluate when data requirements fail
    ci, cc = score(incumbent, X_fbval), score(candidate, X_fbval)
    d["fbval_cost_inc"], d["fbval_cost_cand"] = cost_per_1k(y_fbval, ci, thr, fn_c, fp_c), cost_per_1k(y_fbval, cc, thr, fn_c, fp_c)
    if not (d["fbval_cost_inc"] - d["fbval_cost_cand"] > gate.min_gain): reasons.append("no_improvement_on_fresh_feedback_validation")
    ri, rc = score(incumbent, X_ref), score(candidate, X_ref)
    d["ref_auc_inc"], d["ref_auc_cand"] = float(roc_auc_score(y_ref, ri)), float(roc_auc_score(y_ref, rc))
    d["ref_cost_inc"], d["ref_cost_cand"] = cost_per_1k(y_ref, ri, thr, fn_c, fp_c), cost_per_1k(y_ref, rc, thr, fn_c, fp_c)
    if d["ref_auc_inc"] - d["ref_auc_cand"] > gate.max_reference_regression: reasons.append("reference_auc_regression")
    if d["ref_cost_cand"] > d["ref_cost_inc"] * (1 + gate.max_reference_cost_increase): reasons.append("reference_cost_regression")
    return (len(reasons) == 0), reasons, d
