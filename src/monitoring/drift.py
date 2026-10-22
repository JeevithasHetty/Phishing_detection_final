"""Drift / performance monitoring.
LIMITATION (by construction): input-only detectors (feature PSI, KS, score PSI) are blind to *concept drift* where
P(x) is unchanged but P(y|x) changes. Only label feedback (error rate, calibration, cost) can reveal it; the
shift experiment demonstrates this explicitly."""
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp
from ..evaluation.metrics import ece

def psi(ref, cur, n_bins=10, eps=1e-4):
    """Population Stability Index with reference-quantile bins. <0.1 stable, 0.1-0.25 moderate, >0.25 major (rule of thumb)."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    if len(edges) == 0: return 0.0
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))

def feature_drift(X_ref, X_cur, names=None, psi_alert=0.25):
    scores = np.array([psi(X_ref[:, j], X_cur[:, j]) for j in range(X_ref.shape[1])])
    ks = np.array([ks_2samp(X_ref[:, j], X_cur[:, j]).statistic for j in range(X_ref.shape[1])])
    top = np.argsort(-scores)[:5]
    return {"max_psi": float(scores.max()), "mean_psi": float(scores.mean()), "n_features_psi_gt_alert": int((scores > psi_alert).sum()),
            "mean_ks": float(ks.mean()), "top_drift_features": [(names[j] if names else int(j), float(scores[j])) for j in top]}

def monitor_window(p_ref, p_cur, y_cur=None, thr=0.5, cm=None, X_ref=None, X_cur=None, names=None, ece_alert=0.05):
    """One monitoring record (matches the supplied drift_monitoring_template schema in spirit)."""
    rec = {"n": int(len(p_cur)), "pred_phish_rate": float((np.asarray(p_cur) >= thr).mean()), "mean_score": float(np.mean(p_cur)),
           "score_psi": psi(p_ref, p_cur)}
    if X_ref is not None: rec["feature_drift"] = feature_drift(X_ref, X_cur, names)
    rec["input_drift_alert"] = bool(rec["score_psi"] > 0.25 or (X_ref is not None and rec["feature_drift"]["max_psi"] > 0.25))
    if y_cur is not None:
        y = np.asarray(y_cur).astype(int); pred = (np.asarray(p_cur) >= thr).astype(int)
        rec.update(prevalence=float(y.mean()), error_rate=float((pred != y).mean()), ece=ece(y, p_cur),
                   fn=int(((pred == 0) & (y == 1)).sum()), fp=int(((pred == 1) & (y == 0)).sum()))
        if cm is not None: rec["cost_per_1k"] = float(1000 * (cm.fn * rec["fn"] + cm.fp * rec["fp"]) / len(y))
        rec["label_alert"] = bool(rec["ece"] > ece_alert)
    return rec
