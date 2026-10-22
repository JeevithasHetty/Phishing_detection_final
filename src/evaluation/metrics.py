"""Classification, calibration and cost metrics + group bootstrap."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def ece(y, p, n_bins=15):
    """Expected calibration error, equal-width bins over [0,1] (documented scheme), weighted by bin size."""
    y, p = np.asarray(y), np.asarray(p)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def reliability(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return [(float(p[b == k].mean()), float(y[b == k].mean()), int((b == k).sum())) for k in range(n_bins) if (b == k).any()]

def confusion(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    return tp, fp, fn, tn

def classification_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int); tp, fp, fn, tn = confusion(y, pred)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    two = len(np.unique(y)) == 2
    return {"n": int(len(y)), "threshold": float(thr), "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-12, prec + rec),
            "roc_auc": float(roc_auc_score(y, p)) if two else None, "pr_auc": float(average_precision_score(y, p)) if two else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}

def group_bootstrap_ci(y, p, groups, fn, n_boot=200, seed=0, alpha=0.05):
    """Cluster (group) bootstrap: resample *groups* with replacement; percentile CI of fn(y,p) over resamples.
    Resampling unit = registered domain (or whatever `groups` holds). Returns (lo, hi, point)."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable"); starts = np.r_[0, np.cumsum(np.bincount(inv))]
    members = [order[starts[i]:starts[i + 1]] for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        try: vals.append(fn(y[idx], p[idx]))
        except Exception: pass
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))), float(fn(y, p))

def threshold_for_cost(y, p, c_fn, c_fp):
    """Validation-set threshold minimising c_fn*FN + c_fp*FP (binary block/allow)."""
    y, p = np.asarray(y), np.asarray(p)
    o = np.argsort(-p); ys = y[o]; ps = p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    cost = c_fn * (P - tp) + c_fp * fp           # block top-k
    cost = np.r_[c_fn * P, cost]                  # k = 0 (block nothing)
    k = int(np.argmin(cost))
    if k == 0: return 1.0 + 1e-9
    if k == len(ps): return float(ps[-1])
    return float((ps[k - 1] + ps[k]) / 2)

def threshold_for_f1(y, p):
    """Validation threshold maximising F1 (never computed on test labels)."""
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(1e-12, pr[:-1] + rc[:-1])
    return float(th[int(np.argmax(f1))])
