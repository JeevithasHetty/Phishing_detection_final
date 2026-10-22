"""Recall-first operating points with a finite-sample guarantee, plus recall-oriented metrics.

Guarantee (split-conformal / Learn-then-Test style, order-statistic form). Let s_(1) <= ... <= s_(n) be the scores of n labelled
phishing calibration items, assumed exchangeable with a future phishing item (continuous scores). Flag an item as phishing iff
score >= s_(k). The miss probability m = P(S < s_(k)) is distributed Beta(k, n-k+1), and P(m <= alpha) = P(Bin(n, alpha) >= k).
Hence choosing the LARGEST k with P(Bin(n, alpha) <= k-1) <= delta gives
        P_calibration-draw( recall >= 1 - alpha ) >= 1 - delta .
The guarantee concerns the data-generating distribution of the calibration positives. It does NOT hold under distribution shift
between calibration and deployment, and URLs of one registered domain are not independent; we therefore calibrate on ONE random
positive per registered domain (`one_per_group`) and report both domain-level and URL-level recall at test time.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import binom
from sklearn.metrics import roc_auc_score

def ltt_threshold(scores_pos, alpha: float, delta: float):
    """Return (threshold, k, n). threshold=-inf (flag everything) if n is too small for the requested (alpha, delta)."""
    s = np.sort(np.asarray(scores_pos, dtype=float)); n = len(s)
    if n == 0: return -np.inf, 0, 0
    ks = np.arange(1, n + 1); ok = binom.cdf(ks - 1, n, alpha) <= delta
    if not ok.any(): return -np.inf, 0, n
    k = int(ks[ok].max()); return float(s[k - 1]), k, n

def naive_threshold(scores_pos, alpha: float) -> float:
    """Empirical alpha-quantile of calibration positives (no finite-sample correction)."""
    return float(np.quantile(np.asarray(scores_pos, dtype=float), alpha, method="lower"))

def min_n_for(alpha: float, delta: float) -> int:
    """Smallest number of calibration positives for which a finite threshold exists: (1-alpha)^n <= delta."""
    return int(np.ceil(np.log(delta) / np.log(1 - alpha)))

def one_per_group(scores, groups, rng):
    """One uniformly random item per group (approximates independence across registered domains)."""
    scores, groups = np.asarray(scores), np.asarray(groups)
    order = rng.permutation(len(scores)); _, first = np.unique(groups[order], return_index=True)
    return scores[order][first]

def url_recall(scores_pos, t): return float((np.asarray(scores_pos) >= t).mean())

def domain_recall(scores_pos, groups_pos, t):
    """Expected recall when one phishing URL is drawn uniformly at random from each phishing domain (= mean of per-domain hit rates)."""
    hit = (np.asarray(scores_pos) >= t).astype(float)
    import pandas as pd
    return float(pd.Series(hit).groupby(np.asarray(groups_pos)).mean().mean())

def fpr_at(y, p, t):
    y, p = np.asarray(y).astype(int), np.asarray(p); neg = y == 0
    return float((p[neg] >= t).mean())

def recall_at_fpr(y, p, max_fpr):
    """Largest recall achievable with FPR <= max_fpr, and the threshold used."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float); neg = np.sort(p[y == 0])
    if len(neg) == 0: return 1.0, -np.inf
    t = np.nextafter(np.quantile(neg, 1 - max_fpr, method="higher"), np.inf)
    return float((p[y == 1] >= t).mean()), float(t)

def fpr_at_recall(y, p, min_recall):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float); pos = p[y == 1]
    t = float(np.quantile(pos, 1 - min_recall, method="lower")); return fpr_at(y, p, t), t

def partial_auc(y, p, max_fpr=0.05):
    """McClish-standardised partial AUC over FPR in [0, max_fpr] (0.5 = chance, 1 = perfect)."""
    return float(roc_auc_score(y, p, max_fpr=max_fpr))
