"""Latency-aware two-stage cascade analysis: a fast URL-only stage decides confident cases and DEFERS the uncertain band to a slower
enrichment stage (e.g. DNS/certificate/page checks). The slower stage is NOT implemented or evaluated here (the supplied data has
no DNS, certificate or DOM fields, and the project makes no network requests); we only quantify how many items would be deferred
and what mean latency results for ASSUMED stage-2 latencies. Hypothetical recalls are labelled as such."""
from __future__ import annotations
import numpy as np

def precision_threshold(y, p, min_precision=0.99):
    """Smallest threshold t such that precision of {p >= t} >= min_precision (on the given calibration data); 1.0+ if none."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float); o = np.argsort(-p); ys, ps = y[o], p[o]
    tp = np.cumsum(ys); prec = tp / np.arange(1, len(ys) + 1); ok = np.flatnonzero(prec >= min_precision)
    if len(ok) == 0: return 1.0 + 1e-9
    k = ok.max(); return float(ps[k])

def cascade_outcome(y, p, t_allow, t_block, lat1_ms, lat2_ms):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float); assert t_allow <= t_block
    allow, block = p < t_allow, p >= t_block; defer = ~allow & ~block; P, N = max(1, (y == 1).sum()), max(1, (y == 0).sum())
    tp_b, tp_d, fn_a = int(((y == 1) & block).sum()), int(((y == 1) & defer).sum()), int(((y == 1) & allow).sum())
    return {"frac_auto_allow": float(allow.mean()), "frac_auto_block": float(block.mean()), "frac_deferred": float(defer.mean()),
            "phish_in_auto_allow": fn_a, "recall_if_deferred_allowed": tp_b / P, "recall_if_deferred_blocked": (tp_b + tp_d) / P,
            "hypothetical_recall_if_stage2_perfect": 1 - fn_a / P, "fp_in_auto_block": int(((y == 0) & block).sum()) / N,
            "mean_latency_ms": float(lat1_ms + defer.mean() * lat2_ms), "lat1_ms": lat1_ms, "lat2_ms_assumed": lat2_ms}
