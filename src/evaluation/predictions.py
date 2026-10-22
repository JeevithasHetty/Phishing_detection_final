"""Load saved raw predictions, align them with partition labels/groups, and calibrate using VALIDATION only."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from ..common import ROOT, load_config, read_json
from ..models.calibration import Calibrator
from .metrics import ece, threshold_for_cost
from sklearn.metrics import brier_score_loss
from ..data.splits import _u

_CACHE = {}

def load_pool(cfg=None):
    cfg = cfg or load_config()
    if "pool" not in _CACHE:
        proc = ROOT / cfg["processed_dir"]
        _CACHE["pool"] = pd.read_pickle(proc / "pool.pkl"); _CACHE["recent"] = pd.read_pickle(proc / "recent.pkl")
    return _CACHE["pool"], _CACHE["recent"]

def partitions(proto, cfg=None):
    cfg = cfg or load_config(); pool, rec = load_pool(cfg)
    key = f"split_{proto}"
    if key not in _CACHE:
        spf = pd.read_csv(ROOT / cfg["processed_dir"] / f"split_{proto}.csv.gz")
        _CACHE[key] = pool["row_id"].map(dict(zip(spf.row_id, spf.split))).fillna("drop").values
    sp = _CACHE[key]
    pool = pool.assign(pos=np.arange(len(pool)))
    return {p: pool[sp == p].reset_index(drop=True) for p in ["train", "val", "policy", "test"]}, rec.assign(pos=np.arange(len(rec)))

def select_calibrator(p_val, y_val, groups_val, seed=0):
    """Choose among none/platt/isotonic by cross-fitting on two group-halves of the validation set
    (fit on one half, score Brier on the other); then refit the winner on all validation data."""
    g = pd.Series(groups_val); half = (g.map(lambda k: _u(k, seed) < 0.5)).values.astype(bool)
    scores = {}
    for meth in ["none", "platt", "isotonic"]:
        s = []
        for a, b in [(half, ~half), (~half, half)]:
            if a.sum() < 50 or b.sum() < 50: continue
            c = Calibrator(meth).fit(p_val[a], y_val[a]); s.append(brier_score_loss(y_val[b], c.transform(p_val[b])))
        scores[meth] = float(np.mean(s)) if s else float("inf")
    best = min(scores, key=scores.get)
    return Calibrator(best).fit(p_val, y_val), best, scores

def load_calibrated(proto, variant, kind, cfg=None, pred_dir="predictions"):
    """Returns dict with raw/calibrated probabilities per partition, labels, groups, chosen calibrator."""
    cfg = cfg or load_config(); parts, rec = partitions(proto, cfg)
    f = np.load(ROOT / "results" / pred_dir / f"{proto}__{variant}__{kind}.npz")
    meta = read_json(ROOT / "results" / pred_dir / f"{proto}__{variant}__{kind}.json")
    out = {"meta": meta, "parts": parts, "recent": rec, "recent_novel": f["recent_novel"]}
    cal, name, scores = select_calibrator(f["p_val"], parts["val"]["label"].values, parts["val"]["rd"].values)
    out["calibrator"], out["calibrator_name"], out["calibrator_scores"] = cal, name, scores
    for p in ["val", "policy", "test", "recent"]:
        raw = f[f"p_{p}"]; out[f"raw_{p}"] = raw; out[f"cal_{p}"] = np.clip(cal.transform(raw), 0, 1)
    return out
