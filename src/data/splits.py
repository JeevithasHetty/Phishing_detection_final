"""Leakage-resistant split construction and verification.

Protocols (all on the same de-duplicated pool derived from master_training_candidate.csv):
  random      : stratified row-level split (reference baseline only; NOT an unseen-domain evaluation)
  hostname    : whole hostnames go to exactly one partition
  regdomain   : whole registered domains (eTLD+1, ICANN PSL section) go to exactly one partition  [PRIMARY]
Partitions: train / val (calibration, thresholds, model selection) / policy (RL & bandit training,
adaptation pool; never used to fit classifiers) / test (untouched until protocol is fixed).
"""
from __future__ import annotations
import hashlib
import numpy as np, pandas as pd
from .urls import hostname, registered_domain, normalise_url

PARTS = ["train", "val", "policy", "test"]
DEFAULT_FRACS = {"train": 0.60, "val": 0.10, "policy": 0.10, "test": 0.20}

def build_pool(df: pd.DataFrame):
    """Add host / registered-domain / normalised key; drop normalised duplicates and label-conflicting keys.
    Returns (pool, exclusions_dict). `row_id` = row position in the original file (stable reference)."""
    d = df.copy()
    d["row_id"] = np.arange(len(d))
    d["host"] = d["url"].map(hostname)
    d["rd"] = d["url"].map(registered_domain)
    d["norm"] = d["url"].map(normalise_url)
    ex = {"input_rows": len(d)}
    bad = d["host"] == ""
    ex["empty_host"] = int(bad.sum()); d = d[~bad]
    nl = d.groupby("norm")["label"].nunique()
    conflict = set(nl[nl > 1].index)
    m = d["norm"].isin(conflict)
    ex["label_conflict_rows"] = int(m.sum()); ex["label_conflict_keys"] = len(conflict); d = d[~m]
    dup = d["norm"].duplicated(keep="first")
    ex["normalised_duplicates_dropped"] = int(dup.sum()); d = d[~dup]
    ex["pool_rows"] = len(d)
    return d.reset_index(drop=True), ex

def _u(key: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)

def group_assign(keys: pd.Series, seed: int, fracs=DEFAULT_FRACS) -> np.ndarray:
    """Deterministic hash-based assignment of each group key to a partition (all rows of a key share it)."""
    uniq = pd.Series(keys.unique())
    u = uniq.map(lambda k: _u(k, seed)).values
    edges = np.cumsum([fracs[p] for p in PARTS]); edges[-1] = 1.0 + 1e-9
    part = np.array(PARTS)[np.searchsorted(edges, u, side="right")]
    return keys.map(dict(zip(uniq, part))).values

def make_split(pool: pd.DataFrame, protocol: str, seed: int, fracs=DEFAULT_FRACS) -> np.ndarray:
    if protocol == "regdomain":
        return group_assign(pool["rd"], seed, fracs)
    if protocol == "hostname":
        return group_assign(pool["host"], seed, fracs)
    if protocol == "random":
        rng = np.random.default_rng(seed); out = np.empty(len(pool), dtype=object)
        for lab in (0, 1):
            idx = np.flatnonzero(pool["label"].values == lab); rng.shuffle(idx)
            cuts = (np.cumsum([fracs[p] for p in PARTS]) * len(idx)).astype(int); cuts[-1] = len(idx)
            lo = 0
            for p, hi in zip(PARTS, cuts):
                out[idx[lo:hi]] = p; lo = hi
        return out
    raise ValueError(protocol)

def source_heldout(pool: pd.DataFrame, train_src: str, test_src: str, seed: int):
    """Train on rows whose `sources` == train_src, test on rows whose `sources` == test_src.
    Registered domains that occur in the test source are removed from train/val/policy (rd-disjoint).
    Rows with sources containing both ids are excluded. Returns split array (object, 'drop' for excluded)."""
    out = np.array(["drop"] * len(pool), dtype=object)
    s = pool["sources"].values
    tr_mask, te_mask = s == train_src, s == test_src
    out[te_mask] = "test"
    te_rd = set(pool.loc[te_mask, "rd"])
    cand = tr_mask & ~pool["rd"].isin(te_rd).values
    sub = pool.loc[cand]
    a = group_assign(sub["rd"], seed, {"train": 0.75, "val": 0.125, "policy": 0.125, "test": 0.0})
    out[np.flatnonzero(cand)] = a
    return out

def verify_split(pool: pd.DataFrame, split: np.ndarray, level: str) -> dict:
    """Overlap counts between every pair of partitions at url / hostname / regdomain level."""
    res = {"counts": {}, "overlap": {}}
    key = {"url": "norm", "hostname": "host", "regdomain": "rd"}
    for p in PARTS:
        m = split == p
        res["counts"][p] = {"rows": int(m.sum()), "phishing": int(pool.loc[m, "label"].sum()),
                            "phish_pct": round(100 * float(pool.loc[m, "label"].mean()), 2) if m.any() else None}
    for i, a in enumerate(PARTS):
        for b in PARTS[i + 1:]:
            ma, mb = split == a, split == b
            res["overlap"][f"{a}~{b}"] = {k: len(set(pool.loc[ma, c]) & set(pool.loc[mb, c])) for k, c in key.items()}
    res["disjoint_at_level"] = (all(v[level] == 0 for v in res["overlap"].values()) if level in key else None)
    return res

def split_checksum(row_ids: np.ndarray, split: np.ndarray) -> str:
    h = hashlib.sha256()
    for p in PARTS:
        h.update(p.encode()); h.update(np.sort(row_ids[split == p]).astype("int64").tobytes())
    return h.hexdigest()
