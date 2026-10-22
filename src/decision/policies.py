"""Static (non-sequential) decision policies evaluated on a fixed stream of cases with inspection capacity K.
All policies see only the calibrated probability p (and arrival order); labels are used only for scoring.
Inspection is assumed instantaneous and perfect here (the sequential environment relaxes this)."""
from __future__ import annotations
import numpy as np
from .costs import CostModel, ALLOW, BLOCK, INSPECT

def _score(actions, y, cm: CostModel, K):
    y = np.asarray(y).astype(int); a = np.asarray(actions)
    insp = a == INSPECT
    fn = int(((a == ALLOW) & (y == 1)).sum()); fp = int(((a == BLOCK) & (y == 0)).sum())
    n_insp = int(insp.sum())
    cost = cm.fn * fn + cm.fp * fp + cm.inspect * n_insp
    return {"cost": float(cost), "cost_per_1k": float(1000 * cost / len(y)), "fn": fn, "fp": fp, "inspections": n_insp,
            "capacity_K": int(K), "capacity_violations": int(max(0, n_insp - K)), "n": int(len(y))}

def enforce_capacity_fcfs(requested_inspect, p, thr, K):
    """First-come-first-served: the first K inspection requests are honoured; the rest fall back to threshold decision.
    Returns (actions, n_overridden_requests)."""
    req = np.asarray(requested_inspect, bool)
    rank = np.cumsum(req)
    grant = req & (rank <= K)
    a = (np.asarray(p) >= thr).astype(int)
    a[grant] = INSPECT
    return a, int((req & ~grant).sum())

def fixed_threshold(p, y, cm, K, thr=0.5):
    a = (np.asarray(p) >= thr).astype(int); r = _score(a, y, cm, K); r["policy"] = f"fixed_{thr}"; return r

def tuned_threshold(p, y, cm, K, thr):
    a = (np.asarray(p) >= thr).astype(int); r = _score(a, y, cm, K); r["policy"] = "val_tuned_threshold"; r["threshold"] = float(thr); return r

def bayes_unconstrained(p, y, cm, K):
    """Min expected cost over {ALLOW,BLOCK,INSPECT}. Ignores capacity -> reports violations (K is only used for scoring)."""
    p = np.asarray(p); e_a, e_b = cm.fn * p, cm.fp * (1 - p)
    a = np.where(cm.inspect < np.minimum(e_a, e_b), INSPECT, np.where(e_b < e_a, BLOCK, ALLOW))
    r = _score(a, y, cm, K); r["policy"] = "bayes_expected_cost_unconstrained"; return r

def bayes_fcfs(p, y, cm, K):
    p = np.asarray(p); e_a, e_b = cm.fn * p, cm.fp * (1 - p)
    req = cm.inspect < np.minimum(e_a, e_b)
    a, over = enforce_capacity_fcfs(req, p, cm.bayes_threshold(), K)
    r = _score(a, y, cm, K); r["policy"] = "bayes_capacity_fcfs"; r["requests_denied"] = over; return r

def bayes_batch_topk(p, y, cm, K):
    """Capacity-constrained BATCH policy: requires the whole queue to be visible; inspect the K cases with the
    largest expected saving min(fn p, fp(1-p)) - inspect (only if positive). Non-causal wrt arrival order."""
    p = np.asarray(p); sav = np.minimum(cm.fn * p, cm.fp * (1 - p)) - cm.inspect
    order = np.argsort(-sav)[:K]; order = order[sav[order] > 0]
    a = (p >= cm.bayes_threshold()).astype(int); a[order] = INSPECT
    r = _score(a, y, cm, K); r["policy"] = "bayes_batch_topK(needs full queue)"; return r

def uncertainty_inspection(p, y, cm, K, lo, hi):
    p = np.asarray(p); req = (p >= lo) & (p <= hi)
    a, over = enforce_capacity_fcfs(req, p, cm.bayes_threshold(), K)
    r = _score(a, y, cm, K); r["policy"] = f"uncertainty_band[{lo:.2f},{hi:.2f}]_fcfs"; r["requests_denied"] = over; return r

def tune_uncertainty_band(p, y, cm, K, grid=None):
    """Grid-search the band on a *tuning* stream (validation/policy split), never on test."""
    best = None
    grid = grid or [(l, h) for l in (0.02, 0.05, 0.1, 0.2, 0.3) for h in (0.5, 0.6, 0.7, 0.8, 0.9)]
    for lo, hi in grid:
        c = uncertainty_inspection(p, y, cm, K, lo, hi)["cost"]
        if best is None or c < best[0]: best = (c, lo, hi)
    return best[1], best[2]

def evaluate_all_static(p, y, cm, K, thr_tuned, band, extra=None):
    out = [fixed_threshold(p, y, cm, K, 0.5), tuned_threshold(p, y, cm, K, thr_tuned),
           fixed_threshold(p, y, cm, K, cm.bayes_threshold()),
           bayes_unconstrained(p, y, cm, K), bayes_fcfs(p, y, cm, K), bayes_batch_topk(p, y, cm, K),
           uncertainty_inspection(p, y, cm, K, *band)]
    if extra:
        out += extra
    return out

class ContextualBandit:
    """One-step contextual bandit (tabular, epsilon-greedy): context = calibrated-probability bin; reward = -immediate cost.
    No state transitions, no notion of remaining capacity: capacity is enforced externally (FCFS). This is deliberately
    NOT sequential RL."""
    EDGES = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97])
    def __init__(self, cm: CostModel, seed=0, eps=0.1):
        self.cm, self.eps, self.rng = cm, eps, np.random.default_rng(seed)
        self.n = np.zeros((11, 3)); self.mean = np.zeros((11, 3))
    def fit(self, p, y, passes=1):
        b = np.searchsorted(self.EDGES, p, side="right")
        for _ in range(passes):
            for i in self.rng.permutation(len(p)):
                a = int(self.rng.integers(0, 3)) if self.rng.random() < self.eps else int(np.argmax(self.mean[b[i]]))
                r = -(self.cm.decision_cost(a, int(y[i])))
                self.n[b[i], a] += 1; self.mean[b[i], a] += (r - self.mean[b[i], a]) / self.n[b[i], a]
        return self
    def request_inspect(self, p):
        b = np.searchsorted(self.EDGES, p, side="right"); return np.argmax(self.mean, axis=1)[b] == INSPECT
    def evaluate(self, p, y, K):
        p = np.asarray(p); b = np.searchsorted(self.EDGES, p, side="right"); greedy = np.argmax(self.mean, axis=1)[b]
        a, over = enforce_capacity_fcfs(greedy == INSPECT, p, self.cm.bayes_threshold(), K)
        a = np.where(greedy != INSPECT, greedy, a)  # own ALLOW/BLOCK choice; for inspection requests: granted -> INSPECT, denied -> threshold fallback
        r = _score(a, y, self.cm, K); r["policy"] = "contextual_bandit_fcfs"; r["requests_denied"] = over; return r
