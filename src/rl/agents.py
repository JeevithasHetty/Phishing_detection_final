"""Tabular Q-learning (sequential RL) and the comparison policies for the sequential environment."""
from __future__ import annotations
import numpy as np
from ..decision.costs import CostModel, ALLOW, BLOCK, INSPECT
from .env import PhishInspectionEnv, encode_state, run_episode, make_episode, N_STATES, N_STATES_BUDGET_ONLY

def bellman_update(Q, s, a, r, s2, done, alpha, gamma, valid2=None):
    """Q(s,a) <- Q(s,a) + alpha*(r + gamma*max_{a' valid} Q(s',a') - Q(s,a)); terminal states do NOT bootstrap."""
    if done: target = r
    else:
        q2 = Q[s2] if valid2 is None else Q[s2][list(valid2)]
        target = r + gamma * float(np.max(q2))
    Q[s, a] += alpha * (target - Q[s, a])
    return Q[s, a]

class QAgent:
    """gamma>0: sequential RL.  gamma=0: contextual bandit on the same state (one-step reward)."""
    def __init__(self, mode="full", gamma=0.97, alpha=0.1, seed=0, alpha_mode="const", alpha_min=0.01, alpha_n0=20.0):
        self.mode, self.gamma, self.alpha = mode, gamma, alpha
        self.alpha_mode, self.alpha_min, self.alpha_n0 = alpha_mode, alpha_min, alpha_n0
        n = N_STATES if mode == "full" else N_STATES_BUDGET_ONLY
        self.Q = np.zeros((n, 3)); self.rng = np.random.default_rng(seed); self.visits = np.zeros((n, 3), int)

    def act(self, s, valid, eps):
        if self.rng.random() < eps: return valid[int(self.rng.integers(0, len(valid)))]
        q = self.Q[s]; best = max(valid, key=lambda a: q[a]); return best

    def train(self, env: PhishInspectionEnv, make_ep, n_episodes, eps_start=1.0, eps_end=0.05, log_every=20):
        curve = []; T, K = env.T, env.K; cost_acc = []
        for e in range(n_episodes):
            eps = max(eps_end, eps_start - (eps_start - eps_end) * e / max(1, 0.7 * n_episodes))
            obs = env.reset(make_ep(e)); s = encode_state(obs, T, K, self.mode); done = False
            while not done:
                valid = env.valid_actions(); a = self.act(s, valid, eps)
                obs2, r, done = env.step(a)
                s2 = None if done else encode_state(obs2, T, K, self.mode)
                v2 = None if done else env.valid_actions()
                al = self.alpha if self.alpha_mode == "const" else max(self.alpha_min, self.alpha / (1.0 + self.visits[s, a] / self.alpha_n0))
                self.visits[s, a] += 1
                bellman_update(self.Q, s, a, r, s2, done, al, self.gamma, v2)
                s = s2
            cost_acc.append(env.m["cost"])
            if (e + 1) % log_every == 0: curve.append((e + 1, float(np.mean(cost_acc[-log_every:])), eps)); 
        return curve

    def policy(self, T, K):
        def f(obs, valid):
            return self.act(encode_state(obs, T, K, self.mode), valid, 0.0)
        return f

# ---------------- baselines (all use only the observable tuple obs=(p,B,t,queue,e_arr,c_conf)) ----------------
def pol_fixed(thr):
    return lambda obs, valid: BLOCK if obs[0] >= thr else ALLOW

def pol_bayes(cm: CostModel, masked: bool):
    thr = cm.bayes_threshold()
    def f(obs, valid):
        p = obs[0]
        if cm.inspect < min(cm.fn * p, cm.fp * (1 - p)) and (INSPECT in valid or not masked): return INSPECT
        return BLOCK if p >= thr else ALLOW
    return f

def pol_uncertainty(cm, lo, hi):
    thr = cm.bayes_threshold()
    def f(obs, valid):
        p = obs[0]
        if lo <= p <= hi and INSPECT in valid: return INSPECT
        return BLOCK if p >= thr else ALLOW
    return f

def _lam_table(p_ref, cm, n=201):
    v = np.sort(np.minimum(cm.fn * p_ref, cm.fp * (1 - p_ref)) - cm.inspect)
    rates = np.linspace(0, 1, n)
    lam = np.array([np.quantile(v, 1 - r) if r > 0 else np.inf for r in rates]); return np.maximum(lam, 0.0)

def pol_pacing(cm, p_ref, T, K, prior_adapt=False, prior=0.28):
    """Capacity-aware pacing heuristic: inspect when expected saving exceeds a threshold set so that the remaining
    budget is spent at the uniform rate B/(T-t). prior_adapt additionally corrects p for the running arrival-risk EMA
    (odds-ratio prior-shift correction) -> the strongest hand-built baseline using the same information as RL."""
    lam = _lam_table(p_ref, cm); thr = cm.bayes_threshold()
    def f(obs, valid):
        p, B, t, q, e, c = obs
        if prior_adapt:
            e_ = min(max(e, 1e-3), 1 - 1e-3); pr = min(max(prior, 1e-3), 1 - 1e-3)
            odds = p / max(1e-9, 1 - p) * (e_ / (1 - e_)) / (pr / (1 - pr)); p = odds / (1 + odds)
        if B > 0:
            rate = min(1.0, B / max(1, T - t)); v = min(cm.fn * p, cm.fp * (1 - p)) - cm.inspect
            if v > lam[int(round(rate * 200))] and v > 0: return INSPECT
        return BLOCK if p >= thr else ALLOW
    return f

def pol_random(seed):
    rng = np.random.default_rng(seed)
    return lambda obs, valid: valid[int(rng.integers(0, len(valid)))]
