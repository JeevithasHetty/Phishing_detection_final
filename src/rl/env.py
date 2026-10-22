"""Sequential phishing-inspection environment (SIMULATED; transitions/delays/noise are modelling assumptions,
not observations of any real SOC). Dataset facts used: URL labels and classifier probabilities of real held-out URLs.

Episode: T URL arrivals. At each step the agent sees the arriving URL's calibrated phishing probability p and
aggregate operational state, and chooses ALLOW / BLOCK / INSPECT.
Transitions (what makes this sequential, not a one-step bandit):
  * INSPECT consumes one unit of a finite per-episode capacity budget B (never replenished) -> early use limits later use.
  * The inspected case is quarantined; the analyst label (flipped w.p. `noise`) arrives after a delay d_t; the case
    accrues `hold` cost per step while pending, and any analyst mistake is charged on resolution (delayed reward).
  * Resolved analyst labels update the agent-visible confirmed-phish rate EMA; arrival probabilities update an
    arrival-risk EMA (regime signal). Both feed the next states.
  * Pending cases at the end are flushed (charged) -> cannot dodge costs by inspecting at the end.
Hidden from the agent: true labels, regime, future arrivals, noise/delay draws.
Reward r_t = -(cost incurred between decision t and decision t+1), derived exactly from the CostModel.
Invalid INSPECT (B=0) is overridden by a threshold decision and counted as a capacity violation (policies that
use action masking never trigger it).
"""
from __future__ import annotations
import bisect
from dataclasses import dataclass
import numpy as np
from ..decision.costs import CostModel, ALLOW, BLOCK, INSPECT

P_EDGES = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97]
RB_EDGES = [0.5, 1.0, 2.0]
E_EDGES = [0.2, 0.4]
C_EDGES = [0.3, 0.6]
N_STATES = 11 * 5 * 3 * 3 * 3
N_STATES_BUDGET_ONLY = 11 * 5

@dataclass
class Scenario:
    name: str = "stationary"       # "stationary" | "bursty"
    delay: int = 0                 # mean feedback delay (steps); 0 = immediate
    noise: float = 0.0             # analyst label-flip probability
    pi_low: float = 0.10           # bursty: phishing share in low regime
    pi_high: float = 0.60          # bursty: phishing share in high regime
    switch: float = 0.01           # bursty: per-step regime switch probability

def make_episode(p_pool, y_pool, scen: Scenario, T: int, seed: int, idx_by_label=None):
    """Pre-draw ALL randomness (arrivals, regimes, delays, analyst flips) so every policy sees identical episodes
    (common random numbers => paired comparisons)."""
    rng = np.random.default_rng(seed)
    if scen.name == "stationary":
        idx = rng.integers(0, len(p_pool), T); regime = np.zeros(T, int)
    else:
        i0, i1 = idx_by_label if idx_by_label is not None else (np.flatnonzero(y_pool == 0), np.flatnonzero(y_pool == 1))
        regime = np.zeros(T, int); r = int(rng.integers(0, 2))
        sw = rng.random(T) < scen.switch
        for t in range(T):
            if sw[t]: r = 1 - r
            regime[t] = r
        lab = rng.random(T) < np.where(regime == 1, scen.pi_high, scen.pi_low)
        idx = np.where(lab, i1[rng.integers(0, len(i1), T)], i0[rng.integers(0, len(i0), T)])
    delay = np.zeros(T, int) if scen.delay == 0 else rng.integers(1, 2 * scen.delay, T)
    flip = (rng.random(T) < scen.noise).astype(int)
    return {"p": p_pool[idx].astype(float), "y": y_pool[idx].astype(int), "regime": regime, "delay": delay, "flip": flip, "idx": idx}

class PhishInspectionEnv:
    def __init__(self, cm: CostModel, T: int, K: int, prior: float, ema_arr=0.02, ema_conf=0.1):
        self.cm, self.T, self.K, self.prior, self.ema_arr, self.ema_conf = cm, T, K, prior, ema_arr, ema_conf
        self.thr = cm.bayes_threshold()

    # ---- episode control ----
    def reset(self, ep):
        self.ep = ep; self.p = ep["p"].tolist(); self.y = ep["y"].tolist(); self.dl = ep["delay"].tolist(); self.fl = ep["flip"].tolist()
        self.t = 0; self.B = self.K; self.pending = {}; self.n_pending = 0
        self.e_arr = self.prior; self.c_conf = self.prior
        self.m = dict(cost=0.0, fn=0, fp=0, inspections=0, violations=0, n_allow=0, n_block=0, resolved=0,
                      analyst_errors=0, hold_cost=0.0, reward=0.0)
        return self._obs()

    def _obs(self):
        p = self.p[self.t]
        self.e_arr = (1 - self.ema_arr) * self.e_arr + self.ema_arr * p  # arrival-risk EMA updated on arrival
        return (p, self.B, self.t, self.n_pending, self.e_arr, self.c_conf)

    def valid_actions(self):
        return (ALLOW, BLOCK, INSPECT) if self.B > 0 else (ALLOW, BLOCK)

    def _resolve(self, y, yhat):
        cm, c = self.cm, 0.0
        self.m["resolved"] += 1
        self.c_conf = (1 - self.ema_conf) * self.c_conf + self.ema_conf * yhat
        if y == 1 and yhat == 0: c += cm.fn; self.m["fn"] += 1; self.m["analyst_errors"] += 1
        elif y == 0 and yhat == 1: c += cm.fp; self.m["fp"] += 1; self.m["analyst_errors"] += 1
        return c

    def step(self, a):
        cm, t = self.cm, self.t; y = self.y[t]; p = self.p[t]; cost = 0.0
        if a == INSPECT and self.B <= 0:
            self.m["violations"] += 1; a = BLOCK if p >= self.thr else ALLOW
        if a == ALLOW:
            cost += cm.fn * y; self.m["n_allow"] += 1; self.m["fn"] += y
        elif a == BLOCK:
            cost += cm.fp * (1 - y); self.m["n_block"] += 1; self.m["fp"] += 1 - y
        else:
            self.B -= 1; cost += cm.inspect; self.m["inspections"] += 1
            yhat = y ^ self.fl[t]; d = self.dl[t]
            if d == 0: cost += self._resolve(y, yhat)
            else:
                self.pending.setdefault(t + d, []).append((y, yhat)); self.n_pending += 1
        # advance time
        self.t += 1
        hold = cm.hold * self.n_pending; cost += hold; self.m["hold_cost"] += hold
        done = self.t >= self.T
        if not done:
            for (yy, yh) in self.pending.pop(self.t, []):
                cost += self._resolve(yy, yh); self.n_pending -= 1
        else:  # flush everything still pending at episode end
            for lst in self.pending.values():
                for (yy, yh) in lst: cost += self._resolve(yy, yh)
            self.pending = {}; self.n_pending = 0
        self.m["cost"] += cost; self.m["reward"] -= cost
        obs = None if done else self._obs()
        return obs, -cost, done

def encode_state(obs, T, K, mode="full"):
    p, B, t, q, e, c = obs
    pb = bisect.bisect(P_EDGES, p)
    rb = 0 if B <= 0 else 1 + bisect.bisect(RB_EDGES, (B / max(1, T - t)) / (K / T))
    if mode == "budget_only": return pb * 5 + rb
    eb = bisect.bisect(E_EDGES, e); cb = bisect.bisect(C_EDGES, c); qb = 0 if q == 0 else (1 if q <= 3 else 2)
    return (((pb * 5 + rb) * 3 + eb) * 3 + cb) * 3 + qb

def run_episode(env: PhishInspectionEnv, ep, policy):
    """policy(obs, valid) -> action. Returns metrics dict (cost incl. delayed components)."""
    obs = env.reset(ep); done = False
    while not done:
        obs, r, done = env.step(policy(obs, env.valid_actions()))
    m = dict(env.m); m["T"] = env.T; m["phishing_arrivals"] = int(sum(ep["y"]))
    m["missed_phishing"] = m["fn"]; m["false_blocks"] = m["fp"]
    return m
