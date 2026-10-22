import unittest
import numpy as np
from helpers import *
from src.rl.env import PhishInspectionEnv, Scenario, make_episode, run_episode, encode_state, N_STATES
from src.rl.agents import QAgent, bellman_update, pol_fixed, pol_bayes, pol_random, pol_pacing
from src.decision.costs import CostModel, ALLOW, BLOCK, INSPECT

def toy_pool(n=3000, seed=0):
    r = np.random.default_rng(seed); y = (r.random(n) < .3).astype(int); p = np.clip(np.where(y == 1, r.beta(5, 1.5, n), r.beta(1, 6, n)), 0, 1); return p, y

class TestBellman(unittest.TestCase):
    def test_nonterminal_update(self):
        Q = np.zeros((3, 3)); Q[1] = [1.0, 5.0, 2.0]; Q[0, 1] = 2.0
        bellman_update(Q, 0, 1, -1.0, 1, False, alpha=0.5, gamma=0.9); self.assertAlmostEqual(Q[0, 1], 2.0 + 0.5 * (-1 + 0.9 * 5.0 - 2.0))
    def test_terminal_does_not_bootstrap(self):
        Q = np.zeros((3, 3)); Q[1] = [100, 100, 100]; bellman_update(Q, 0, 2, -3.0, 1, True, alpha=1.0, gamma=0.99); self.assertAlmostEqual(Q[0, 2], -3.0)
    def test_valid_action_masking_in_target(self):
        Q = np.zeros((2, 3)); Q[1] = [0, 0, 50]; bellman_update(Q, 0, 0, 0.0, 1, False, 1.0, 1.0, valid2=(ALLOW, BLOCK)); self.assertEqual(Q[0, 0], 0.0)
    def test_gamma_zero_is_bandit(self):
        Q = np.zeros((2, 3)); Q[1] = [9, 9, 9]; bellman_update(Q, 0, 0, -2.0, 1, False, 1.0, 0.0); self.assertEqual(Q[0, 0], -2.0)

class TestEnv(unittest.TestCase):
    def setUp(self): self.p, self.y = toy_pool(); self.cm = CostModel(10, 1, .5, .01)
    def env(self, T=50, K=3): return PhishInspectionEnv(self.cm, T, K, .3)
    def ep(self, scen=Scenario(), T=50, seed=0): return make_episode(self.p, self.y, scen, T, seed)
    def test_common_random_numbers(self):
        a, b = self.ep(Scenario("bursty", 5, .1), seed=3), self.ep(Scenario("bursty", 5, .1), seed=3)
        for k in ["p", "y", "delay", "flip"]: self.assertTrue((a[k] == b[k]).all())
    def test_capacity_enforced_and_budget_transition(self):
        env = self.env(T=20, K=3); env.reset(self.ep(T=20)); 
        for i in range(3): env.step(INSPECT); self.assertEqual(env.B, 2 - i)
        self.assertEqual(env.valid_actions(), (ALLOW, BLOCK)); self.assertEqual(env.m["violations"], 0)
        env.step(INSPECT); self.assertEqual(env.m["violations"], 1); self.assertEqual(env.m["inspections"], 3)  # overridden, not executed
    def test_delayed_feedback_arrives_after_delay(self):
        sc = Scenario("stationary", 5, 0.0); ep = self.ep(sc, T=40); ep["delay"][:] = 4; ep["y"][0] = 1
        env = self.env(40, 5); env.reset(ep); c0 = env.c_conf; env.step(INSPECT)
        self.assertEqual(env.n_pending, 1)
        for _ in range(2): env.step(ALLOW)
        self.assertEqual(env.n_pending, 1); self.assertEqual(env.c_conf, c0)          # not yet resolved
        env.step(ALLOW); env.step(ALLOW)                                                # t=4 -> resolved
        self.assertEqual(env.n_pending, 0); self.assertGreater(env.c_conf, c0)          # analyst said phishing -> EMA up
    def test_pending_flushed_at_episode_end_and_charged(self):
        ep = self.ep(Scenario("stationary", 50, 1.0), T=10); ep["delay"][:] = 30; ep["flip"][:] = 1; ep["y"][0] = 1
        env = self.env(10, 3); env.reset(ep); done = False; first = True
        while not done:
            _, r, done = env.step(INSPECT if first else ALLOW); first = False
        self.assertEqual(env.n_pending, 0); self.assertEqual(env.m["resolved"], 1); self.assertGreaterEqual(env.m["cost"], self.cm.fn)  # noisy analyst missed phishing
    def test_reward_equals_negative_cost(self):
        env = self.env(); ep = self.ep(Scenario("bursty", 3, .2)); m = run_episode(env, ep, pol_bayes(self.cm, True))
        self.assertAlmostEqual(-m["reward"], m["cost"])
        # independent recomputation of cost from counters: fn*FN + fp*FP + inspect*n + hold
        self.assertAlmostEqual(m["cost"], self.cm.fn * m["fn"] + self.cm.fp * m["fp"] + self.cm.inspect * m["inspections"] + m["hold_cost"], places=6)
    def test_no_label_in_observation(self):
        env = self.env(); ep = self.ep(); obs = env.reset(ep); self.assertEqual(len(obs), 6)
        self.assertNotIn(float(ep["y"][0]), [obs[1], obs[2], obs[3]]) if ep["y"][0] == 1 else None
        self.assertTrue(all(isinstance(o, (int, float)) for o in obs))
    def test_state_encoding_range(self):
        for obs in [(0.0, 0, 0, 0, .1, .1), (1.0, 50, 10, 9, .9, .9), (.5, 3, 900, 1, .3, .5)]:
            s = encode_state(obs, 1000, 50); self.assertTrue(0 <= s < N_STATES)
    def test_terminal_transition_returns_done(self):
        env = self.env(T=3, K=1); env.reset(self.ep(T=3)); d = [env.step(ALLOW)[2] for _ in range(3)]; self.assertEqual(d, [False, False, True])
    def test_q_learning_beats_random_and_learns_budget_use(self):
        p, y = self.p, self.y; cm = self.cm; T, K = 200, 10; env = PhishInspectionEnv(cm, T, K, .3); sc = Scenario("stationary", 0, 0)
        ag = QAgent("full", 0.97, 0.1, seed=0); ag.train(env, lambda e: make_episode(p, y, sc, T, 100 + e), 150)
        ev = [make_episode(p, y, sc, T, 9000 + i) for i in range(20)]
        rl = np.mean([run_episode(env, e, ag.policy(T, K))["cost"] for e in ev]); rnd = np.mean([run_episode(env, e, pol_random(0))["cost"] for e in ev])
        self.assertLess(rl, rnd); self.assertTrue(all(run_episode(env, e, ag.policy(T, K))["violations"] == 0 for e in ev[:5]))   # masking => no violations
    def test_baselines_run_and_unmasked_bayes_violates(self):
        env = self.env(T=300, K=5); ep = self.ep(T=300); v = run_episode(env, ep, pol_bayes(self.cm, False))["violations"]; self.assertGreater(v, 0)
        self.assertEqual(run_episode(env, ep, pol_bayes(self.cm, True))["violations"], 0)
        self.assertEqual(run_episode(env, ep, pol_pacing(self.cm, self.p, 300, 5))["violations"], 0)
if __name__ == "__main__": unittest.main()
