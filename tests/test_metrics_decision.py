import unittest
import numpy as np
from helpers import *
from src.evaluation.metrics import ece, confusion, classification_metrics, threshold_for_cost, group_bootstrap_ci, reliability
from src.decision.costs import CostModel, ALLOW, BLOCK, INSPECT
from src.decision import policies as P
from sklearn.metrics import roc_auc_score

class TestMetrics(unittest.TestCase):
    def test_confusion_and_metrics(self):
        y = np.array([1, 1, 0, 0, 1]); p = np.array([.9, .2, .8, .1, .6])
        self.assertEqual(confusion(y, p >= .5), (2, 1, 1, 1))
        m = classification_metrics(y, p); self.assertAlmostEqual(m["precision"], 2 / 3); self.assertAlmostEqual(m["recall"], 2 / 3); self.assertAlmostEqual(m["accuracy"], 3 / 5)
    def test_ece_perfect_and_worst(self):
        self.assertAlmostEqual(ece(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])), 0.0)
        self.assertAlmostEqual(ece(np.array([0, 0, 1, 1]), np.array([1.0, 1.0, 0.0, 0.0])), 1.0)
    def test_cost_threshold_is_optimal(self):
        r = np.random.default_rng(0); p = r.random(400); y = (r.random(400) < p).astype(int); t = threshold_for_cost(y, p, 10, 1)
        cost = lambda th: 10 * ((p < th) & (y == 1)).sum() + 1 * ((p >= th) & (y == 0)).sum()
        best = min(cost(th) for th in np.r_[p, 2.0]); self.assertEqual(cost(t), best)
    def test_group_bootstrap_ci_contains_point(self):
        r = np.random.default_rng(1); y = r.integers(0, 2, 600); p = np.clip(y * .3 + r.random(600) * .7, 0, 1); g = r.integers(0, 60, 600)
        lo, hi, pt = group_bootstrap_ci(y, p, g, roc_auc_score, n_boot=50, seed=0); self.assertLessEqual(lo, pt); self.assertGreaterEqual(hi, pt)

class TestDecision(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0); self.y = (r.random(2000) < .3).astype(int); self.p = np.clip(np.where(self.y == 1, r.beta(4, 2, 2000), r.beta(1, 5, 2000)), 0, 1); self.cm = CostModel(10, 1, .5)
    def test_cost_model(self):
        cm = self.cm; self.assertEqual(cm.decision_cost(ALLOW, 1), 10); self.assertEqual(cm.decision_cost(ALLOW, 0), 0); self.assertEqual(cm.decision_cost(BLOCK, 0), 1)
        self.assertEqual(cm.decision_cost(INSPECT, 1), .5); self.assertAlmostEqual(cm.bayes_threshold(), 1 / 11)
    def test_fcfs_capacity_enforced(self):
        K = 30; r = P.bayes_fcfs(self.p, self.y, self.cm, K); self.assertLessEqual(r["inspections"], K); self.assertEqual(r["capacity_violations"], 0)
        u = P.bayes_unconstrained(self.p, self.y, self.cm, K); self.assertGreater(u["inspections"], K); self.assertEqual(u["capacity_violations"], u["inspections"] - K)
    def test_batch_topk_respects_capacity_and_beats_fcfs(self):
        K = 30; b = P.bayes_batch_topk(self.p, self.y, self.cm, K); f = P.bayes_fcfs(self.p, self.y, self.cm, K)
        self.assertLessEqual(b["inspections"], K); self.assertLessEqual(b["cost"], f["cost"] + 1e-9)
    def test_cost_accounting_by_hand(self):
        y = np.array([1, 0, 1, 0]); p = np.array([.9, .8, .1, .05]); r = P.fixed_threshold(p, y, self.cm, 1, .5)
        self.assertEqual((r["fn"], r["fp"], r["inspections"]), (1, 1, 0)); self.assertEqual(r["cost"], 10 + 1)
    def test_uncertainty_band(self):
        r = P.uncertainty_inspection(self.p, self.y, self.cm, 25, .2, .8); self.assertLessEqual(r["inspections"], 25)
    def test_bandit_one_step(self):
        b = P.ContextualBandit(self.cm, seed=0).fit(self.p, self.y); r = b.evaluate(self.p, self.y, 40); self.assertLessEqual(r["inspections"], 40)
if __name__ == "__main__": unittest.main()
