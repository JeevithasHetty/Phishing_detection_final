import unittest
import numpy as np
from scipy.stats import binom
from helpers import *
from src.decision import recall_control as RC
from src.decision.cascade import precision_threshold, cascade_outcome

class TestRecallControl(unittest.TestCase):
    def test_threshold_rule_matches_beta_binomial_identity(self):
        s = np.linspace(0, 1, 200); alpha, delta = 0.05, 0.1
        t, k, n = RC.ltt_threshold(s, alpha, delta)
        self.assertEqual(n, 200); self.assertGreaterEqual(k, 1); self.assertLessEqual(binom.cdf(k - 1, n, alpha), delta)
        if k < n: self.assertGreater(binom.cdf(k, n, alpha), delta)          # k is the LARGEST admissible index
        self.assertEqual(t, np.sort(s)[k - 1])
    def test_infeasible_when_too_few_calibration_positives(self):
        n_min = RC.min_n_for(0.05, 0.1); self.assertEqual(n_min, int(np.ceil(np.log(0.1) / np.log(0.95))))
        t, k, n = RC.ltt_threshold(np.arange(n_min - 1) / 100.0, 0.05, 0.1); self.assertTrue(np.isneginf(t)); self.assertEqual(k, 0)
        t2, k2, _ = RC.ltt_threshold(np.arange(n_min + 5) / 100.0, 0.05, 0.1); self.assertTrue(np.isfinite(t2))
    def test_guarantee_holds_empirically_for_exchangeable_data(self):
        rng = np.random.default_rng(0); alpha, delta, n = 0.1, 0.1, 150; viol = 0; reps = 1500
        for _ in range(reps):
            cal = rng.beta(5, 2, n); t, _, _ = RC.ltt_threshold(cal, alpha, delta)
            true_recall = 1 - __import__("scipy.stats", fromlist=["beta"]).beta.cdf(t, 5, 2)        # exact recall of threshold t under Beta(5,2)
            viol += true_recall < 1 - alpha
        self.assertLessEqual(viol / reps, delta + 0.03)                                        # empirical violation rate <= delta (+MC slack)
    def test_naive_quantile_violates_more_often(self):
        rng = np.random.default_rng(1); alpha, n, reps = 0.1, 150, 1500; from scipy.stats import beta
        v_naive = np.mean([1 - beta.cdf(RC.naive_threshold(rng.beta(5, 2, n), alpha), 5, 2) < 1 - alpha for _ in range(reps)])
        self.assertGreater(v_naive, 0.3)                                                       # ~50% expected without correction
    def test_one_per_group_and_domain_recall(self):
        sc = np.array([.9, .8, .1, .2, .95]); g = np.array(["a", "a", "b", "b", "c"]); rng = np.random.default_rng(0)
        self.assertEqual(len(RC.one_per_group(sc, g, rng)), 3)
        self.assertAlmostEqual(RC.domain_recall(sc, g, 0.5), (1 + 0 + 1) / 3); self.assertAlmostEqual(RC.url_recall(sc, 0.5), 3 / 5)
    def test_recall_at_fpr_and_fpr_at_recall(self):
        r = np.random.default_rng(2); y = np.r_[np.ones(500), np.zeros(5000)]; p = np.r_[r.beta(5, 2, 500), r.beta(2, 5, 5000)]
        rec, t = RC.recall_at_fpr(y, p, 0.01); self.assertLessEqual(RC.fpr_at(y, p, t), 0.0101)
        f, t2 = RC.fpr_at_recall(y, p, 0.95); self.assertGreaterEqual((p[y == 1] >= t2).mean(), 0.95); self.assertGreater(RC.partial_auc(y, p, 0.05), 0.5)

class TestCascade(unittest.TestCase):
    def test_outcome_accounting(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0]); p = np.array([.95, .6, .1, .05, .3, .55, .9, .02])
        o = cascade_outcome(y, p, 0.2, 0.9, 1.0, 100.0)
        self.assertAlmostEqual(o["frac_auto_allow"] + o["frac_auto_block"] + o["frac_deferred"], 1.0)
        self.assertEqual(o["phish_in_auto_allow"], 1); self.assertAlmostEqual(o["hypothetical_recall_if_stage2_perfect"], 2 / 3)
        self.assertAlmostEqual(o["mean_latency_ms"], 1.0 + o["frac_deferred"] * 100.0); self.assertLessEqual(o["recall_if_deferred_allowed"], o["recall_if_deferred_blocked"])
    def test_precision_threshold(self):
        y = np.array([1, 1, 1, 0, 1, 0]); p = np.array([.99, .98, .97, .96, .5, .1]); t = precision_threshold(y, p, 1.0)
        self.assertAlmostEqual(t, .97)

if __name__ == "__main__": unittest.main()
