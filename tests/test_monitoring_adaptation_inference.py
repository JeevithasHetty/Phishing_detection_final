import unittest, tempfile, os
import numpy as np, scipy.sparse as sp
from helpers import *
from src.monitoring.drift import psi, monitor_window
from src.adaptation.safe_update import GateConfig, ModelRegistry, evaluate_candidate, cost_per_1k
from src.decision.costs import CostModel
from sklearn.linear_model import SGDClassifier

class TestMonitoring(unittest.TestCase):
    def test_psi(self):
        r = np.random.default_rng(0); a = r.normal(size=4000)
        self.assertLess(psi(a, r.normal(size=4000)), 0.05); self.assertGreater(psi(a, r.normal(1.5, 1, 4000)), 0.25)
    def test_input_detector_blind_to_concept_drift_but_label_monitor_sees_it(self):
        r = np.random.default_rng(0); X = r.normal(size=(4000, 3)); y = (X[:, 0] > 0).astype(int); p = np.clip(.5 + .45 * np.sign(X[:, 0]), 0, 1)
        y2 = 1 - y   # concept drift: labels change, inputs identical
        rec = monitor_window(p, p, y2, 0.5, CostModel(), X, X); self.assertFalse(rec["input_drift_alert"]); self.assertTrue(rec["label_alert"]); self.assertGreater(rec["error_rate"], .9)

def lin_data(n, seed, flip=False):
    r = np.random.default_rng(seed); X = r.normal(size=(n, 5)); y = (X[:, 0] + .3 * X[:, 1] > 0).astype(int); return X, (1 - y if flip else y)
score = lambda m, X: m.predict_proba(X)[:, 1]
def fit(X, y): return SGDClassifier(loss="log_loss", random_state=0).fit(X, y)

class TestAdaptation(unittest.TestCase):
    def setUp(self):
        self.Xr, self.yr = lin_data(2000, 0); self.inc = fit(self.Xr, self.yr); self.g = GateConfig(min_feedback=100, min_per_class=10)
    def run_gate(self, X_fb, y_fb, cand, gate=None):
        Xv, yv = lin_data(400, 7, flip=False)
        return evaluate_candidate(gate or self.g, self.inc, cand, X_fb, y_fb, Xv, yv, self.Xr, self.yr, .5, 10, 1, score)
    def test_rejects_insufficient_feedback(self):
        X, y = lin_data(50, 1); ok, why, _ = self.run_gate(X, y, self.inc); self.assertFalse(ok); self.assertTrue(any("insufficient" in w for w in why))
    def test_rejects_poisoned_feedback_via_confident_disagreement(self):
        X, y = lin_data(300, 2, flip=True); cand = fit(X, y); ok, why, d = self.run_gate(X, y, cand); self.assertFalse(ok); self.assertTrue(any("suspicious" in w for w in why))
    def test_rejects_reference_regression(self):
        X, y = lin_data(300, 3); X = X * 1.0; Xf, yf = lin_data(300, 4, flip=True); cand = fit(Xf, yf)   # candidate trained on flipped labels
        g = GateConfig(min_feedback=100, min_per_class=10, max_confident_disagreement=1.0)            # disable suspicion gate to isolate regression gate
        ok, why, d = self.run_gate(X, y, cand, g); self.assertFalse(ok); self.assertTrue(any("regression" in w or "no_improvement" in w for w in why))
    def test_accepts_genuinely_better_candidate(self):
        # incumbent weak (trained on 30 points), candidate trained on much more clean data; feedback-validation on same distribution
        Xs, ys = lin_data(30, 11); inc = fit(Xs, ys); Xf, yf = lin_data(600, 5); cand = fit(Xf, yf)
        Xv, yv = lin_data(600, 8); ok, why, d = evaluate_candidate(self.g, inc, cand, Xf, yf, Xv, yv, self.Xr, self.yr, .5, 10, 1, score)
        self.assertTrue(ok, (why, d))
    def test_registry_commit_rollback_and_audit(self):
        reg = ModelRegistry(); reg.commit("m0", {}); reg.commit("m1", {"round": 0}); self.assertEqual(reg.model, "m1")
        reg.rollback("regression"); self.assertEqual(reg.model, "m0"); self.assertEqual(reg.rollback("again"), 0)  # cannot roll back past v0
        self.assertEqual([e["event"] for e in reg.log], ["commit", "commit", "rollback", "rollback_refused"])

class TestInference(unittest.TestCase):
    def test_bundle_roundtrip_and_consistency(self):
        from src.models.zoo import UrlModel; from src.models.calibration import Calibrator; from src.inference.predict import Detector
        d = synth_urls(1200, 5); m = UrlModel("lr_lex", seed=0).fit(d.url.values[:900], d.label.values[:900]); raw = m.predict_proba(d.url.values[900:])
        cal = Calibrator("platt").fit(raw, d.label.values[900:]); det = Detector(m, cal, CostModel(10, 1, .5), {"t": 1})
        f = os.path.join(tempfile.mkdtemp(), "b.joblib"); det.save(f); det2 = Detector.load(f)
        a, b = det.predict(list(d.url.values[:20])), det2.predict(list(d.url.values[:20])); self.assertEqual(a, b)
        self.assertTrue(all(x["action"] in ("ALLOW", "BLOCK", "INSPECT") and 0 <= x["risk_score"] <= 1 for x in a))
        self.assertEqual(det.predict("http://secure-login-1.xyz/verify/account.php?id=1")[0]["predicted_class"], "phishing")
    def test_decide_rule(self):
        from src.inference.predict import Detector; det = Detector(None, None, CostModel(10, 1, .5), {})
        self.assertEqual(det.decide_p(0.0), 0); self.assertEqual(det.decide_p(1.0), 1); self.assertEqual(det.decide_p(.3), 2)
if __name__ == "__main__": unittest.main()
