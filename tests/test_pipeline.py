import unittest, subprocess, sys, os, zlib
import numpy as np, pandas as pd
from helpers import *
from src.data import splits as S
from src.models.zoo import UrlModel
from src.models.calibration import Calibrator
from src.evaluation.metrics import classification_metrics, threshold_for_cost
from src.decision.costs import CostModel
from src.decision import policies as P
from src.rl.env import PhishInspectionEnv, Scenario, make_episode, run_episode
from src.rl.agents import QAgent, pol_pacing

class TestEndToEndOnFixture(unittest.TestCase):
    """Whole chain on a SYNTHETIC fixture (software test only): pool -> domain split -> fit -> calibrate on val -> test metrics
    -> static policies -> sequential env + Q-learning. Research results never use this fixture."""
    def test_chain(self):
        d = synth_urls(4000, 3); d["domain"] = d["domain"].astype(str)
        pool, _ = S.build_pool(d)
        pool["rd"] = [f"g{zlib.crc32(u.encode()) % 300}.com" for u in pool["url"]]   # coarse groups so that the split is non-trivial
        sp = S.make_split(pool, "regdomain", 1); self.assertTrue(S.verify_split(pool, sp, "regdomain")["disjoint_at_level"])
        m = {p: sp == p for p in S.PARTS}; tr, va, te, po = (pool[m[p]] for p in ["train", "val", "test", "policy"])
        mod = UrlModel("lr_lex", seed=0).fit(tr.url.values, tr.label.values)
        cal = Calibrator("platt").fit(mod.predict_proba(va.url.values), va.label.values)
        pt = cal.transform(mod.predict_proba(te.url.values)); met = classification_metrics(te.label.values, pt, 0.5)
        self.assertGreater(met["roc_auc"], 0.9); self.assertEqual(met["tp"] + met["fp"] + met["fn"] + met["tn"], len(te))
        cm = CostModel(10, 1, 0.5); thr = threshold_for_cost(va.label.values, cal.transform(mod.predict_proba(va.url.values)), 10, 1)
        K = max(1, int(0.05 * len(te))); res = P.evaluate_all_static(pt, te.label.values, cm, K, thr, (0.05, 0.6))
        self.assertTrue(all(r["n"] == len(te) for r in res)); self.assertTrue(all(r["capacity_violations"] == 0 for r in res if "unconstrained" not in r["policy"]))
        ppol = cal.transform(mod.predict_proba(po.url.values)); ypol = po.label.values.astype(int)
        env = PhishInspectionEnv(cm, 100, 5, float(ypol.mean())); sc = Scenario("bursty", 5, 0.1)
        ag = QAgent("full", 0.97, 0.1, seed=0); curve = ag.train(env, lambda e: make_episode(ppol, ypol, sc, 100, e), 30, log_every=10)
        self.assertEqual(len(curve), 3); self.assertGreater(np.abs(ag.Q).sum(), 0)
        ev = make_episode(pt, te.label.values.astype(int), sc, 100, 999)
        c1 = run_episode(env, ev, ag.policy(100, 5)); c2 = run_episode(env, ev, ag.policy(100, 5))
        self.assertEqual(c1["cost"], c2["cost"]); self.assertLessEqual(c1["inspections"], 5)       # greedy policy is deterministic, budget respected
        self.assertEqual(c1["violations"], 0)

class TestReproducibility(unittest.TestCase):
    def test_group_assignment_order_independent(self):
        keys = pd.Series([f"d{i % 50}.com" for i in range(500)]); a = S.group_assign(keys, 7)
        perm = np.random.default_rng(0).permutation(len(keys)); b = S.group_assign(keys.iloc[perm].reset_index(drop=True), 7)
        self.assertTrue((a[perm] == b).all())
    def test_model_training_deterministic(self):
        d = synth_urls(800, 2); a = UrlModel("hgb_lex", seed=1).fit(d.url.values, d.label.values).predict_proba(d.url.values[:50])
        b = UrlModel("hgb_lex", seed=1).fit(d.url.values, d.label.values).predict_proba(d.url.values[:50]); np.testing.assert_allclose(a, b)

class TestRunAllFailsClearly(unittest.TestCase):
    def test_unknown_stage_rejected(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"), "--from", "no_such_stage"], capture_output=True, text=True, cwd=ROOT)
        self.assertNotEqual(r.returncode, 0); self.assertIn("unknown stage", r.stderr + r.stdout)

@unittest.skipUnless((ROOT / "artifacts" / "deployable_bundle.joblib").exists(), "inference bundle not built yet (run scripts/build_inference_bundle.py)")
class TestPredictCLI(unittest.TestCase):
    def test_cli_returns_class_score_action(self):
        import json
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict_url.py"), "http://example.com/x"], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr); out = json.loads(r.stdout)["predictions"][0]
        self.assertIn(out["predicted_class"], ("phishing", "legitimate")); self.assertIn(out["action"], ("ALLOW", "BLOCK", "INSPECT")); self.assertTrue(0 <= out["risk_score"] <= 1)

if __name__ == "__main__": unittest.main()
