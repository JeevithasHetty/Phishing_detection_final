import unittest
import numpy as np
from helpers import *
from src.models.charcnn import CharCNN, encode, prep, _sigmoid

def _tiny(seed=0, pw=2.0):
    return CharCNN(max_len=14, emb=4, widths=(2, 3), n_filters=3, hidden=4, seed=seed, dtype=np.float64, pos_weight=pw)

class TestCharCNN(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        m = _tiny(); urls = ["http://paypa1-login.xyz/a", "http://example.com/wiki", "http://a.b", "http://secure-bank.top/verify?id=3"]
        X, lens = encode(urls, 14); y = np.array([1, 0, 0, 1], dtype=np.float64)
        _, G = m.loss_and_grads(X, lens, y); rng = np.random.default_rng(0); worst = 0.0; checked = 0
        for k, v in m.P.items():
            for _ in range(6):
                idx = tuple(rng.integers(0, s) for s in v.shape); old = v[idx]; h = 1e-6
                v[idx] = old + h; lp, _ = m.loss_and_grads(X, lens, y); v[idx] = old - h; lm, _ = m.loss_and_grads(X, lens, y); v[idx] = old
                num = (lp - lm) / (2 * h); ana = G[k][idx]
                if abs(num) + abs(ana) < 1e-9: continue
                worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-12)); checked += 1
        self.assertGreater(checked, 20); self.assertLess(worst, 1e-4)

    def test_padding_windows_ignored(self):
        m = _tiny(); a = m.predict_proba(["http://abc.com"]); X, lens = encode(["http://abc.com"], 14)
        self.assertEqual(lens[0], len("abc.com")); self.assertTrue(np.isfinite(a).all())

    def test_learns_a_separable_toy_problem(self):
        r = np.random.default_rng(1); pos = [f"http://secure-login{r.integers(0,999)}.xyz/verify" for _ in range(150)]
        neg = [f"http://www.wiki{r.integers(0,999)}.org/page" for _ in range(150)]; urls = pos + neg; y = np.r_[np.ones(150), np.zeros(150)]
        m = CharCNN(max_len=32, emb=8, widths=(3, 5), n_filters=8, hidden=8, seed=0, dtype=np.float32).fit(urls, y, epochs=8, batch=32, lr=5e-3, log=lambda s: None)
        p = m.predict_proba(urls); self.assertGreater(((p > 0.5) == (y == 1)).mean(), 0.95)

    def test_deterministic_given_seed(self):
        u = ["http://secure-login1.xyz/verify", "http://wiki.org/x"] * 20; y = np.array([1, 0] * 20)
        f = lambda: CharCNN(max_len=24, emb=4, widths=(3,), n_filters=4, hidden=4, seed=3).fit(u, y, epochs=2, batch=8, log=lambda s: None).predict_proba(u[:4])
        np.testing.assert_allclose(f(), f())

    def test_explain_shapes_and_prep(self):
        m = _tiny(); text, s, z = m.explain_chars("http://WWW.Paypa1-login.xyz/a"); self.assertEqual(len(text), len(s)); self.assertEqual(text, prep("http://WWW.Paypa1-login.xyz/a")[:14])
        self.assertEqual(prep("http://WWW.Paypa1-login.xyz/a"), "paypa1-login.xyz/a")

if __name__ == "__main__": unittest.main()
