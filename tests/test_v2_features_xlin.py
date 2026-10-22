import unittest
import numpy as np, pandas as pd
from helpers import *
from src.features.typosquat import skeleton, BrandIndex, levenshtein, TYPO_FEATURE_NAMES
from src.models.xlin import XLin

def _brand_fixture():
    rd = ["paypal.com"] * 40 + ["microsoft.com"] * 40 + ["amazon.com"] * 30 + ["randomsite.xyz"] * 30 + ["wikipedia.org"] * 20
    y = [0] * 40 + [0] * 40 + [0] * 30 + [1] * 30 + [0] * 20
    return rd, y

class TestSkeleton(unittest.TestCase):
    def test_homoglyphs_and_digits(self):
        self.assertEqual(skeleton("paypa1"), skeleton("paypal")); self.assertEqual(skeleton("rnicrosoft"), skeleton("microsoft"))
        self.assertEqual(skeleton("pаypal"), skeleton("paypal"))            # Cyrillic 'а'
        self.assertEqual(skeleton("pay-pal"), skeleton("paypal")); self.assertNotEqual(skeleton("paypal"), skeleton("amazon"))
    def test_levenshtein(self):
        self.assertEqual(levenshtein("kitten", "sitting", cap=9), 3); self.assertEqual(levenshtein("abc", "abc"), 0); self.assertEqual(levenshtein("abcdef", "uvwxyz"), 3)

class TestBrandIndex(unittest.TestCase):
    def setUp(self): self.bi = BrandIndex(top_k=10, min_len=5, min_urls=5).fit(*_brand_fixture())
    def test_reference_set_is_train_derived_and_excludes_phishing_domains(self):
        self.assertIn("paypal.com", self.bi.rd_set); self.assertNotIn("randomsite.xyz", self.bi.rd_set)
    def test_features(self):
        f = lambda u: dict(zip(TYPO_FEATURE_NAMES, self.bi.features_one(u)))
        self.assertEqual(f("http://paypal.com/login")["is_brand_exact"], 1); self.assertEqual(f("http://paypal.com/login")["edit_le1"], 0)
        sq = f("http://paypa1.com/login"); self.assertEqual(sq["is_brand_exact"], 0); self.assertEqual(sq["skeleton_collision"], 1); self.assertEqual(sq["confusable_digit_in_sld"], 1)
        self.assertEqual(f("http://paypall.com/x")["edit_le1"], 1)
        self.assertEqual(f("http://login.paypal.evil.xyz/a")["brand_in_subdomain"], 1)
        self.assertEqual(f("http://evil.xyz/amazon/signin")["brand_in_path"], 1); self.assertEqual(f("http://pay-pal.com/")["hyphen_squat"], 1)
        self.assertEqual(f("http://example.com/")["edit_le2"], 0)
    def test_no_label_leak_from_other_splits(self):
        bi2 = BrandIndex(top_k=10, min_len=5, min_urls=5).fit(["a-new-brand.com"] * 10, [0] * 10); self.assertNotIn("paypal.com", bi2.rd_set)

class TestXLin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = synth_urls(1500, 5); cls.urls, cls.y = d.url.values, d.label.values
        cls.m = XLin(pos_weight=2.0, max_features=5000, host_features=2000, brand_top_k=20).fit(cls.urls[:1000], cls.y[:1000])
    def test_explanation_is_exact_decomposition(self):
        for u in self.urls[1000:1010]:
            e = self.m.explain(u); z = float(self.m.decision_function([u])[0]); self.assertAlmostEqual(e["logit"], z, places=4)
    def test_fits_and_ranks(self):
        from sklearn.metrics import roc_auc_score
        self.assertGreater(roc_auc_score(self.y[1000:], self.m.predict_proba(self.urls[1000:])), 0.9)
    def test_pos_weight_raises_scores(self):
        m1 = XLin(pos_weight=1.0, max_features=5000, host_features=2000, brand_top_k=20).fit(self.urls[:1000], self.y[:1000])
        self.assertGreaterEqual(self.m.predict_proba(self.urls[1000:]).mean(), m1.predict_proba(self.urls[1000:]).mean())
    def test_char_attribution_shape(self):
        t, s = self.m.char_attribution(self.urls[1000]); self.assertEqual(len(t), len(s))
    def test_pickle_roundtrip(self):
        import pickle; m2 = pickle.loads(pickle.dumps(self.m)); np.testing.assert_allclose(m2.predict_proba(self.urls[1000:1005]), self.m.predict_proba(self.urls[1000:1005]))

if __name__ == "__main__": unittest.main()
