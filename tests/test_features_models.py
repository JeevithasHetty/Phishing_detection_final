import unittest, io, joblib
import numpy as np
from helpers import *
from src.features.lexical import featurize, FEATURE_NAMES, strip_www_url, url_features
from src.models.zoo import UrlModel
from src.models.calibration import Calibrator
from src.models.stacking import oof_predictions, Stacker
from sklearn.model_selection import GroupKFold

class TestFeatures(unittest.TestCase):
    def test_shape_and_determinism(self):
        u = ["http://www.a.com/x?y=1", "http://1.2.3.4/login.php"]; X = featurize(u); self.assertEqual(X.shape, (2, len(FEATURE_NAMES)))
        self.assertTrue((X == featurize(u)).all())
    def test_ip_flag_and_keywords(self):
        f = dict(zip(FEATURE_NAMES, url_features("http://1.2.3.4/login/verify.php")))
        self.assertEqual(f["is_ip"], 1); self.assertEqual(f["kw_login"], 1); self.assertEqual(f["ext_php"], 1)
    def test_strip_www(self):
        self.assertEqual(strip_www_url("http://www.a.com/x"), "http://a.com/x"); self.assertEqual(strip_www_url("http://awww.com"), "http://awww.com")
        self.assertTrue((featurize(["http://www.a.com/x"], True) == featurize(["http://a.com/x"], True)).all())

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = synth_urls(1500, 3); cls.tr, cls.te = d.iloc[:1000], d.iloc[1000:]
    def test_train_only_fit(self):
        m = UrlModel("lr_lex", seed=0).fit(self.tr.url.values, self.tr.label.values)
        expect = np.log1p(np.abs(featurize(self.tr.url.values, True))).mean(0)
        self.assertTrue(np.allclose(m.sc.mean_, expect, atol=1e-4))       # scaler statistics come from TRAIN only
        mt = UrlModel("tfidf_lr", seed=0).fit(self.tr.url.values, self.tr.label.values)
        self.assertTrue(len(mt.tf.vocabulary_) > 0)
        before = dict(mt.tf.vocabulary_); mt.predict_proba(self.te.url.values); self.assertEqual(before, mt.tf.vocabulary_)  # predict never refits
    def test_all_kinds_learn_and_roundtrip(self):
        for k in ["lr_lex", "rf_lex", "hgb_lex", "tfidf_lr", "hybrid_lr"]:
            m = UrlModel(k, seed=0).fit(self.tr.url.values, self.tr.label.values); p = m.predict_proba(self.te.url.values)
            self.assertTrue(((p >= 0) & (p <= 1)).all()); acc = ((p > .5) == self.te.label.values).mean(); self.assertGreater(acc, 0.9, k)
            buf = io.BytesIO(); joblib.dump(m, buf); buf.seek(0); m2 = joblib.load(buf)
            self.assertTrue(np.allclose(p, m2.predict_proba(self.te.url.values)), k)
    def test_calibration_fit_on_val_only(self):
        r = np.random.default_rng(0); p = r.random(3000); y = (r.random(3000) < p ** 2).astype(int)   # miscalibrated scores
        from src.evaluation.metrics import ece
        for meth in ["platt", "isotonic"]:
            c = Calibrator(meth).fit(p[:2000], y[:2000]); q = c.transform(p[2000:])
            self.assertTrue(((q >= 0) & (q <= 1)).all()); self.assertLess(ece(y[2000:], q), ece(y[2000:], p[2000:]))
    def test_stacking_oof_never_sees_own_group(self):
        d = self.tr; groups = np.arange(len(d)) % 50; lex = featurize(d.url.values, True)
        # verify the fold generator separates groups
        for a, b in GroupKFold(3).split(d.url.values, d.label.values, groups): self.assertEqual(len(set(groups[a]) & set(groups[b])), 0)
        oof = oof_predictions(["lr_lex", "hgb_lex"], d.url.values, d.label.values, groups, lex, True, 0, n_folds=3)
        self.assertEqual(oof.shape, (len(d), 2)); self.assertTrue(((oof >= 0) & (oof <= 1)).all())
        st = Stacker().fit(oof, d.label.values); self.assertEqual(st.predict_proba(oof[:5]).shape, (5,))
if __name__ == "__main__": unittest.main()
