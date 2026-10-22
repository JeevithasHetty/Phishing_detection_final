"""Model definitions. Every transformation (scaler, TF-IDF) is fitted on TRAINING rows only (inside fit())."""
from __future__ import annotations
import time
import numpy as np, scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ..features.lexical import featurize, strip_www_url

def _prep_text(urls, strip_www):
    u = (strip_www_url(x) for x in urls) if strip_www else urls
    return [x.strip().lower() for x in u]

class UrlModel:
    """kind in {lr_lex, rf_lex, hgb_lex, tfidf_lr, hybrid_lr}. fit/predict_proba on raw URL strings."""
    def __init__(self, kind, strip_www=True, seed=42, tree_cap=120000, max_features=300000, C=1.0):
        self.kind, self.strip_www, self.seed, self.tree_cap, self.max_features, self.C = kind, strip_www, seed, tree_cap, max_features, C

    # --- feature builders (fit only on train) ---
    def _lex(self, urls): return featurize(urls, self.strip_www)

    def fit(self, urls, y, lex=None):
        urls = list(urls); y = np.asarray(y); rng = np.random.default_rng(self.seed)
        t0 = time.time()
        k = self.kind
        if k in ("lr_lex", "rf_lex", "hgb_lex", "hybrid_lr"):
            Xl = lex if lex is not None else self._lex(urls)
        if k in ("rf_lex", "hgb_lex"):
            idx = np.arange(len(y))
            if len(y) > self.tree_cap:
                idx = rng.choice(len(y), self.tree_cap, replace=False)
            Xs, ys = Xl[idx], y[idx]
            if k == "rf_lex":
                self.m = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=2, random_state=self.seed,
                                                class_weight=None, max_features="sqrt").fit(Xs, ys)
            else:
                self.m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, max_leaf_nodes=63,
                                                        random_state=self.seed, early_stopping=False).fit(Xs, ys)
        elif k == "lr_lex":
            self.sc = StandardScaler().fit(np.log1p(np.abs(Xl)))
            self.m = LogisticRegression(C=self.C, max_iter=300, random_state=self.seed).fit(self.sc.transform(np.log1p(np.abs(Xl))), y)
        elif k in ("tfidf_lr", "hybrid_lr"):
            self.tf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=self.max_features, lowercase=False,
                                      sublinear_tf=True, dtype=np.float32)
            Xt = self.tf.fit_transform(_prep_text(urls, self.strip_www))
            if k == "hybrid_lr":
                self.sc = StandardScaler().fit(np.log1p(np.abs(Xl)))
                Xt = sp.hstack([Xt, sp.csr_matrix(self.sc.transform(np.log1p(np.abs(Xl))).astype(np.float32) * 0.3)]).tocsr()
            self.m = LogisticRegression(C=self.C, max_iter=200, solver="lbfgs", random_state=self.seed).fit(Xt, y)
        else:
            raise ValueError(k)
        self.fit_seconds = time.time() - t0
        return self

    def _X(self, urls, lex=None):
        k = self.kind
        if k in ("rf_lex", "hgb_lex"): return lex if lex is not None else self._lex(urls)
        if k == "lr_lex":
            Xl = lex if lex is not None else self._lex(urls); return self.sc.transform(np.log1p(np.abs(Xl)))
        Xt = self.tf.transform(_prep_text(urls, self.strip_www))
        if k == "hybrid_lr":
            Xl = lex if lex is not None else self._lex(urls)
            Xt = sp.hstack([Xt, sp.csr_matrix(self.sc.transform(np.log1p(np.abs(Xl))).astype(np.float32) * 0.3)]).tocsr()
        return Xt

    def predict_proba(self, urls, lex=None):
        urls = list(urls)
        return self.m.predict_proba(self._X(urls, lex))[:, 1]

    def top_coefficients(self, k=20):
        """Interpretable weights for linear models (association, not causation)."""
        if self.kind == "tfidf_lr":
            names = np.array(self.tf.get_feature_names_out()); w = self.m.coef_[0]
            o = np.argsort(w); return {"phish": [(names[i], float(w[i])) for i in o[::-1][:k]], "legit": [(names[i], float(w[i])) for i in o[:k]]}
        return None
