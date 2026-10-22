"""XLin: an inherently interpretable linear URL model with recall-oriented class weighting.

Feature blocks (all fitted on TRAINING rows only):
  A  char 2-4-gram TF-IDF of the normalised URL (scheme and leading 'www.' removed)
  B  char 2-4-gram TF-IDF of the confusable *skeleton* of the host   (optional: use_canon)
  C  hand-crafted lexical features (log1p, standardised, weight 0.3)
  D  typosquat / brand-impersonation features from a train-derived BrandIndex (optional: use_typo)
Because the model is linear, every prediction decomposes EXACTLY: logit = intercept + sum_j coef_j * x_j. `explain()` returns that
decomposition, so explanations cost one sparse dot product (microseconds) and need no SHAP/LIME sampling.
Contributions are associations learned by the model, not causal effects.
"""
from __future__ import annotations
import time
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ..features.lexical import featurize, FEATURE_NAMES
from ..features.typosquat import BrandIndex, skeleton, TYPO_FEATURE_NAMES
from ..data.urls import hostname, registered_domain, is_ip_host
from .zoo import _prep_text

class XLin:
    def __init__(self, use_canon=True, use_typo=True, pos_weight=2.0, C=1.0, seed=42, max_features=300000, host_features=50000, brand_top_k=2000):
        self.use_canon, self.use_typo, self.pos_weight, self.C, self.seed = use_canon, use_typo, pos_weight, C, seed
        self.max_features, self.host_features, self.brand_top_k = max_features, host_features, brand_top_k

    def _aux(self, urls, lex, rd, host):
        if host is None: host = [hostname(u) for u in urls]
        if rd is None: rd = [h if is_ip_host(h) else registered_domain(h, is_host=True) for h in host]
        if lex is None: lex = featurize(urls, True)
        return lex, rd, host

    def _blocks(self, urls, lex, rd, host, fit=False):
        text = _prep_text(urls, True); B = []
        if fit:
            self.tf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=self.max_features, lowercase=False, sublinear_tf=True, dtype=np.float32)
            B.append(self.tf.fit_transform(text))
        else: B.append(self.tf.transform(text))
        if self.use_canon:
            ht = [skeleton(h[4:] if h.startswith("www.") else h) for h in host]
            if fit:
                self.tfh = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=self.host_features, lowercase=False, sublinear_tf=True, dtype=np.float32)
                B.append(self.tfh.fit_transform(ht))
            else: B.append(self.tfh.transform(ht))
        L = np.log1p(np.abs(lex))
        if fit: self.sc = StandardScaler().fit(L)
        B.append(sp.csr_matrix(self.sc.transform(L).astype(np.float32) * 0.3))
        if self.use_typo:
            T = self.brand.features(urls, rd, host)
            if fit: self.tsc = StandardScaler().fit(T)
            B.append(sp.csr_matrix(self.tsc.transform(T).astype(np.float32) * 0.5))
        return sp.hstack(B).tocsr()

    def fit(self, urls, y, lex=None, rd=None, host=None):
        t0 = time.time(); urls = list(urls); y = np.asarray(y).astype(int)
        lex, rd, host = self._aux(urls, lex, rd, host)
        if self.use_typo: self.brand = BrandIndex(top_k=self.brand_top_k).fit(rd, y)
        X = self._blocks(urls, lex, rd, host, fit=True)
        self.m = LogisticRegression(C=self.C, max_iter=200, solver="lbfgs", random_state=self.seed, class_weight={0: 1.0, 1: float(self.pos_weight)}).fit(X, y)
        self._names()
        self.fit_seconds = time.time() - t0
        return self

    def _names(self):
        n = [("url_ngram", g) for g in self.tf.get_feature_names_out()]
        if self.use_canon: n += [("host_skeleton_ngram", g) for g in self.tfh.get_feature_names_out()]
        n += [("lexical", f) for f in FEATURE_NAMES]
        if self.use_typo: n += [("typosquat", f) for f in TYPO_FEATURE_NAMES]
        self.names = n

    def decision_function(self, urls, lex=None, rd=None, host=None):
        urls = list(urls); lex, rd, host = self._aux(urls, lex, rd, host)
        return self.m.decision_function(self._blocks(urls, lex, rd, host))

    def predict_proba(self, urls, lex=None, rd=None, host=None):
        z = self.decision_function(urls, lex, rd, host); return 1.0 / (1.0 + np.exp(-z))

    # ---------------------------------------------------------------- explanation
    def explain(self, url, k=8):
        """Exact additive decomposition of the logit for one URL. Returns dict with intercept, total, top positive/negative terms."""
        lex, rd, host = self._aux([url], None, None, None)
        X = self._blocks([url], lex, rd, host); coef = self.m.coef_[0]
        idx, val = X.indices, X.data; c = coef[idx] * val
        order = np.argsort(c)
        terms = [(self.names[idx[i]][0], self.names[idx[i]][1], float(val[i]), float(c[i])) for i in order]
        total = float(self.m.intercept_[0] + c.sum())
        return {"intercept": float(self.m.intercept_[0]), "logit": total, "risk": 1 / (1 + np.exp(-total)),
                "toward_phishing": [t for t in terms[::-1][:k] if t[3] > 0], "toward_legitimate": [t for t in terms[:k] if t[3] < 0],
                "n_active_features": int(len(idx))}

    def char_attribution(self, url):
        """Spread url-ngram contributions over the characters they cover (descriptive; lexical/typosquat terms are not char-local)."""
        lex, rd, host = self._aux([url], None, None, None)
        text = _prep_text([url], True)[0]; xa = self.tf.transform([text]); coef = self.m.coef_[0][: xa.shape[1]]
        s = np.zeros(len(text))
        for j, v in zip(xa.indices, xa.data):
            g = self.names[j][1]; pos = [i for i in range(len(text) - len(g) + 1) if text[i:i + len(g)] == g]
            for p in pos: s[p:p + len(g)] += coef[j] * v / (len(g) * len(pos))
        return text, s
