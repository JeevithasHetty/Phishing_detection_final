"""Character-level CNN for URLs, implemented in pure NumPy (no PyTorch/TensorFlow available offline).

Architecture: char embedding -> parallel 1-D convolutions (several widths) -> ReLU -> global max-pool -> dense(ReLU) -> logit.
Loss: class-weighted binary cross-entropy (pos_weight > 1 favours phishing recall). Optimiser: Adam. Backprop is hand-derived
and verified against finite differences in tests/test_v2_charcnn.py.

Input text = lower-cased URL with scheme and a leading 'www.' removed (the dataset audit showed both are collection artefacts),
truncated/padded to `max_len` characters. This is a small model by design (millisecond CPU inference); it is NOT a Transformer.
"""
from __future__ import annotations
import re, time
import numpy as np, scipy.sparse as sps
from numpy.lib.stride_tricks import sliding_window_view
from ..features.lexical import strip_www_url

VOCAB = 98            # 0 = pad, 1..95 = printable ASCII 32..126, 96 = other
_SCHEME = re.compile(r"^\s*[a-zA-Z][a-zA-Z0-9+.-]*:/{0,2}\s*")

def prep(url: str) -> str:
    return _SCHEME.sub("", strip_www_url(url).strip(), count=1).lower()

def encode(urls, max_len: int):
    X = np.zeros((len(urls), max_len), dtype=np.int64); lens = np.zeros(len(urls), dtype=np.int64)
    for i, u in enumerate(urls):
        t = prep(u)[:max_len]
        lens[i] = max(1, len(t))
        for j, ch in enumerate(t):
            o = ord(ch); X[i, j] = o - 31 if 32 <= o <= 126 else 96
    return X, lens

def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

class CharCNN:
    def __init__(self, max_len=128, emb=24, widths=(3, 5), n_filters=64, hidden=32, seed=0, dtype=np.float32, pos_weight=1.0):
        self.max_len, self.emb, self.widths, self.nf, self.hidden = max_len, emb, tuple(widths), n_filters, hidden
        self.seed, self.dtype, self.pos_weight = seed, dtype, pos_weight
        r = np.random.default_rng(seed); P = {}
        P["E"] = r.normal(0, 0.1, (VOCAB, emb)); P["E"][0] = 0
        for k in self.widths:
            P[f"W{k}"] = r.normal(0, np.sqrt(2.0 / (k * emb)), (k * emb, n_filters)); P[f"b{k}"] = np.zeros(n_filters)
        F = len(self.widths) * n_filters
        P["H"] = r.normal(0, np.sqrt(2.0 / F), (F, hidden)); P["hb"] = np.zeros(hidden)
        P["O"] = r.normal(0, np.sqrt(1.0 / hidden), (hidden, 1)); P["ob"] = np.zeros(1)
        self.P = {k: v.astype(dtype) for k, v in P.items()}
        self.history = []

    # ------------------------------------------------------------------ forward / backward
    def _forward(self, X, lens):
        P = self.P; B, L = X.shape; emb = P["E"][X]                           # (B,L,d)
        c = {"X": X, "B": B, "L": L, "pool": [], "arg": [], "win": []}
        for k in self.widths:
            T = L - k + 1
            win = np.swapaxes(sliding_window_view(emb, k, axis=1), 2, 3).reshape(B, T, k * self.emb)   # win[b,t,j*d+c]=emb[b,t+j,c]
            act = np.maximum(win.reshape(B * T, -1) @ P[f"W{k}"] + P[f"b{k}"], 0).reshape(B, T, self.nf)
            act = np.where((np.arange(T)[None, :] < lens[:, None])[:, :, None], act, 0)       # ignore windows that start in the padding
            arg = act.argmax(axis=1); pool = np.take_along_axis(act, arg[:, None, :], axis=1)[:, 0, :]
            c["win"].append(win); c["arg"].append(arg); c["pool"].append(pool)
        pool = np.concatenate(c["pool"], axis=1); hpre = pool @ P["H"] + P["hb"]; h = np.maximum(hpre, 0)
        z = (h @ P["O"] + P["ob"])[:, 0]
        c.update(pool_all=pool, hpre=hpre, h=h)
        return z, c

    def _backward(self, dz, c):
        P = self.P; B, L = c["B"], c["L"]; G = {}
        G["O"] = c["h"].T @ dz[:, None]; G["ob"] = np.array([dz.sum()])
        dh = (dz[:, None] @ P["O"].T) * (c["hpre"] > 0)
        G["H"] = c["pool_all"].T @ dh; G["hb"] = dh.sum(0)
        dpool = dh @ P["H"].T
        demb_rows, demb_cols, demb_vals = [], [], []
        dE_in = np.zeros((B * L, self.emb), dtype=self.dtype)
        for i, k in enumerate(self.widths):
            sl = slice(i * self.nf, (i + 1) * self.nf)
            pool, arg, win = c["pool"][i], c["arg"][i], c["win"][i]
            dact = dpool[:, sl] * (pool > 0)                                         # (B,nf)
            sel = win[np.arange(B)[:, None], arg]                                    # (B,nf,k*d)
            G[f"W{k}"] = np.einsum("bfk,bf->kf", sel, dact); G[f"b{k}"] = dact.sum(0)
            dsel = np.einsum("kf,bf->bfk", P[f"W{k}"], dact).reshape(B, self.nf, k, self.emb)
            pos = arg[:, :, None] + np.arange(k)[None, None, :]                      # (B,nf,k) char positions
            flat = (np.arange(B)[:, None, None] * L + pos).ravel()
            S = sps.csr_matrix((np.ones(flat.size, dtype=self.dtype), (flat, np.arange(flat.size))), shape=(B * L, flat.size))
            dE_in += S @ dsel.reshape(-1, self.emb)
        tok = sps.csr_matrix((np.ones(B * L, dtype=self.dtype), (c["X"].ravel(), np.arange(B * L))), shape=(VOCAB, B * L))
        G["E"] = tok @ dE_in; G["E"][0] = 0
        return G

    def loss_and_grads(self, X, lens, y):
        z, c = self._forward(X, lens); p = _sigmoid(z); w = self.pos_weight; B = len(y)
        eps = 1e-12
        loss = -np.mean(w * y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        dz = (-w * y * (1 - p) + (1 - y) * p) / B
        return float(loss), self._backward(dz.astype(self.dtype), c)

    # ------------------------------------------------------------------ training / inference
    def predict_proba(self, urls, batch=1024, enc=None):
        X, lens = enc if enc is not None else encode(urls, self.max_len)
        out = np.empty(len(X), dtype=np.float64)
        for i in range(0, len(X), batch):
            z, _ = self._forward(X[i:i + batch], lens[i:i + batch]); out[i:i + batch] = _sigmoid(z)
        return out

    def fit(self, urls, y, val=None, epochs=3, batch=256, lr=2e-3, seed=0, time_budget_s=None, log=print):
        """val = (urls_val, y_val); the epoch with the best validation average precision is kept (model selection on validation only)."""
        from sklearn.metrics import average_precision_score
        X, lens = encode(urls, self.max_len); y = np.asarray(y, dtype=self.dtype)
        Xv = encode(val[0], self.max_len) if val is not None else None
        r = np.random.default_rng(seed); m = {k: np.zeros_like(v) for k, v in self.P.items()}; v2 = {k: np.zeros_like(v) for k, v in self.P.items()}
        step, best, t0 = 0, (-1, None), time.time(); n = len(X); nb = (n + batch - 1) // batch
        for ep in range(epochs):
            order = r.permutation(n); cur_lr = lr * (0.5 ** ep); tl = 0.0
            for bi in range(nb):
                ix = order[bi * batch:(bi + 1) * batch]
                loss, G = self.loss_and_grads(X[ix], lens[ix], y[ix]); tl += loss; step += 1
                for k in self.P:
                    m[k] = 0.9 * m[k] + 0.1 * G[k]; v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
                    self.P[k] -= (cur_lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)).astype(self.dtype)
                if bi % 200 == 0: log(f"epoch {ep} batch {bi}/{nb} loss {tl / (bi + 1):.4f} elapsed {time.time() - t0:.0f}s")
                if time_budget_s and time.time() - t0 > time_budget_s: break
            rec = {"epoch": ep, "train_loss": tl / max(1, bi + 1), "seconds": time.time() - t0}
            if val is not None:
                pv = self.predict_proba(None, enc=Xv); rec["val_ap"] = float(average_precision_score(val[1], pv))
                if rec["val_ap"] > best[0]: best = (rec["val_ap"], {k: v.copy() for k, v in self.P.items()})
            self.history.append(rec); log(str(rec))
            if time_budget_s and time.time() - t0 > time_budget_s: break
        if best[1] is not None: self.P = best[1]
        self.fit_seconds = time.time() - t0
        return self

    # ------------------------------------------------------------------ explanation
    def explain_chars(self, url: str):
        """Fast gradient x activation attribution per character of the model's input text (association with the logit,
        not a causal effect; one forward pass + dense-layer gradient). Returns (text, scores)."""
        text = prep(url)[: self.max_len]; X, lens = encode([url], self.max_len)
        z, c = self._forward(X, lens); P = self.P
        dpool = ((c["hpre"][0] > 0).astype(self.dtype) * P["O"][:, 0]) @ P["H"].T            # d logit / d pooled feature
        contrib = dpool * c["pool_all"][0]; scores = np.zeros(self.max_len, dtype=np.float64)
        for i, k in enumerate(self.widths):
            sl = slice(i * self.nf, (i + 1) * self.nf)
            for f, t in enumerate(c["arg"][i][0]):
                scores[t:t + k] += contrib[sl][f] / k
        return text, scores[: len(text)], float(z[0])

    def n_params(self): return int(sum(v.size for v in self.P.values()))
