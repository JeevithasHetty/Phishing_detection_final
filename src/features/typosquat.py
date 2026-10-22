"""Confusable canonicalisation ("skeleton") and typosquat / brand-impersonation features.

Everything here is offline string processing. The *brand reference set* is NOT an external list: it is derived from the
TRAINING partition only (registered domains that are almost purely legitimate and most frequent), so fitting it never
touches validation/test labels. The confusable map is a small hand-written subset (Latin look-alike digits/symbols plus
common Cyrillic/Greek homoglyphs); it is NOT the full Unicode confusables table, which is not available offline.
"""
from __future__ import annotations
import re, unicodedata
from collections import Counter, defaultdict
import numpy as np
from ..data.urls import split_url, is_ip_host, registered_domain

_SINGLE = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "i": "l", "|": "l", "$": "s", "@": "a",
    # Cyrillic
    "а": "a", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x",
    "ѕ": "s", "і": "l", "ј": "j", "ӏ": "l", "ԁ": "d", "ɡ": "g", "ԛ": "q", "ԝ": "w",
    # Greek
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "ι": "l", "κ": "k", "υ": "u", "χ": "x",
}
_CONF_DIGITS = set("013457 8".replace(" ", ""))
_TRANS = {ord(k): v for k, v in _SINGLE.items()}

def skeleton(s: str) -> str:
    """Canonical look-alike form: NFKC, lowercase, single-char confusables, rn->m, vv->w, drop '-'/'_'/'.'."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.translate(_TRANS)
    s = s.replace("rn", "m").replace("vv", "w")
    return re.sub(r"[-_.]", "", s)

def has_non_ascii_or_puny(host: str) -> bool:
    return any(ord(c) > 127 for c in host) or "xn--" in host

def levenshtein(a: str, b: str, cap: int = 3) -> int:
    if a == b: return 0
    if abs(len(a) - len(b)) >= cap: return cap
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return min(prev[-1], cap)

def _deletions(s: str):
    return {s[:i] + s[i + 1:] for i in range(len(s))}

TYPO_FEATURE_NAMES = ["is_brand_exact", "skeleton_collision", "min_edit_to_brand", "edit_le1", "edit_le2",
                      "brand_in_subdomain", "brand_in_path", "confusable_digit_in_sld", "non_ascii_or_punycode", "hyphen_squat"]

class BrandIndex:
    """Reference set of popular, near-purely-legitimate registered domains from TRAINING rows."""
    def __init__(self, top_k: int = 2000, min_len: int = 5, min_urls: int = 5, min_legit_frac: float = 0.99):
        self.top_k, self.min_len, self.min_urls, self.min_legit_frac = top_k, min_len, min_urls, min_legit_frac
        self.fitted = False; self._cache = {}

    def fit(self, rd, label):
        rd = np.asarray(rd); y = np.asarray(label).astype(int)
        tot, leg = Counter(rd.tolist()), Counter(rd[y == 0].tolist())
        cands = [(leg[d], d) for d in leg if tot[d] >= self.min_urls and leg[d] / tot[d] >= self.min_legit_frac and "." in d
                 and not re.fullmatch(r"[\d.]+", d) and len(d.split(".")[0]) >= self.min_len]
        cands.sort(reverse=True)
        self.brand_rd = [d for _, d in cands[: self.top_k]]
        self.rd_set = set(self.brand_rd)
        self.slds = sorted({d.split(".")[0] for d in self.brand_rd})
        self.sld_set = set(self.slds)
        self.skel_to_slds = defaultdict(set); self.skel_set = set()
        for s in self.slds:
            k = skeleton(s); self.skel_to_slds[k].add(s); self.skel_set.add(k)
        self.del_index = defaultdict(set)
        for k in self.skel_set:
            if len(k) >= self.min_len:
                self.del_index[k].add(k)
                for d in _deletions(k): self.del_index[d].add(k)
        self.fitted = True; self._cache = {}
        return self

    def _nearest(self, q: str):
        """(min edit distance to a *different* brand skeleton, capped at 3). Uses a deletion-neighbourhood index."""
        if len(q) < self.min_len: return 3
        cands = set(self.del_index.get(q, ()))
        for d in _deletions(q): cands |= self.del_index.get(d, set())
        best = 3
        for b in cands:
            if b == q: continue
            best = min(best, levenshtein(q, b))
        return best

    def sld_features(self, rd: str):
        if rd in self._cache: return self._cache[rd]
        sld = rd.split(".")[0] if rd else ""
        sk = skeleton(sld)
        exact = int(rd in self.rd_set)
        coll = int(any(b != sld for b in self.skel_to_slds.get(sk, ())) )
        ed = 0 if exact else self._nearest(sk)
        if exact: ed = 3
        conf_digit = int(bool(re.search(r"[a-z][0-9]+[a-z]|[a-z][0-9]+$", sld)) and any(c in _CONF_DIGITS for c in sld)
                         and sk.isalpha())
        hy = int("-" in sld and (skeleton(sld.replace("-", "")) in self.skel_set) and sld not in self.sld_set)
        out = (exact, coll, ed, int(ed <= 1), int(ed <= 2), conf_digit, hy)
        self._cache[rd] = out
        return out

    def features_one(self, url: str, rd: str | None = None, host: str | None = None) -> list:
        assert self.fitted
        _, h, rest = split_url(url)
        host = host if host is not None else h
        rd = rd if rd is not None else (host if is_ip_host(host) else registered_domain(host, is_host=True))
        exact, coll, ed, le1, le2, cd, hy = self.sld_features(rd)
        sub = host[: -len(rd)].rstrip(".") if rd and host.endswith(rd) else host
        toks = [t for t in re.split(r"[^a-z0-9]+", sub) if len(t) >= self.min_len]
        sub_hit = int(any(skeleton(t) in self.skel_set for t in toks) and not exact)
        ptoks = [t for t in re.split(r"[^a-z0-9]+", rest.lower()) if len(t) >= self.min_len]
        path_hit = int(any(skeleton(t) in self.skel_set for t in ptoks) and not exact)
        return [exact, coll, ed, le1, le2, sub_hit, path_hit, cd, int(has_non_ascii_or_puny(host)), hy]

    def features(self, urls, rds=None, hosts=None) -> np.ndarray:
        n = len(urls)
        rows = [self.features_one(u, None if rds is None else rds[i], None if hosts is None else hosts[i]) for i, u in enumerate(urls)]
        X = np.asarray(rows, dtype=np.float32)
        assert X.shape == (n, len(TYPO_FEATURE_NAMES))
        return X
