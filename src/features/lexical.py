"""Hand-crafted lexical URL features. Pure string processing; no network, no reputation lookups.
`strip_www=True` removes a leading 'www.' from the host *before* featurisation: the dataset audit showed 'www.'
is a collection artefact (present in ~99.9% of legitimate but ~28% of phishing URLs in the pooled data)."""
from __future__ import annotations
import math, re
from collections import Counter
import numpy as np, pandas as pd
from ..data.urls import split_url, is_ip_host, registered_domain
from ..data.psl import get_psl

SUSPICIOUS = ["login", "signin", "secure", "account", "update", "verify", "bank", "paypal", "confirm", "password",
              "webscr", "ebay", "apple", "support", "billing", "wallet", "free", "bonus", "click", "service"]
EXT = [".php", ".html", ".htm", ".exe", ".zip", ".js", ".asp", ".jpg", ".png"]
_HEX = re.compile(r"[0-9a-f]{16,}")
_DIGRUN = re.compile(r"\d+")

def strip_www_url(url: str) -> str:
    s = url.strip()
    return re.sub(r"^(\s*[a-zA-Z][a-zA-Z0-9+.-]*:/{0,2}\s*)www\d{0,3}\.", r"\1", s, count=1, flags=re.I)

def _entropy(s: str) -> float:
    if not s: return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())

FEATURE_NAMES = (["len_url", "len_host", "len_path", "len_query", "n_dots_host", "n_subdomains", "n_hyphen_host",
                  "n_digits_host", "n_digits_url", "digit_ratio_url", "alpha_ratio_url", "n_special_url", "has_at",
                  "n_slash", "n_question", "n_equal", "n_amp", "n_percent", "n_underscore", "is_ip", "tld_len",
                  "host_entropy", "path_entropy", "longest_host_token", "longest_digit_run", "has_hex_blob",
                  "has_punycode", "has_port_like", "path_depth", "has_http_in_path", "rd_len", "host_label_max_len"]
                 + [f"kw_{k}" for k in SUSPICIOUS] + [f"ext_{e.strip('.')}" for e in EXT])

def url_features(url: str) -> list:
    scheme, host, rest = split_url(url)
    ip = is_ip_host(host)
    path, _, q = rest.partition("?")
    labels = host.split(".") if host else [""]
    rd = host if ip else get_psl().registered_domain(host)
    suf = "" if ip else get_psl().public_suffix(host)
    nsub = max(0, len(labels) - len(rd.split(".")) ) if not ip else 0
    low = url.lower()
    n = max(1, len(url))
    digits = sum(ch.isdigit() for ch in url); alpha = sum(ch.isalpha() for ch in url)
    toks = re.split(r"[^a-z0-9]+", host)
    runs = _DIGRUN.findall(url)
    f = [len(url), len(host), len(path), len(q), host.count("."), nsub, host.count("-"),
         sum(c.isdigit() for c in host), digits, digits / n, alpha / n, n - digits - alpha,
         int("@" in url), url.count("/"), url.count("?"), url.count("="), url.count("&"), url.count("%"),
         url.count("_"), int(ip), len(suf.split(".")[-1]) if suf else 0, _entropy(host), _entropy(path),
         max((len(t) for t in toks), default=0), max((len(r) for r in runs), default=0),
         int(bool(_HEX.search(low))), int("xn--" in host), int(bool(re.search(r":\d+", url.split("/")[2] if url.count("/") >= 2 else ""))),
         path.count("/"), int("http" in path.lower()), len(rd), max(len(l) for l in labels)]
    f += [int(k in low) for k in SUSPICIOUS]
    f += [int(path.lower().endswith(e) or (e + "?") in low) for e in EXT]
    return f

def featurize(urls, strip_www=False) -> np.ndarray:
    it = (strip_www_url(u) for u in urls) if strip_www else iter(urls)
    X = np.asarray([url_features(u) for u in it], dtype=np.float32)
    assert X.shape[1] == len(FEATURE_NAMES)
    return X
