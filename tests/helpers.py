import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def synth_urls(n=1200, seed=0):
    """SYNTHETIC fixture for isolated software tests only (never used for research results)."""
    r = np.random.default_rng(seed); rows = []
    legit = ["wikipedia.org", "example.com", "github.io", "bbc.co.uk", "python.org", "news.example.net"]
    for i in range(n):
        y = int(r.random() < 0.35)
        if y: host = f"secure-login-{r.integers(0, 9999)}.{['xyz', 'top', 'repl.co', 'tk'][i % 4]}"; path = "/verify/account.php?id=" + str(r.integers(0, 10**8))
        else: host = f"{['www.', 'blog.', ''][i % 3]}{legit[i % len(legit)]}"; path = "/wiki/page" + str(r.integers(0, 50))
        rows.append(("http://" + host + path, y, host))
    return pd.DataFrame(rows, columns=["url", "label", "domain"]).assign(sources="synthetic")
