"""Shared paths, hashing, seeding and JSON helpers."""
from __future__ import annotations
import hashlib, json, os, platform, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    return np.random.default_rng(seed)

def load_config(path=None) -> dict:
    import yaml
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

def data_dir(cfg=None) -> Path:
    """Dataset directory: env PHISH_DATA_DIR, else config data.dir (relative to ROOT)."""
    cfg = cfg or load_config()
    p = os.environ.get("PHISH_DATA_DIR") or cfg["data"]["dir"]
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)

def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_default)

def read_json(path):
    with open(path) as f:
        return json.load(f)

def _default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    raise TypeError(type(o))

def env_info() -> dict:
    import sklearn, pandas, scipy
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pandas.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__,
            "cpu_count": os.cpu_count()}
