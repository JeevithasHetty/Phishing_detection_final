"""Dataset loading + schema validation. Raw CSVs are opened read-only."""
from __future__ import annotations
import pandas as pd
from ..common import data_dir, load_config

URL_FILES = ["train", "val", "test", "recent"]

def load_split(name: str, cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = data_dir(cfg) / cfg["data"]["files"][name]
    df = pd.read_csv(path, dtype={"url": str, "label": "int8"}, keep_default_na=False)
    validate_schema(df, name, cfg)
    if "sources" not in df.columns:
        df["sources"] = df.get("source", "unknown")
    return df

def validate_schema(df: pd.DataFrame, name: str, cfg: dict):
    u, l = cfg["data"]["url_col"], cfg["data"]["label_col"]
    missing = {u, l, "domain"} - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if not set(df[l].unique()) <= {0, 1}:
        raise ValueError(f"{name}: labels outside {{0,1}}: {sorted(df[l].unique())}")
    if (df[u].astype(str).str.len() == 0).any():
        raise ValueError(f"{name}: empty URLs present")
