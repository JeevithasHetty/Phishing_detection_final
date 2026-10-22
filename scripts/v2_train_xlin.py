"""Train XLin ablation variants on the TRAIN partition of a registered-domain-disjoint split and save raw predictions.
Variants (name -> pos_weight, use_canon, use_typo):
  xlin_pw1_base  (1.0, no canon, no typo)   ~ v1 'hybrid_lr' feature set, no recall weighting
  xlin_pw2_base  (2.0, no canon, no typo)   isolates the effect of recall-oriented class weighting
  xlin_pw2_canon (2.0, canon, no typo)      + confusable-skeleton host n-grams
  xlin_pw2_full  (2.0, canon, typo)         + typosquat/brand features (brand set derived from TRAIN only)
  xlin_pw2_aug   (2.0, no canon, no typo)   + augmentation: half of TRAIN phishing URLs get a padded copy (benign path segments sampled from TRAIN legit URLs)
Usage: python scripts/v2_train_xlin.py --seed 11 [--variants a b ...] [--smoke]"""
import _bootstrap, argparse, io, time
import numpy as np, pandas as pd, joblib
from src.common import ROOT, load_config, write_json, sha256_file, env_info
from src.evaluation.predictions import partitions

VARIANTS = {"xlin_pw1_base": (1.0, False, False), "xlin_pw2_base": (2.0, False, False), "xlin_pw2_canon": (2.0, True, False), "xlin_pw2_full": (2.0, True, True),
            "xlin_pw2_aug": (2.0, False, False)}   # aug = pw2_base features + padding-evasion augmentation of TRAIN phishing URLs



def augment_padding(tr, lexp, seed=0, frac=0.5, n_seg=(2, 6)):
    """Return (urls, labels, lex, rd, host) with padded copies of a random `frac` of TRAIN phishing URLs. Padding segments are sampled from the
    path segments of TRAIN legitimate URLs only (no validation/test information, and not the evaluation attack's tokens)."""
    import re
    from src.features.lexical import featurize
    rng = np.random.default_rng(seed); legit = tr[tr.label.astype(int) == 0].url.values
    segs = []
    for u in legit[:100000]:
        path = re.sub(r"^[a-zA-Z]+:/{0,2}\s*[^/]*", "", u); segs += [x for x in re.split(r"[/]+", path.split("?")[0]) if 3 <= len(x) <= 20 and re.fullmatch(r"[A-Za-z0-9._-]+", x)]
    segs = np.array(sorted(set(segs))); ph = tr[tr.label.astype(int) == 1]; pick = ph.sample(frac=frac, random_state=seed)
    padded = [u.rstrip("/") + "/" + "/".join(rng.choice(segs, int(rng.integers(n_seg[0], n_seg[1] + 1)))) for u in pick.url.values]
    lex_pad = featurize(padded, True)
    urls = np.concatenate([tr.url.values, np.array(padded, dtype=object)]); y = np.concatenate([tr.label.values.astype(int), np.ones(len(padded), dtype=int)])
    return urls, y, np.vstack([lexp[tr.pos.values], lex_pad]), np.concatenate([tr.rd.values, pick.rd.values]), np.concatenate([tr.host.values, pick.host.values])

def main():
    from src.models.xlin import XLin
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=11); ap.add_argument("--variants", nargs="*", default=list(VARIANTS)); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); cfg = load_config(); proto = f"regdomain_seed{a.seed}"; proc = ROOT / cfg["processed_dir"]
    parts, rec = partitions(proto, cfg); lexp = np.load(proc / "lex_norm_pool.npy"); lexr = np.load(proc / "lex_norm_rec.npy")
    out = ROOT / "results" / "v2" / ("predictions_smoke" if a.smoke else "predictions"); out.mkdir(parents=True, exist_ok=True)
    mdir = ROOT / "artifacts" / "v2"; mdir.mkdir(parents=True, exist_ok=True)
    tr = parts["train"].sample(20000, random_state=0) if a.smoke else parts["train"]
    for name in a.variants:
        pw, canon, typo = VARIANTS[name]; tag = f"{proto}__{name}"
        t0 = time.time()
        if name == "xlin_pw2_aug": U, Y, LX, RD, HS = augment_padding(tr, lexp, seed=a.seed)
        else: U, Y, LX, RD, HS = tr.url.values, tr.label.values.astype(int), lexp[tr.pos.values], tr.rd.values, tr.host.values
        m = XLin(use_canon=canon, use_typo=typo, pos_weight=pw, seed=a.seed).fit(U, Y, lex=LX, rd=RD, host=HS)
        preds = {}
        for k in ["val", "policy", "test"]:
            d = parts[k]; preds[f"p_{k}"] = m.predict_proba(d.url.values, lex=lexp[d.pos.values], rd=d.rd.values, host=d.host.values)
        preds["p_recent"] = m.predict_proba(rec.url.values, lex=lexr, rd=rec.rd.values, host=rec.host.values)
        d = parts["test"].iloc[:3000]; t1 = time.time(); m.predict_proba(d.url.values); lat = (time.time() - t1) / len(d) * 1000     # includes feature extraction
        buf = io.BytesIO(); joblib.dump(m, buf, compress=3)
        np.savez_compressed(out / f"{tag}.npz", **preds)
        if name in ("xlin_pw2_full", "xlin_pw2_base", "xlin_pw2_aug") and not a.smoke: joblib.dump(m, mdir / f"{tag}.joblib", compress=3)
        write_json({"tag": tag, "protocol": proto, "variant": name, "pos_weight": pw, "use_canon": canon, "use_typo": typo, "fit_seconds": m.fit_seconds,
                    "latency_ms_per_url_incl_features": lat, "model_bytes_compressed": buf.getbuffer().nbytes, "n_train": int(len(U)), "smoke": a.smoke,
                    "dataset_master_sha256": sha256_file(ROOT / cfg["data"]["dir"] / cfg["data"]["files"]["master"]), "env": env_info()}, out / f"{tag}.json")
        print("done", tag, f"{time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)

if __name__ == "__main__": main()
