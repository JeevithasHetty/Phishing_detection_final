"""Train supervised baselines for every protocol/variant/seed and save raw predictions (+ model sizes, latency).
Resumable: a job is skipped only if its cache key (config, data hash, split checksum, source of model code) matches."""
import _bootstrap, argparse, io, time, json, hashlib
import numpy as np, pandas as pd, joblib
from pathlib import Path
from src.common import ROOT, load_config, sha256_file, write_json, read_json, sha256_text, env_info
from src.data.splits import PARTS, split_checksum
from src.features.lexical import featurize
from src.models.zoo import UrlModel
from src.models.stacking import oof_predictions, Stacker

KINDS = ["lr_lex", "rf_lex", "hgb_lex", "tfidf_lr", "hybrid_lr"]
FAST = ["lr_lex", "hgb_lex", "tfidf_lr"]

def code_hash():
    return sha256_text("".join((ROOT / p).read_text() for p in ["src/models/zoo.py", "src/features/lexical.py", "src/models/stacking.py", "src/data/splits.py", "config/config.yaml"]))

def jobs(cfg):
    J = []
    for seed in cfg["seeds"]:
        J.append((f"regdomain_seed{seed}", "norm", KINDS + ["stack"], seed))
    J.append((f"regdomain_seed{cfg['seeds'][0]}", "full", KINDS + ["stack"], cfg["seeds"][0]))
    for pr in ["random_seed42", "hostname_seed42"]:
        for v in ["norm", "full"]: J.append((pr, v, KINDS, cfg["seed"]))
    for pr in ["source_PhiUSIIL_to_Mendeley-450K", "source_Mendeley-450K_to_PhiUSIIL"]:
        for v in ["norm", "full"]: J.append((pr, v, FAST, cfg["seed"]))
    # primary first
    order = {("regdomain_seed%d" % cfg["seeds"][0], "norm"): 0, ("regdomain_seed%d" % cfg["seeds"][0], "full"): 1}
    return sorted(J, key=lambda j: order.get((j[0], j[1]), 2))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    cfg = load_config(); proc = ROOT / cfg["processed_dir"]; pool = pd.read_pickle(proc / "pool.pkl"); rec = pd.read_pickle(proc / "recent.pkl")
    if a.smoke:
        pool = pool.sample(40000, random_state=0).reset_index(drop=True); rec = rec.sample(5000, random_state=0).reset_index(drop=True)
    lexc = {}
    def lex(variant, which):
        f = proc / f"lex_{variant}_{which}{'_smoke' if a.smoke else ''}.npy"
        if f.exists(): return np.load(f)
        X = featurize((pool if which == "pool" else rec).url.values, strip_www=(variant == "norm")); np.save(f, X); return X
    outdir = ROOT / "results" / ("predictions_smoke" if a.smoke else "predictions"); outdir.mkdir(parents=True, exist_ok=True)
    ch = code_hash(); dh = sha256_file(Path(__import__("src.common", fromlist=["x"]).data_dir(cfg)) / cfg["data"]["files"]["master"])
    modeldir = ROOT / "artifacts" / "models"; modeldir.mkdir(parents=True, exist_ok=True)
    for proto, variant, kinds, seed in jobs(cfg):
        if a.only and a.only not in f"{proto}/{variant}": continue
        spf = pd.read_csv(proc / f"split_{proto}.csv.gz"); sp = pool["row_id"].map(dict(zip(spf.row_id, spf.split))).fillna("drop").values
        if a.smoke: sp = np.array(["train", "val", "policy", "test"])[np.random.default_rng(1).integers(0, 4, len(pool))] if proto.startswith("random") else sp
        scs = split_checksum(pool["row_id"].values, sp)
        Xp, Xr = lex(variant, "pool"), lex(variant, "rec")
        m = {p: sp == p for p in PARTS}
        tr_idx = np.flatnonzero(m["train"]); train_rd = set(pool.loc[m["train"], "rd"])
        rec_novel = (~rec["rd"].isin(train_rd)).values
        for kind in kinds:
            tag = f"{proto}__{variant}__{kind}"; fpath = outdir / f"{tag}.npz"; mpath = outdir / f"{tag}.json"
            key = sha256_text(f"{ch}|{dh}|{scs}|{tag}|{cfg['budget']}|{a.smoke}")
            if fpath.exists() and mpath.exists() and read_json(mpath).get("key") == key:
                print("cached", tag, flush=True); continue
            t0 = time.time(); urls_tr = pool["url"].values[tr_idx]; y_tr = pool["label"].values[tr_idx]
            info = {"protocol": proto, "variant": variant, "kind": kind, "seed": seed, "key": key, "split_checksum": scs, "dataset_sha256_master": dh}
            if kind == "stack":
                bases = ["lr_lex", "hgb_lex", "tfidf_lr"]
                oof = oof_predictions(bases, urls_tr, y_tr, pool["rd"].values[tr_idx], Xp[tr_idx], variant == "norm", seed, tree_cap=cfg["budget"]["tree_train_cap"], max_features=cfg["budget"]["tfidf_max_features"])
                st = Stacker().fit(oof, y_tr)
                P = {}
                for b in bases: P[b] = np.load(outdir / f"{proto}__{variant}__{b}.npz")
                out = {f"p_{p}": st.predict_proba(np.column_stack([P[b][f"p_{p}"] for b in bases])) for p in ["val", "policy", "test", "recent"]}
                info.update({"base_models": bases, "meta_coef": st.meta.coef_.tolist(), "oof_folds": 3, "fit_seconds": time.time() - t0})
                np.savez_compressed(fpath, **out, recent_novel=rec_novel); write_json(info, mpath); print("done", tag, f"{time.time()-t0:.0f}s", flush=True); continue
            model = UrlModel(kind, strip_www=(variant == "norm"), seed=seed, tree_cap=cfg["budget"]["tree_train_cap"], max_features=cfg["budget"]["tfidf_max_features"])
            model.fit(urls_tr, y_tr, lex=Xp[tr_idx])
            out = {}
            for p in ["val", "policy", "test"]:
                ix = np.flatnonzero(m[p]); out[f"p_{p}"] = model.predict_proba(pool["url"].values[ix], lex=Xp[ix]) if len(ix) else np.array([])
            out["p_recent"] = model.predict_proba(rec["url"].values, lex=Xr)
            # latency (featurisation + inference) and serialised size
            ix = np.flatnonzero(m["test"])[:3000]; t1 = time.time(); model.predict_proba(pool["url"].values[ix]); lat = (time.time() - t1) / max(1, len(ix)) * 1000
            buf = io.BytesIO(); joblib.dump(model, buf, compress=3)
            info.update({"fit_seconds": model.fit_seconds, "latency_ms_per_url_incl_features": lat, "model_bytes_compressed": buf.getbuffer().nbytes, "n_train": int(len(tr_idx)), "env": env_info()})
            if proto == f"regdomain_seed{cfg['seeds'][0]}" and variant == "norm" and kind != "rf_lex":
                joblib.dump(model, modeldir / f"{tag}.joblib", compress=3)
            np.savez_compressed(fpath, **out, recent_novel=rec_novel); write_json(info, mpath)
            print("done", tag, f"{time.time()-t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)
if __name__ == "__main__": main()
