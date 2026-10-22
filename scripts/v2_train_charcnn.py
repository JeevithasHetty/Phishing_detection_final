"""Train the NumPy char-CNN on the TRAIN partition of a registered-domain-disjoint split; select the epoch by validation AP;
save predictions for val/policy/test/recent and the model weights. Usage: python scripts/v2_train_charcnn.py [--seed 11] [--pos-weight 2]"""
import _bootstrap, argparse, json, time
import numpy as np, pandas as pd
from src.common import ROOT, load_config, write_json, sha256_file, env_info
from src.evaluation.predictions import partitions
from src.models.charcnn import CharCNN

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=11); ap.add_argument("--pos-weight", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--budget-min", type=float, default=40); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); cfg = load_config(); proto = f"regdomain_seed{a.seed}"
    parts, rec = partitions(proto, cfg); tr, va = parts["train"], parts["val"]
    if a.smoke: tr, va = tr.sample(20000, random_state=0), va.sample(5000, random_state=0)
    out = ROOT / "results" / "v2" / ("predictions_smoke" if a.smoke else "predictions"); out.mkdir(parents=True, exist_ok=True)
    mdir = ROOT / "artifacts" / "v2"; mdir.mkdir(parents=True, exist_ok=True)
    tag = f"{proto}__charcnn_pw{a.pos_weight:g}"
    m = CharCNN(max_len=128, emb=32, widths=(3, 5, 7), n_filters=64, hidden=32, seed=a.seed, pos_weight=a.pos_weight)
    t0 = time.time()
    m.fit(tr.url.values, tr.label.values.astype(int), val=(va.url.values, va.label.values.astype(int)), epochs=a.epochs, batch=256, lr=2e-3,
          seed=a.seed, time_budget_s=a.budget_min * 60, log=lambda s: print(s, flush=True))
    preds = {f"p_{k}": m.predict_proba(parts[k].url.values) for k in ["val", "policy", "test"]}
    preds["p_recent"] = m.predict_proba(rec.url.values)
    t1 = time.time(); m.predict_proba(parts["test"].url.values[:3000]); lat = (time.time() - t1) / 3000 * 1000
    np.savez_compressed(out / f"{tag}.npz", **preds)
    if not a.smoke: np.savez_compressed(mdir / f"{tag}.model.npz", **m.P)
    write_json({"tag": tag, "protocol": proto, "pos_weight": a.pos_weight, "epochs_run": len(m.history), "history": m.history, "n_params": m.n_params(),
                "latency_ms_per_url": lat, "fit_seconds": time.time() - t0, "n_train": int(len(tr)), "smoke": a.smoke,
                "config": {"max_len": 128, "emb": 32, "widths": [3, 5, 7], "n_filters": 64, "hidden": 32, "lr": 2e-3, "batch": 256},
                "dataset_master_sha256": sha256_file(ROOT / cfg["data"]["dir"] / cfg["data"]["files"]["master"]), "env": env_info()}, out / f"{tag}.json")
    print("DONE", tag, flush=True)

if __name__ == "__main__": main()
