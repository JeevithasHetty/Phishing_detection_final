"""Package the best saved model (by validation average precision) + calibrator + cost model for scripts/predict_url.py."""
import _bootstrap, joblib
from sklearn.metrics import average_precision_score
from src.common import ROOT, load_config, sha256_file, sha256_text, data_dir
from src.evaluation.predictions import load_calibrated
from src.decision.costs import CostModel
from src.inference.predict import Detector

def main():
    cfg = load_config(); c = cfg["cost"]; proto = f"regdomain_seed{cfg['seeds'][0]}"; best = None
    for k in ["lr_lex", "hgb_lex", "tfidf_lr", "hybrid_lr"]:
        D = load_calibrated(proto, "norm", k, cfg); s = average_precision_score(D["parts"]["val"]["label"].values, D["cal_val"])
        if best is None or s > best[0]: best = (s, k, D)
    s, k, D = best; model = joblib.load(ROOT / "artifacts" / "models" / f"{proto}__norm__{k}.joblib")
    meta = {"kind": k, "protocol": proto, "variant": "norm", "val_average_precision": float(s), "calibrator": D["calibrator_name"],
            "dataset_master_sha256": D["meta"]["dataset_sha256_master"], "config_sha256": sha256_text((ROOT / "config" / "config.yaml").read_text()),
            "costs_assumed": c, "note": "offline research prototype; URL-string features only; not production-validated"}
    Detector(model, D["calibrator"], CostModel(c["fn"], c["fp"], c["inspect"]), meta).save(ROOT / "artifacts" / "deployable_bundle.joblib"); print("bundle:", meta)
if __name__ == "__main__": main()
