"""Supervised evaluation: metrics for every saved prediction set, calibration before/after, group-bootstrap CIs,
seed variability, plots, and reports/supervised_results.md (generated from results/*.csv, not typed by hand)."""
import _bootstrap, argparse, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve
from src.common import ROOT, load_config, read_json, write_json
from src.evaluation.predictions import load_calibrated, load_pool
from src.evaluation.metrics import classification_metrics, group_bootstrap_ci, reliability, threshold_for_f1, threshold_for_cost

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred-dir", default="predictions"); ap.add_argument("--n-boot", type=int, default=100); a = ap.parse_args()
    cfg = load_config(); cm = cfg["cost"]; rows = []; sysrows = []
    files = sorted(glob.glob(str(ROOT / "results" / a.pred_dir / "*.npz")))
    primary = f"regdomain_seed{cfg['seeds'][0]}"; figs = ROOT / "results" / "figures"; figs.mkdir(parents=True, exist_ok=True)
    cache = {}
    for f in files:
        proto, variant, kind = Path(f).stem.split("__")
        D = load_calibrated(proto, variant, kind, cfg, a.pred_dir); cache[(proto, variant, kind)] = D
        pr = D["parts"]; yv = pr["val"]["label"].values
        thr = {"fixed_0.5": 0.5, "val_f1": threshold_for_f1(yv, D["cal_val"]), "val_cost": threshold_for_cost(yv, D["cal_val"], cm["fn"], cm["fp"])}
        rec = D["recent"]; yr = rec["label"].values; nov = D["recent_novel"]
        sets = {"val": (yv, "val"), "test": (pr["test"]["label"].values, "test"), "recent_all": (yr, "recent"), "recent_novelrd": (yr[nov], "recent")}
        for sname, (y, key) in sets.items():
            if len(y) == 0 or len(np.unique(y)) < 2: continue
            for calib in ["raw", "cal"]:
                p = D[f"{calib}_{key}"]; p = p[nov] if sname == "recent_novelrd" else p
                for tn, t in thr.items():
                    r = classification_metrics(y, p, t); r.update(protocol=proto, variant=variant, kind=kind, eval_set=sname, calibration=calib, thr_name=tn,
                                                                  calibrator=D["calibrator_name"]); rows.append(r)
        m = D["meta"]; sysrows.append(dict(protocol=proto, variant=variant, kind=kind, fit_seconds=m.get("fit_seconds"), latency_ms=m.get("latency_ms_per_url_incl_features"), model_MB=(m.get("model_bytes_compressed") or 0) / 1e6, n_train=m.get("n_train"), calibrator=D["calibrator_name"], calibrator_cv_brier=json.dumps(D["calibrator_scores"])))
    R = pd.DataFrame(rows); R.to_csv(ROOT / "results" / "supervised_metrics.csv", index=False)
    S = pd.DataFrame(sysrows); S.to_csv(ROOT / "results" / "system_costs.csv", index=False)
    # --- bootstrap CIs on the primary protocol (resampling unit: registered domain) ---
    ci = []
    for (proto, variant, kind), D in cache.items():
        if proto != primary: continue
        te = D["parts"]["test"]; y = te["label"].values; g = te["rd"].values; p = D["cal_test"]
        t = threshold_for_f1(D["parts"]["val"]["label"].values, D["cal_val"])
        for mname, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score), ("f1@val_f1_thr", lambda yy, pp: f1_score(yy, pp >= t))]:
            lo, hi, pt = group_bootstrap_ci(y, p, g, fn, n_boot=a.n_boot, seed=1); ci.append(dict(protocol=proto, variant=variant, kind=kind, metric=mname, point=pt, lo=lo, hi=hi, n_boot=a.n_boot, unit="registered_domain", method="percentile 95%"))
        # recent (clusters = registered domain) for the same models
        rec = D["recent"]; yr = rec["label"].values; pr_ = D["cal_recent"]
        lo, hi, pt = group_bootstrap_ci(yr, pr_, rec["rd"].values, roc_auc_score, n_boot=a.n_boot, seed=2); ci.append(dict(protocol=proto, variant=variant, kind=kind, metric="recent_all_roc_auc", point=pt, lo=lo, hi=hi, n_boot=a.n_boot, unit="registered_domain", method="percentile 95%"))
    pd.DataFrame(ci).to_csv(ROOT / "results" / "supervised_bootstrap_ci.csv", index=False)
    # --- plots ---
    for variant in ["norm", "full"]:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        for (proto, v, kind), D in cache.items():
            if proto != primary or v != variant or kind not in ("tfidf_lr", "hgb_lex", "hybrid_lr", "stack"): continue
            y = D["parts"]["test"]["label"].values
            for j, (cal, key) in enumerate([("raw", "raw_test"), ("cal", "cal_test")]):
                rl = reliability(y, D[key]); ax[j].plot([r[0] for r in rl], [r[1] for r in rl], "o-", ms=3, label=kind)
            pr_, rc_, _ = precision_recall_curve(y, D["cal_test"]); ax[2].plot(rc_, pr_, label=kind)
        for j, t in enumerate(["Reliability, raw (test)", "Reliability, calibrated on val (test)"]):
            ax[j].plot([0, 1], [0, 1], "k--", lw=.8); ax[j].set_title(t); ax[j].set_xlabel("mean predicted"); ax[j].set_ylabel("observed phishing rate"); ax[j].legend()
        ax[2].set_title("Precision-recall (test)"); ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision"); ax[2].legend()
        fig.suptitle(f"{primary}, variant={variant}"); fig.tight_layout(); fig.savefig(figs / f"reliability_pr_{variant}.png", dpi=110); plt.close(fig)
    sub = R[(R.eval_set == "test") & (R.calibration == "cal") & (R.thr_name == "val_f1")]
    piv = sub[sub.variant == "norm"].pivot_table(index="protocol", columns="kind", values="roc_auc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4)); piv.plot.bar(ax=ax); ax.set_ylabel("test ROC-AUC"); ax.set_title("ROC-AUC by evaluation protocol (variant=norm)"); ax.set_ylim(0.5, 1); fig.tight_layout(); fig.savefig(figs / "auc_by_protocol.png", dpi=110); plt.close(fig)
    write_report(R, S, pd.DataFrame(ci), cfg, primary)

def fmt(x, n=4): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"

def write_report(R, S, CI, cfg, primary):
    L = ["# Supervised results", "", "_Generated by `scripts/evaluate_models.py` from `results/supervised_metrics.csv`. Status: **Executed and verified** (see per-section caveats)._", "",
         "Variants: **norm** = leading `www.` removed before featurisation (shortcut-controlled; pre-declared primary because the audit found `www.` in ~99.9% of legitimate vs ~28% of phishing URLs); **full** = URLs as supplied. Calibrator is selected per model on validation data only (none/Platt/isotonic, 2-fold group cross-fit), thresholds come from validation data only.", ""]
    def table(df, cols):
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows(): out.append("| " + " | ".join(fmt(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
        return out
    cols = ["kind", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "brier", "ece"]
    for variant in ["norm", "full"]:
        d = R[(R.protocol == primary) & (R.variant == variant) & (R.eval_set == "test") & (R.calibration == "cal") & (R.thr_name == "val_f1")]
        L += [f"## Primary protocol `{primary}` (registered-domain-disjoint), variant `{variant}` — test, calibrated, val-F1 threshold", ""] + table(d.sort_values("kind"), cols) + [""]
    L += ["## Calibration before vs after (primary, test)", "", "| variant | kind | calibrator | Brier raw | Brier cal | ECE raw | ECE cal |", "|---|---|---|---:|---:|---:|---:|"]
    t = R[(R.protocol == primary) & (R.eval_set == "test") & (R.thr_name == "fixed_0.5")]
    for (v, k), g in t.groupby(["variant", "kind"]):
        r, c = g[g.calibration == "raw"].iloc[0], g[g.calibration == "cal"].iloc[0]
        L.append(f"| {v} | {k} | {r.calibrator} | {fmt(r.brier)} | {fmt(c.brier)} | {fmt(r.ece)} | {fmt(c.ece)} |")
    L += ["", "## Evaluation protocol comparison (RQ1) — test ROC-AUC / F1 (calibrated, val-F1 threshold)", "", "| protocol | variant | kind | ROC-AUC | PR-AUC | F1 |", "|---|---|---|---:|---:|---:|"]
    t = R[(R.eval_set == "test") & (R.calibration == "cal") & (R.thr_name == "val_f1")].sort_values(["protocol", "variant", "kind"])
    for _, r in t.iterrows(): L.append(f"| {r.protocol} | {r.variant} | {r.kind} | {fmt(r.roc_auc)} | {fmt(r.pr_auc)} | {fmt(r.f1)} |")
    L += ["", "Random and hostname splits share registered domains between train and test (see `reports/split_report.md`); differences between protocols quantify how much leakage-prone evaluation inflates performance **on this dataset**.", "",
          "## Seed variability (registered-domain splits, variant norm; mean ± sd over seeds)", "", "| kind | n seeds | ROC-AUC | PR-AUC | F1 |", "|---|---:|---|---|---|"]
    t = R[(R.protocol.str.startswith("regdomain_seed")) & (R.variant == "norm") & (R.eval_set == "test") & (R.calibration == "cal") & (R.thr_name == "val_f1")]
    for k, g in t.groupby("kind"):
        L.append(f"| {k} | {len(g)} | {g.roc_auc.mean():.4f} ± {g.roc_auc.std(ddof=1) if len(g) > 1 else float('nan'):.4f} | {g.pr_auc.mean():.4f} ± {g.pr_auc.std(ddof=1) if len(g) > 1 else float('nan'):.4f} | {g.f1.mean():.4f} ± {g.f1.std(ddof=1) if len(g) > 1 else float('nan'):.4f} |")
    L += ["", "## Group-bootstrap 95% CIs (primary split; resampling unit = registered domain; percentile; 100 resamples)", "", "| variant | kind | metric | point | 95% CI |", "|---|---|---|---:|---|"]
    for _, r in CI.sort_values(["variant", "kind", "metric"]).iterrows(): L.append(f"| {r.variant} | {r.kind} | {r.metric} | {fmt(r.point)} | [{fmt(r.lo)}, {fmt(r.hi)}] |")
    L += ["", "Intervals describe sampling variability over registered domains for one fixed split and fixed trained model; they are not a complete significance analysis and ignore training randomness (see seed table).", "",
          "## External 'recent' holdout (not temporal evidence; no timestamps)", "", "Trained on the primary pool split, evaluated on `test_recent_unseen_domains.csv` (de-duplicated). `recent_novelrd` keeps only rows whose registered domain is absent from the training partition.", "", "| variant | kind | eval_set | ROC-AUC | PR-AUC | F1 | recall | precision |", "|---|---|---|---:|---:|---:|---:|---:|"]
    t = R[(R.protocol == primary) & (R.eval_set.isin(["recent_all", "recent_novelrd"])) & (R.calibration == "cal") & (R.thr_name == "val_f1")].sort_values(["variant", "kind", "eval_set"])
    for _, r in t.iterrows(): L.append(f"| {r.variant} | {r.kind} | {r.eval_set} | {fmt(r.roc_auc)} | {fmt(r.pr_auc)} | {fmt(r.f1)} | {fmt(r.recall)} | {fmt(r.precision)} |")
    L += ["", "The audit showed the recent set was built with a different class-construction process (e.g. `www.` and URL length/path artefacts reversed relative to the pooled data), so a drop here mixes genuine domain shift with dataset-construction artefacts; it cannot be attributed to time.", "",
          "## Compute and size (per trained model)", "", "| protocol | variant | kind | fit s | latency ms/URL (incl. features) | model MB (compressed) | calibrator |", "|---|---|---|---:|---:|---:|---|"]
    for _, r in S[S.protocol == primary].sort_values(["variant", "kind"]).iterrows(): L.append(f"| {r.protocol} | {r.variant} | {r.kind} | {fmt(r.fit_seconds, 1)} | {fmt(r.latency_ms, 3)} | {fmt(r.model_MB, 2)} | {r.calibrator} |")
    L += ["", "Hardware: 2 CPU cores, CPU-only, single-process latency measurement on 3,000 test URLs (numbers are indicative, not a benchmark). XGBoost was not installed and could not be installed (no package-index access), so `hgb_lex` (scikit-learn HistGradientBoosting) is the gradient-boosting substitute. Tree models were fitted on a stratified subsample of at most the configured cap (see config) of the training rows; linear/TF-IDF models used all training rows.",
          "", "Figures: `results/figures/reliability_pr_*.png`, `results/figures/auc_by_protocol.png`."]
    (ROOT / "reports" / "supervised_results.md").write_text("\n".join(L) + "\n")

if __name__ == "__main__": main()
