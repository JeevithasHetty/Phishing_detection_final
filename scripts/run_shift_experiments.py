"""Distribution-shift experiments: OBSERVED shifts (held-out domains/sources/external set) and SIMULATED shifts
(prevalence, covariate perturbation, evasion padding, input-preserving concept drift). Monitoring detectors included."""
import _bootstrap, argparse, hashlib
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from src.common import ROOT, load_config, write_json
from src.evaluation.predictions import load_calibrated, partitions
from src.evaluation.metrics import classification_metrics, threshold_for_cost
from src.features.lexical import featurize, FEATURE_NAMES
from src.monitoring.drift import monitor_window
from src.decision.costs import CostModel

SAVED = ["lr_lex", "hgb_lex", "tfidf_lr", "hybrid_lr"]

def rowrec(name, kind_, y, p, thr, cm, p_ref, X_ref, X_cur, extra=None):
    r = monitor_window(p_ref, p, y, thr, cm, X_ref, X_cur, FEATURE_NAMES)
    fd = r.pop("feature_drift", None)
    r.update(name=name, kind=kind_, max_feature_psi=fd["max_psi"] if fd else None, mean_feature_psi=fd["mean_psi"] if fd else None,
             top_drift_features=str(fd["top_drift_features"][:3]) if fd else None)
    c = classification_metrics(y, p, thr); r.update(roc_auc=c["roc_auc"], pr_auc=c["pr_auc"], f1=c["f1"], precision=c["precision"], recall=c["recall"], brier=c["brier"])
    r.update(extra or {}); return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30000); a = ap.parse_args()
    cfg = load_config(); c = cfg["cost"]; cm = CostModel(c["fn"], c["fp"], c["inspect"]); proto = f"regdomain_seed{cfg['seeds'][0]}"
    best = None
    for k in SAVED:
        D = load_calibrated(proto, "norm", k, cfg); s = average_precision_score(D["parts"]["val"]["label"].values, D["cal_val"])
        if best is None or s > best[0]: best = (s, k, D)
    _, kind, D = best; model = joblib.load(ROOT / "artifacts" / "models" / f"{proto}__norm__{kind}.joblib"); cal = D["calibrator"]
    parts = D["parts"]; yv, pv = parts["val"]["label"].values, D["cal_val"]
    thr = threshold_for_cost(yv, pv, cm.fn, cm.fp)   # val-tuned cost threshold, fixed for every shift scenario
    lexp = np.load(ROOT / cfg["processed_dir"] / "lex_norm_pool.npy"); lexr = np.load(ROOT / cfg["processed_dir"] / "lex_norm_rec.npy")
    Xref = lexp[parts["val"]["pos"].values]
    rows = []
    # ---------- OBSERVED shifts ----------
    rows.append(rowrec("reference: val (same protocol)", "observed", yv, pv, thr, cm, pv, Xref, Xref))
    te = parts["test"]; rows.append(rowrec("registered-domain-disjoint test", "observed", te["label"].values, D["cal_test"], thr, cm, pv, Xref, lexp[te["pos"].values]))
    rec = D["recent"]; nov = D["recent_novel"]
    rows.append(rowrec("external 'recent' set (all rows)", "observed", rec["label"].values, D["cal_recent"], thr, cm, pv, Xref, lexr))
    rows.append(rowrec("external 'recent' set (novel registered domains only)", "observed", rec["label"].values[nov], D["cal_recent"][nov], thr, cm, pv, Xref, lexr[nov]))
    for sp in ["source_PhiUSIIL_to_Mendeley-450K", "source_Mendeley-450K_to_PhiUSIIL"]:
        try:
            S = load_calibrated(sp, "norm", kind if kind in ("lr_lex", "hgb_lex", "tfidf_lr") else "tfidf_lr", cfg)
            rows.append(rowrec(f"cross-source {sp[7:]}", "observed", S["parts"]["test"]["label"].values, S["cal_test"], threshold_for_cost(S["parts"]["val"]["label"].values, S["cal_val"], cm.fn, cm.fp), cm, S["cal_val"], lexp[S["parts"]["val"]["pos"].values], lexp[S["parts"]["test"]["pos"].values]))
        except FileNotFoundError: pass
    # ---------- SIMULATED shifts on a fixed sample of the test partition ----------
    rng = np.random.default_rng(0); idx = rng.choice(len(te), min(a.n, len(te)), replace=False)
    urls, y = te["url"].values[idx], te["label"].values[idx].astype(int); base_X = lexp[te["pos"].values[idx]]
    score = lambda u: np.clip(cal.transform(model.predict_proba(list(u))), 0, 1)
    p0 = D["cal_test"][idx]
    # (1) prevalence shift: resample to target prevalence; also report prior-corrected posterior
    pi0 = float(yv.mean())
    for pi in [0.05, 0.28, 0.5, 0.8]:
        r2 = np.random.default_rng(1); i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0); n = 20000
        sel = np.r_[r2.choice(i1, int(n * pi)), r2.choice(i0, n - int(n * pi))]
        rows.append(rowrec(f"prevalence {pi:.2f}", "sim:prior_shift", y[sel], p0[sel], thr, cm, pv, Xref, base_X[sel], {"param": pi}))
        odds = p0[sel] / np.clip(1 - p0[sel], 1e-9, None) * (pi / (1 - pi)) / (pi0 / (1 - pi0)); pc = odds / (1 + odds)
        rows.append(rowrec(f"prevalence {pi:.2f} (prior-corrected, oracle prior)", "sim:prior_shift_corrected", y[sel], pc, thr, cm, pv, Xref, base_X[sel], {"param": pi}))
    # (2) covariate shift: tracking parameters appended to every URL (label-independent)
    def track(u, k, i): return u + ("&" if "?" in u else "?") + "&".join(f"utm_{j}=" + hashlib.md5(f"{i}{j}".encode()).hexdigest()[:10] for j in range(k))
    for k in [1, 3]:
        u2 = [track(u, k, i) for i, u in enumerate(urls)]; rows.append(rowrec(f"append {k} tracking params to all URLs", "sim:covariate", y, score(u2), thr, cm, pv, Xref, featurize(u2, True), {"param": k}))
    # (3) evasion: attacker pads PHISHING URLs with benign-looking path tokens
    PAD = "/wiki/Main_Page/news/article/index.html"
    for k in [1, 2, 4]:
        u2 = [u + PAD * k if yy == 1 else u for u, yy in zip(urls, y)]; rows.append(rowrec(f"pad phishing URLs with benign path x{k}", "sim:evasion", y, score(u2), thr, cm, pv, Xref, featurize(u2, True), {"param": k}))
    # (4) concept drift: inputs identical, labels of the subset 'host has >=2 digits' flipped (P(y|x) changes, P(x) does not)
    hostdig = base_X[:, FEATURE_NAMES.index("n_digits_host")] >= 2; yc = np.where(hostdig, 1 - y, y)
    rows.append(rowrec("concept drift: flip labels where host digits>=2 (inputs unchanged)", "sim:concept_drift", yc, p0, thr, cm, pv, Xref, base_X, {"param": float(hostdig.mean())}))
    R = pd.DataFrame(rows); R.to_csv(ROOT / "results" / "shift_results.csv", index=False)
    write_json({"classifier": kind, "protocol": proto, "variant": "norm", "threshold": thr, "calibrator": D["calibrator_name"], "n_sim": int(len(idx))}, ROOT / "results" / "shift_run_info.json")
    figs = ROOT / "results" / "figures"; figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5)); d = R[R.kind != "sim:prior_shift_corrected"]
    ax.barh(range(len(d)), d.cost_per_1k, color=["C0" if k == "observed" else "C1" for k in d.kind]); ax.set_yticks(range(len(d))); ax.set_yticklabels(d.name, fontsize=6); ax.invert_yaxis(); ax.set_xlabel("cost per 1k URLs (fixed val-tuned threshold; ASSUMED costs)"); ax.set_title("Observed (blue) vs simulated (orange) shifts"); fig.tight_layout(); fig.savefig(figs / "shift_cost.png", dpi=110); plt.close(fig)
    write_report(R, kind, thr)

def write_report(R, kind, thr):
    f = lambda x, n=3: "n/a" if x is None or pd.isna(x) else f"{x:.{n}f}"
    L = ["# Distribution-shift results", "", "_Generated by `scripts/run_shift_experiments.py` from `results/shift_results.csv`._", "",
         f"Classifier: `{kind}` (best validation AP among saved models), primary registered-domain split, variant `norm`; threshold fixed at the validation cost-optimal value {thr:.4f}; costs are ASSUMED (fn=10, fp=1). **Observed** rows use real held-out data; **sim:** rows are controlled synthetic perturbations (status: *Simulated*). Chronological shift: **not supported** (no timestamps).", "",
         "| scenario | kind | n | ROC-AUC | PR-AUC | F1 | precision | recall | ECE | Brier | cost/1k | score PSI | max feature PSI | input alert | label (ECE) alert |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r['name']} | {r.kind} | {r.n} | {f(r.roc_auc)} | {f(r.pr_auc)} | {f(r.f1)} | {f(r.precision)} | {f(r.recall)} | {f(r.ece)} | {f(r.brier)} | {f(r.cost_per_1k, 1)} | {f(r.score_psi)} | {f(r.max_feature_psi)} | {r.input_drift_alert} | {r.label_alert} |")
    L += ["", "## Reading the table", "",
          "- PSI > 0.25 is a conventional rule-of-thumb alert level, not a validated threshold for this domain.",
          "- **Concept-drift row**: inputs are identical to the clean sample, so the feature/score PSI detectors cannot fire by construction, while label-based error/ECE/cost degrade. This demonstrates the documented limitation of input-only drift detection; only label feedback exposes it.",
          "- Prevalence rows: ranking metrics (AUC) are unaffected by prior shift but precision, calibration and cost change; the 'prior-corrected' rows use the *true* new prevalence (an oracle that a deployed system would have to estimate).",
          "- Evasion rows are a transparent simulation of one simple attacker strategy (benign path padding); they do not demonstrate robustness to adaptive adversaries.",
          "- The external 'recent' set mixes genuine domain shift with dataset-construction artefacts (see dataset audit); it must not be read as temporal drift.", "", "Figure: `results/figures/shift_cost.png`."]
    (ROOT / "reports" / "distribution_shift_results.md").write_text("\n".join(L) + "\n")

if __name__ == "__main__": main()
