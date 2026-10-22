"""Generate reports/v2_results.md and reports/v2_final_report.md from results/v2/*.csv|json (no hand-typed numbers in the tables).
Hand-written sections (reports/v2_interpretation.md, v2_limitations.md, v2_roadmap.md, v2_literature_addendum.md) are included verbatim."""
import _bootstrap, json
import numpy as np, pandas as pd
from src.common import ROOT

R = ROOT / "reports"; O = ROOT / "results" / "v2"
NICE = {"v1_hybrid_lr": "v1 hybrid LR", "v1_stack": "v1 stacked", "v1_tfidf_lr": "v1 char TF-IDF", "v1_hgb_lex": "v1 grad.-boosting (lexical)", "xlin_pw1_base": "XLin, no class weight",
        "xlin_pw2_base": "XLin base (w=2)", "xlin_pw2_canon": "XLin + canonical host", "xlin_pw2_full": "XLin + canonical + typosquat", "xlin_pw2_aug": "XLin base + padding augmentation",
        "charcnn": "Char-CNN (NumPy)", "ens_xlin+cnn": "Ensemble XLin(full)+CNN"}
def nm(m): return NICE.get(m, m)
def md(df, floatfmt=".3f"): return df.to_markdown(index=False, floatfmt=floatfmt)

def sec_e1():
    d = pd.read_csv(O / "e1_operating_points.csv"); ci = pd.read_csv(O / "e1_bootstrap_ci.csv"); out = []
    t = d[d.split == "test"]
    out += ["## E1. Recall-first comparison (primary split: registered-domain-disjoint, seed 11, `www.`/scheme normalised)", "",
            "Status: **Executed and verified** (thresholds fixed on the validation/policy partitions only; test labels used for evaluation only). "
            "*URL-level* = every URL counts once; *domain-level* = every registered domain counts once (a few very large platforms otherwise dominate URL-level rates).", ""]
    for op, title in [("fpr1%", "Threshold set so that ~1% of legitimate URLs in the policy partition are flagged"), ("recall95%", "Threshold set to catch 95% of phishing URLs in the policy partition (empirical quantile)"),
                      ("recall99%", "Threshold set to catch 99% of phishing URLs in the policy partition (empirical quantile)")]:
        x = t[t.operating_point == op].copy(); x["model"] = x.model.map(nm)
        out += [f"### {title}", "", md(x[["model", "recall", "domain_recall", "fpr", "domain_fpr", "precision", "accuracy"]].rename(columns={"recall": "URL recall", "domain_recall": "domain recall", "fpr": "URL FPR", "domain_fpr": "domain FPR"})), ""]
    x = t[t.operating_point == "fpr1%"].copy(); x["model"] = x.model.map(nm); out += ["### Threshold-free ranking metrics (test)", "", md(x[["model", "roc_auc", "pr_auc", "pauc_fpr5"]].rename(columns={"pauc_fpr5": "pAUC (FPR<=5%, standardised)"}), ".4f"), ""]
    c = ci[ci.metric == "pr_auc"].copy(); c["model"] = c.model.map(nm); out += ["PR-AUC with 95% cluster-bootstrap CIs (200 resamples of registered domains; percentile method):", "", md(c[["model", "point", "lo", "hi"]], ".4f"), ""]
    p = pd.read_csv(O / "e1_paired_differences.csv"); p["model_a"] = p.model_a.map(nm); p["model_b"] = p.model_b.map(nm)
    p["CI excludes 0"] = np.where((p.lo > 0) | (p.hi < 0), "yes", "no"); out += ["### Paired differences (same resampled domains; a minus b)", "", md(p[["model_a", "model_b", "metric", "diff_a_minus_b", "lo", "hi", "CI excludes 0"]], ".4f"), "",
            "Reading note: 200 resamples and no multiple-comparison correction; treat CIs that barely exclude 0 as suggestive only.", ""]
    r = d[(d.operating_point == "fpr1%") & (d.split != "test")].copy(); r["model"] = r.model.map(nm)
    out += ["### External 'recent' set (NOT temporal evidence; different source and class construction)", "", md(r[["split", "model", "roc_auc", "pr_auc", "recall", "fpr"]]), ""]
    k = pd.read_csv(O / "e1_calibration.csv"); k["model"] = k.model.map(nm); out += ["### Calibration (validation-selected calibrator)", "", md(k, ".4f"), ""]
    return out

def sec_e7():
    f = O / "e7_seed_replication.csv"
    if not f.exists(): return ["## E7. Seed replication", "", "**Not executed** (results/v2/e7_seed_replication.csv missing).", ""]
    d = pd.read_csv(f); out = ["## E7. Replication on three registered-domain-disjoint split seeds (11, 22, 33)", "", "Status: **Executed and verified**. Each seed has its own split, models, validation-fitted ensemble and policy-derived thresholds; mean ± sd over seeds (sd is over only 3 splits). The canon-only ablation exists for seed 11 only.", ""]
    for op, title in [("fpr1%", "Threshold at ~1% policy FPR"), ("recall95%", "Threshold at 95% policy recall")]:
        x = d[d.operating_point == op]; g = x.groupby("model").agg(n_seeds=("seed", "nunique"), pr_auc_m=("pr_auc", "mean"), pr_auc_s=("pr_auc", "std"), rec_m=("recall", "mean"), rec_s=("recall", "std"),
            drec_m=("domain_recall", "mean"), drec_s=("domain_recall", "std"), fpr_m=("fpr", "mean"), fpr_s=("fpr", "std"), dfpr_m=("domain_fpr", "mean"), dfpr_s=("domain_fpr", "std")).reset_index()
        pm = lambda a, b: g[a].map("{:.3f}".format) + " ± " + g[b].fillna(0).map("{:.3f}".format)
        t = pd.DataFrame({"model": g.model.map(nm), "seeds": g.n_seeds, "PR-AUC": pm("pr_auc_m", "pr_auc_s"), "URL recall": pm("rec_m", "rec_s"), "domain recall": pm("drec_m", "drec_s"), "URL FPR": pm("fpr_m", "fpr_s"), "domain FPR": pm("dfpr_m", "dfpr_s")})
        out += [f"### {title}", "", t.to_markdown(index=False), ""]
    return out

def sec_e2():
    d = pd.read_csv(O / "e2_recall_control.csv"); out = ["## E2. Recall-controlled thresholds", "",
        "Status: **Executed and verified** on the primary split. Calibration = one random phishing URL per registered domain from the policy partition (200 random draws); test = the registered-domain-disjoint test partition. "
        "'Violation' = the draw's threshold gives test *domain-level* recall below the target. The guarantee concerns the true recall; the test partition is a finite sample, so violation rates also contain test-sampling noise (about ±0.001-0.003 in recall at these sizes). delta = 0.1.", ""]
    big = d[(d.model == "ens_xlin+cnn") & (d.n_cal == d.n_cal.max())].copy(); big["method"] = big.method.map({"naive_quantile": "naive quantile", "ltt_delta0.1": "finite-sample bound"})
    out += ["### Large calibration set (all calibration domains), ensemble model", "", md(big[["alpha", "method", "test_domain_recall_mean", "test_violation_rate", "test_mean_shortfall_when_violated", "test_fpr_mean", "recent_domain_recall_mean", "recent_fpr_mean"]].rename(columns={"alpha": "miss budget", "test_domain_recall_mean": "test domain recall", "test_violation_rate": "violation rate", "test_mean_shortfall_when_violated": "mean shortfall", "test_fpr_mean": "test URL FPR", "recent_domain_recall_mean": "recent domain recall", "recent_fpr_mean": "recent URL FPR"}), ".4f"), ""]
    sm = d[(d.model == "ens_xlin+cnn") & (d.alpha.isin([0.05, 0.01]))].copy(); sm["method"] = sm.method.map({"naive_quantile": "naive quantile", "ltt_delta0.1": "finite-sample bound"})
    out += ["### Scarce calibration labels (number of labelled phishing domains), ensemble model", "", md(sm[["alpha", "n_cal", "method", "infeasible_rate", "test_domain_recall_mean", "test_violation_rate", "test_fpr_mean"]].rename(columns={"alpha": "miss budget", "n_cal": "n calibration domains", "infeasible_rate": "infeasible (flag everything)", "test_domain_recall_mean": "test domain recall", "test_violation_rate": "violation rate", "test_fpr_mean": "test URL FPR"}), ".4f"), "",
            "Where the finite-sample bound is infeasible (too few calibration positives for the requested budget), the rule returns 'flag everything' (FPR 1.0): the honest outcome when labels are too scarce.", ""]
    a = d[(d.alpha == 0.05) & (d.n_cal == d.n_cal.max())].copy(); a["model"] = a.model.map(nm); a["method"] = a.method.map({"naive_quantile": "naive quantile", "ltt_delta0.1": "finite-sample bound"})
    out += ["### Other models (miss budget 0.05, all calibration domains)", "", md(a[["model", "method", "test_domain_recall_mean", "test_violation_rate", "test_fpr_mean"]], ".4f"), ""]
    r = pd.read_csv(O / "e2b_recalibration_on_target_feedback.csv"); r = r[r.model == "ens_xlin+cnn"]
    out += ["### Under shift: external 'recent' set; source-calibrated vs recalibrated with labelled target-domain feedback (ensemble)", "",
            "Evaluation half and calibration half of the recent set are disjoint by registered domain (100 random halvings). Source-calibrated thresholds do not depend on the target sample size (repeated rows).", "", md(r.rename(columns={"n_cal_target": "n target calibration domains"}), ".4f"), ""]
    return out

def sec_e3():
    c = pd.read_csv(O / "e3_cascade.csv"); info = json.load(open(O / "e3_info.json")); c = c[c.split == "test"]
    out = ["## E3. Latency-aware deferral cascade (analysis only)", "", f"Status: **Simulated / analysis only.** Stage 1 = ensemble, measured latency about {info['stage1_latency_ms_per_url']:.2f} ms per URL on this CPU (including feature extraction). "
           "Stage 2 (DNS/certificate/page enrichment) is **not implemented or evaluated** because the supplied data has no such fields; its latency is an assumption, and 'hypothetical recall if stage 2 were perfect' is an upper bound, not a result.", "",
           md(c[["alpha_target_miss", "lat2_ms_assumed", "frac_auto_allow", "frac_deferred", "frac_auto_block", "phish_in_auto_allow", "recall_if_deferred_allowed", "recall_if_deferred_blocked", "hypothetical_recall_if_stage2_perfect", "mean_latency_ms"]].rename(columns={"alpha_target_miss": "miss budget", "lat2_ms_assumed": "assumed stage-2 ms"}), ".4f"), ""]
    return out

def sec_e4():
    d = pd.read_csv(O / "e4_leakage_demo.csv"); return ["## E4. Evaluation-protocol demonstration (oversampling before splitting)", "", "Status: **Executed**, one 150k-URL random sample, one seed, char TF-IDF logistic regression, threshold 0.5. Oversampling here is plain random duplication of phishing rows (the simplest form of the leakage mechanism); SMOTE interpolation was not run. Settings A and B/C/D have different test prevalence, so compare **recall** and the leakage diagnostics, not PR-AUC/accuracy.", "", md(d, ".4f"), ""]

def sec_e5():
    e = json.load(open(O / "e5_explanations.json")); ex = e.pop("example")
    rows = [("exact linear explanation (incl. feature extraction)", e["xlin_exact_explanation_ms_per_url"]), ("prediction only, same model (for reference)", e["xlin_prediction_only_ms_per_url"]), ("occlusion explanation (segment removal, batched)", e["xlin_occlusion_ms_per_url"]), ("char-CNN gradient x activation saliency", e["charcnn_saliency_ms_per_url"])]
    out = ["## E5. Explanation cost and faithfulness", "", "Status: **Executed** on 300 random test phishing URLs (CPU, single process; milliseconds per URL). SHAP/LIME are not installed offline, so they were **not** compared.", "",
           pd.DataFrame(rows, columns=["method", "ms per URL"]).to_markdown(index=False, floatfmt=".3f"), "",
           f"Additivity check: maximum absolute difference between the explanation's total and the model logit over 100 URLs = {e['max_abs_error_additive_decomposition_vs_logit']:.2e}.", "",
           "Faithfulness proxy (mean drop in model logit when the single top-ranked URL segment is deleted; higher = the attribution found a more influential segment):", "",
           pd.DataFrame([("XLin: segment ranked top by exact char attribution", e["xlin_logit_drop_removing_top_segment_by_exact_attribution"]), ("XLin: best single segment found by occlusion (upper bound)", e["xlin_logit_drop_removing_best_segment_by_occlusion"]),
                         ("XLin: random segment", e["xlin_logit_drop_removing_random_segment"]), ("CNN: segment ranked top by saliency", e["charcnn_logit_drop_removing_top_segment_by_saliency"]), ("CNN: best single segment (upper bound)", e["charcnn_logit_drop_best_single_segment"])],
                        columns=["rule", "mean logit drop"]).to_markdown(index=False, floatfmt=".3f"), "",
           "Caveat: the XLin character attribution covers only the character n-gram block; lexical and typosquat features are not character-local and appear only in the exact term list. Contributions of standardised features are relative to the training mean (a 'not a known brand' term can push toward phishing). Attributions are associations learned by the model, not causal effects.", "",
           f"Example (test phishing URL `{ex['url']}`, risk {ex['risk']:.3f}); top terms toward phishing: " + "; ".join(f"{a}:{b} ({d:+.2f})" for a, b, c, d in ex["toward_phishing"][:4]) + ".", ""]
    return out

def sec_e6():
    a = pd.read_csv(O / "e6_obfuscation_attacks.csv"); a["model"] = a.model.map(nm); t = pd.read_csv(O / "e6_typosquat_synthetic.csv"); t["model"] = t.model.map(nm)
    out = ["## E6. Obfuscation and typosquat robustness (**SIMULATED**, non-adaptive, authored by us)", "",
           "### Evasion on 6,000 real test phishing URLs (same thresholds as E1 ~1% policy FPR; recall of the attacked URLs)", "", md(a[["attack", "model", "recall_clean", "recall_attacked"]], ".3f"), "",
           "Digit-for-letter homoglyphs *raise* detection on this data (digits in hostnames are a phishing cue here), so they are not an evasion for these models. Benign path padding is the effective evasion. "
           "The augmentation model was trained with padding segments sampled from training legitimate URLs; the 'unseen vocabulary' attack uses a hand-written padding string that no augmentation used.", "",
           "### Synthetic typosquats of training-derived brand domains (bare `/` path; mutations authored by us; labels are synthetic by construction)", "",
           md(t[t.mutation == "ALL"][["model", "n", "detection_rate", "detection_at_5pct_control_flag_rate", "control_genuine_brand_flag_rate", "control_unseen_legit_domain_flag_rate"]], ".3f"), "",
           "Detection at the standard threshold is not comparable across models when their flag rates on *legitimate* bare domains differ; the 'at 5% control flag rate' column fixes the false-flag rate on unseen legitimate domains. "
           "Per-mutation detection at that matched rate:", "", t[t.mutation != "ALL"].pivot_table(index="mutation", columns="model", values="detection_at_5pct_control_flag_rate").round(3).to_markdown(), "",
           "Limits: brands are the 2,000 most frequent near-purely-legitimate training registered domains, not an external popularity list; 'brand_as_subdomain' hosts also contain phishing-like tokens, so near-100% detection there is confounded.", ""]
    return out

def sec_cost():
    rows = []
    for f in sorted((O / "predictions").glob("regdomain_seed11__*.json")):
        j = json.load(open(f)); rows.append({"model": nm(j.get("variant", "charcnn")), "fit seconds": j.get("fit_seconds"), "ms per URL (incl. features)": j.get("latency_ms_per_url_incl_features", j.get("latency_ms_per_url")), "model size (MB, compressed)": (j.get("model_bytes_compressed") or 0) / 2**20 or None, "parameters": j.get("n_params")})
    return ["## Computational cost (CPU, 2 cores; indicative)", "", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), ""]

def main():
    head = ["# v2 results (generated)", "", "_Generated by `scripts/v2_make_reports.py` from `results/v2/`. Every number in the tables below is read from those files._", "",
            "| component | status |", "|---|---|", "| Recall-first comparison, class weighting, ablations (E1) | Executed and verified |", "| Replication on 3 split seeds (E7) | Executed and verified |",
            "| Recall-controlled thresholds (E2) | Executed and verified (guarantee tested on domain-disjoint data; shift behaviour reported) |", "| Latency-aware cascade (E3) | Simulated / analysis only (stage 2 not implemented) |",
            "| Leakage demonstration (E4) | Executed (single seed) |", "| Explanation cost/faithfulness (E5) | Executed (no SHAP/LIME comparison) |", "| Obfuscation / typosquat robustness (E6) | Simulated |",
            "| Transformers/BERT; DNS/certificate/DOM fusion; temporal evaluation | **Not executed / not supported by the available data** |", ""]
    body = head + sec_e1() + sec_e7() + sec_e2() + sec_e3() + sec_e4() + sec_e5() + sec_e6() + sec_cost()
    (R / "v2_results.md").write_text("\n".join(body) + "\n")
    parts = ["# Phishing URL detection v2: recall-first, domain-aware, robust and explainable (final report)", "",
             "Status labels used throughout: **Executed and verified**, **Simulated**, **Not executed**, **Not supported by the available data**. Dataset: the supplied archive (SHA-256 `75498f52a089e4fe...`), registered-domain-disjoint protocol inherited from v1 (see `reports/final_research_report.md` for the v1 audit).", ""]
    for name in ["v2_results.md", "v2_interpretation.md", "v2_limitations.md", "v2_literature_addendum.md", "v2_roadmap.md"]:
        f = R / name
        parts += [(f.read_text() if f.exists() else f"_{name} is missing._"), "", "---", ""]
    (R / "v2_final_report.md").write_text("\n".join(parts)); print("v2 reports written")

if __name__ == "__main__": main()
