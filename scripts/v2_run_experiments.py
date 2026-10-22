"""v2 experiments (primary protocol: registered-domain-disjoint split, seed 11, variant 'norm').
  e1  recall-first comparison of v1 baselines vs XLin ablations vs char-CNN vs ensemble (thresholds from policy/val ONLY)
  e2  recall control: naive quantile vs finite-sample (LTT) threshold; in-distribution coverage, small-n regime, shift, recalibration
  e3  latency-aware deferral cascade analysis (stage 2 NOT implemented: no DNS/cert/DOM data)
  e4  leakage demonstration: oversampling before splitting vs domain-aware splitting
  e5  explanation cost and faithfulness (exact linear decomposition vs occlusion vs CNN saliency)
  e6  SIMULATED obfuscation / typosquat robustness
Usage: python scripts/v2_run_experiments.py [--only e1 e2 ...]
"""
import _bootstrap, argparse, json, re, time
import numpy as np, pandas as pd, joblib
from scipy.special import logit, expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.common import ROOT, load_config, write_json, sha256_file
from src.evaluation.predictions import partitions, select_calibrator
from src.evaluation.metrics import classification_metrics, threshold_for_f1, group_bootstrap_ci, ece
from src.decision import recall_control as RC
from src.decision.cascade import precision_threshold, cascade_outcome

CFG = load_config(); PROTO = "regdomain_seed11"; OUT = ROOT / "results" / "v2"; V1 = ROOT / "results" / "predictions"; V2 = OUT / "predictions"; ART = ROOT / "artifacts"
V2_MODELS = ["xlin_pw1_base", "xlin_pw2_base", "xlin_pw2_canon", "xlin_pw2_full", "xlin_pw2_aug"]
ENS_PARTS = ["xlin_pw2_full", "charcnn"]; ENS = "ens_xlin+cnn"
DELTA = 0.1

def _clip(p): return np.clip(p, 1e-6, 1 - 1e-6)

def load_all(write=True):
    parts, rec = partitions(PROTO, CFG); S = {}
    def add(name, f): z = np.load(f); S[name] = {k: z[k].astype(float) for k in ["p_val", "p_policy", "p_test", "p_recent"]}
    for k in ["tfidf_lr", "hybrid_lr", "stack", "hgb_lex"]: add("v1_" + k, V1 / f"{PROTO}__norm__{k}.npz")
    for v in V2_MODELS:
        if (V2 / f"{PROTO}__{v}.npz").exists() or PROTO == "regdomain_seed11": add(v, V2 / f"{PROTO}__{v}.npz")      # replication seeds omit the canon-only ablation
    add("charcnn", V2 / f"{PROTO}__charcnn_pw2.npz")
    novel = np.load(V1 / f"{PROTO}__norm__hybrid_lr.npz")["recent_novel"].astype(bool)
    # ensemble = logistic regression on the logits of the component models, fitted on VALIDATION only
    yv = parts["val"].label.values.astype(int)
    comb = LogisticRegression(C=1e3, max_iter=500).fit(np.column_stack([logit(_clip(S[m]["p_val"])) for m in ENS_PARTS]), yv)
    S[ENS] = {k: comb.predict_proba(np.column_stack([logit(_clip(S[m][k])) for m in ENS_PARTS]))[:, 1] for k in ["p_val", "p_policy", "p_test", "p_recent"]}
    if write: write_json({"components": ENS_PARTS, "coef": comb.coef_[0].tolist(), "intercept": float(comb.intercept_[0]), "fitted_on": "validation partition only"}, OUT / "ensemble.json")
    return parts, rec, S, novel

def y_of(parts, k): return parts[k].label.values.astype(int)

def op_thresholds(P, parts):
    yp, yv = y_of(parts, "policy"), y_of(parts, "val"); neg, pos = P["p_policy"][yp == 0], P["p_policy"][yp == 1]; t = {}
    for f in [0.005, 0.01, 0.05]: t[f"fpr{f * 100:g}%"] = float(np.nextafter(np.quantile(neg, 1 - f, method="higher"), np.inf))
    for a in [0.05, 0.01]: t[f"recall{(1 - a) * 100:g}%"] = RC.naive_threshold(pos, a)
    t["val_F1_opt"] = threshold_for_f1(yv, P["p_val"]); return t

# ------------------------------------------------------------------------------------------------------------------ E1
def e1(parts, rec, S, novel):
    yt, rdt, yr = y_of(parts, "test"), parts["test"].rd.values, rec.label.values.astype(int); rows = []; boots = []; TH = {}
    for name, P in S.items():
        th = op_thresholds(P, parts); TH[name] = th
        for split, y, p, rd in [("test", yt, P["p_test"], rdt), ("recent_all", yr, P["p_recent"], rec.rd.values), ("recent_novel_domains", yr[novel], P["p_recent"][novel], rec.rd.values[novel])]:
            base = dict(model=name, split=split, n=int(len(y)), n_pos=int(y.sum()), roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)), pauc_fpr5=RC.partial_auc(y, p, 0.05))
            for tn, t in th.items():
                m = classification_metrics(y, p, t)
                flag = (p >= t).astype(float)       # macro (per-registered-domain) rates: each domain counts once, so a few huge platforms cannot dominate
                dom_rec = float(pd.Series(flag[y == 1]).groupby(rd[y == 1]).mean().mean()); dom_fpr = float(pd.Series(flag[y == 0]).groupby(rd[y == 0]).mean().mean())
                rows.append({**base, "operating_point": tn, "threshold": t, "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                             "fpr": m["fp"] / max(1, m["fp"] + m["tn"]), "domain_recall": dom_rec, "domain_fpr": dom_fpr, "fn": m["fn"], "fp": m["fp"], "missed_phishing_frac": m["fn"] / max(1, m["tp"] + m["fn"])})
        # group-bootstrap CIs on the test split (resampling unit = registered domain, 200 resamples, percentile 95%)
        p = P["p_test"]
        for mname, fn in [("pr_auc", lambda y_, p_: average_precision_score(y_, p_)), ("pauc_fpr5", lambda y_, p_: RC.partial_auc(y_, p_, 0.05)),
                          ("recall@policy_fpr1%", lambda y_, p_, t=th["fpr1%"]: float((p_[y_ == 1] >= t).mean())),
                          ("fpr@policy_recall99%", lambda y_, p_, t=th["recall99%"]: float((p_[y_ == 0] >= t).mean()))]:
            lo, hi, pt = group_bootstrap_ci(yt, p, rdt, fn, n_boot=200, seed=0); boots.append(dict(model=name, metric=mname, point=pt, lo=lo, hi=hi, n_boot=200, unit="registered_domain"))
    pd.DataFrame(rows).to_csv(OUT / "e1_operating_points.csv", index=False); pd.DataFrame(boots).to_csv(OUT / "e1_bootstrap_ci.csv", index=False)
    # paired group bootstrap of differences (same resampled domains for both models)
    pairs = [(ENS, "v1_stack"), (ENS, "v1_hybrid_lr"), ("xlin_pw2_full", "v1_hybrid_lr"), ("charcnn", "v1_hybrid_lr"), ("charcnn", "v1_tfidf_lr"),
             ("xlin_pw2_base", "xlin_pw1_base"), ("xlin_pw2_canon", "xlin_pw2_base"), ("xlin_pw2_full", "xlin_pw2_canon"), ("xlin_pw2_aug", "xlin_pw2_base"), ("xlin_pw2_full", "xlin_pw2_base")]
    uniq, inv = np.unique(rdt, return_inverse=True); order = np.argsort(inv, kind="stable"); st = np.r_[0, np.cumsum(np.bincount(inv))]; mem = [order[st[i]:st[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(1); draws = [np.concatenate([mem[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(200)]; prow = []
    for a, b in pairs:
        for mname in ["pr_auc", "recall@policy_fpr1%", "fpr@policy_recall99%"]:
            def met(name, idx):
                y, p = yt[idx], S[name]["p_test"][idx]
                if mname == "pr_auc": return average_precision_score(y, p)
                if mname.startswith("recall"): return float((p[y == 1] >= TH[name]["fpr1%"]).mean())
                return float((p[y == 0] >= TH[name]["recall99%"]).mean())
            d = np.array([met(a, ix) - met(b, ix) for ix in draws]); pt = met(a, np.arange(len(yt))) - met(b, np.arange(len(yt)))
            prow.append(dict(model_a=a, model_b=b, metric=mname, diff_a_minus_b=pt, lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), n_boot=200))
    pd.DataFrame(prow).to_csv(OUT / "e1_paired_differences.csv", index=False)
    # calibration of the new probabilistic models (validation-selected calibrator)
    cal = []
    for name in ["v1_hybrid_lr", "v1_stack", "xlin_pw2_full", "charcnn", ENS]:
        P = S[name]; c, cn, _ = select_calibrator(P["p_val"], y_of(parts, "val"), parts["val"].rd.values)
        for split, y, p in [("test", yt, P["p_test"]), ("recent_all", yr, P["p_recent"])]:
            q = np.clip(c.transform(p), 0, 1)
            cal.append(dict(model=name, split=split, calibrator=cn, ece_raw=ece(y, p), ece_cal=ece(y, q), brier_raw=float(np.mean((p - y) ** 2)), brier_cal=float(np.mean((q - y) ** 2))))
    pd.DataFrame(cal).to_csv(OUT / "e1_calibration.csv", index=False)
    print("e1 done", flush=True)

# ------------------------------------------------------------------------------------------------------------------ E2
def e2(parts, rec, S, novel, R=200):
    yp, rdp, yt, rdt, yr, rdr = y_of(parts, "policy"), parts["policy"].rd.values, y_of(parts, "test"), parts["test"].rd.values, rec.label.values.astype(int), rec.rd.values
    pol_idx = np.flatnonzero(yp == 1); rng = np.random.default_rng(0)
    def draw():   # one random positive per registered domain from the policy partition
        order = rng.permutation(len(pol_idx)); _, first = np.unique(rdp[pol_idx][order], return_index=True); return pol_idx[order][first]
    draws = [draw() for _ in range(R)]; n_dom = len(draws[0])
    def dom_stats(y, rd, mask_pos):
        dom, inv = np.unique(rd[mask_pos], return_inverse=True); return inv, np.bincount(inv)
    rows = []
    for name in ["v1_hybrid_lr", "xlin_pw2_full", "charcnn", ENS]:
        P = S[name]; pt, pr = P["p_test"], P["p_recent"]
        tpos, rpos = yt == 1, yr == 1; inv_t, cnt_t = dom_stats(yt, rdt, tpos); inv_r, cnt_r = dom_stats(yr, rdr, rpos)
        dom_rec = lambda s, inv, cnt, t: float((np.bincount(inv, weights=(s >= t).astype(float)) / cnt).mean())
        assert abs(dom_rec(pt[tpos], inv_t, cnt_t, 0.5) - RC.domain_recall(pt[tpos], rdt[tpos], 0.5)) < 1e-9          # fast path == tested implementation
        for n_cal in [None, 3000, 1000, 300, 100]:
            for alpha in [0.10, 0.05, 0.02, 0.01]:
                for method in ["naive_quantile", "ltt_delta0.1"]:
                    acc = {k: [] for k in ["dr", "ur", "fpr", "r_dr", "r_ur", "r_fpr", "inf"]}
                    for ix in draws:
                        cal = P["p_policy"][ix]
                        if n_cal and n_cal < len(cal): cal = rng.choice(cal, n_cal, replace=False)
                        if method == "naive_quantile": t, feas = RC.naive_threshold(cal, alpha), True
                        else:
                            t, k, _ = RC.ltt_threshold(cal, alpha, DELTA); feas = bool(np.isfinite(t))
                        acc["inf"].append(not feas)
                        acc["dr"].append(dom_rec(pt[tpos], inv_t, cnt_t, t)); acc["ur"].append(float((pt[tpos] >= t).mean())); acc["fpr"].append(float((pt[~tpos] >= t).mean()))
                        acc["r_dr"].append(dom_rec(pr[rpos], inv_r, cnt_r, t)); acc["r_ur"].append(float((pr[rpos] >= t).mean())); acc["r_fpr"].append(float((pr[~rpos] >= t).mean()))
                    tgt = 1 - alpha; dr, rdr_ = np.array(acc["dr"]), np.array(acc["r_dr"])
                    rows.append(dict(model=name, n_cal=int(n_cal or n_dom), alpha=alpha, target_recall=tgt, method=method, reps=R, infeasible_rate=float(np.mean(acc["inf"])),
                                     test_domain_recall_mean=float(dr.mean()), test_violation_rate=float((dr < tgt).mean()), test_mean_shortfall_when_violated=float(np.mean(np.maximum(0, tgt - dr)[dr < tgt])) if (dr < tgt).any() else 0.0,
                                     test_url_recall_mean=float(np.mean(acc["ur"])), test_fpr_mean=float(np.mean(acc["fpr"])),
                                     recent_domain_recall_mean=float(rdr_.mean()), recent_violation_rate=float((rdr_ < tgt).mean()), recent_url_recall_mean=float(np.mean(acc["r_ur"])), recent_fpr_mean=float(np.mean(acc["r_fpr"]))))
        print("e2 model done", name, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "e2_recall_control.csv", index=False)
    # ---- recalibration with labelled target-domain feedback (recent set split by registered domain: calibration half / evaluation half)
    rr = []; rng2 = np.random.default_rng(5); doms = np.unique(rdr)
    for name in ["xlin_pw2_full", ENS]:
        P = S[name]; pr = P["p_recent"]
        for rep in range(100):
            half = set(rng2.choice(doms, len(doms) // 2, replace=False)); incal = np.array([d in half for d in rdr]); ev = ~incal
            cal_pos = np.flatnonzero(incal & (yr == 1)); o = rng2.permutation(len(cal_pos)); _, first = np.unique(rdr[cal_pos][o], return_index=True); cal_idx = cal_pos[o][first]
            evp = ev & (yr == 1); evn = ev & (yr == 0)
            for n_cal in [100, 300, 1000, 3000]:
                ci = rng2.choice(cal_idx, min(n_cal, len(cal_idx)), replace=False)
                for alpha in [0.05, 0.01]:
                    t_tgt, _, _ = RC.ltt_threshold(pr[ci], alpha, DELTA); t_src, _, _ = RC.ltt_threshold(P["p_policy"][draws[rep % R]], alpha, DELTA)     # source-calibrated threshold (policy partition, one positive per domain)
                    for lab, t in [("source_calibrated", t_src), ("target_feedback_calibrated", t_tgt)]:
                        rr.append(dict(model=name, rep=rep, n_cal_target=int(len(ci)), alpha=alpha, threshold_source=lab, feasible=bool(np.isfinite(t)),
                                       eval_domain_recall=RC.domain_recall(pr[evp], rdr[evp], t), eval_url_recall=float((pr[evp] >= t).mean()), eval_fpr=float((pr[evn] >= t).mean())))
    R2 = pd.DataFrame(rr); R2["violated"] = R2.eval_domain_recall < (1 - R2.alpha)
    R2.groupby(["model", "threshold_source", "n_cal_target", "alpha"]).agg(reps=("rep", "count"), feasible_rate=("feasible", "mean"), domain_recall=("eval_domain_recall", "mean"),
        violation_rate=("violated", "mean"), url_recall=("eval_url_recall", "mean"), fpr=("eval_fpr", "mean")).reset_index().to_csv(OUT / "e2b_recalibration_on_target_feedback.csv", index=False)
    print("e2 done", flush=True)

# ------------------------------------------------------------------------------------------------------------------ E3
def e3(parts, rec, S, novel):
    yp, yt, yr = y_of(parts, "policy"), y_of(parts, "test"), rec.label.values.astype(int); P = S[ENS]; rdp = parts["policy"].rd.values
    lat = sum(json.load(open(V2 / f"{PROTO}__{t}.json"))[k] for t, k in [("xlin_pw2_full", "latency_ms_per_url_incl_features"), ("charcnn_pw2", "latency_ms_per_url")])
    rng = np.random.default_rng(0); pos = np.flatnonzero(yp == 1); o = rng.permutation(len(pos)); _, first = np.unique(rdp[pos][o], return_index=True); cal = P["p_policy"][pos[o][first]]
    t_block = precision_threshold(yp, P["p_policy"], 0.99); rows = []
    for alpha in [0.05, 0.02, 0.01]:
        t_allow, k, n = RC.ltt_threshold(cal, alpha, DELTA); t_b = max(t_block, t_allow)
        for split, y, p in [("test", yt, P["p_test"]), ("recent_all", yr, P["p_recent"])]:
            for lat2 in [50.0, 250.0, 1000.0]:
                rows.append({"split": split, "alpha_target_miss": alpha, "t_allow": t_allow, "t_block": t_b, **cascade_outcome(y, p, t_allow, t_b, lat, lat2)})
    pd.DataFrame(rows).to_csv(OUT / "e3_cascade.csv", index=False); write_json({"stage1_latency_ms_per_url": lat, "note": "stage-2 latencies are ASSUMPTIONS; stage 2 is not implemented/evaluated"}, OUT / "e3_info.json")
    print("e3 done", flush=True)

# ------------------------------------------------------------------------------------------------------------------ E4
def e4():
    from src.models.zoo import UrlModel
    pool = pd.read_pickle(ROOT / CFG["processed_dir"] / "pool.pkl").sample(150000, random_state=0).reset_index(drop=True); rng = np.random.default_rng(0)
    def oversample(df):
        pos, neg = df[df.label.astype(int) == 1], df[df.label.astype(int) == 0]; extra = pos.sample(len(neg) - len(pos), replace=True, random_state=0)
        return pd.concat([df, extra]).sample(frac=1, random_state=0).reset_index(drop=True)
    def row_split(df): idx = rng.permutation(len(df)); c = int(len(df) * 0.75); return df.iloc[idx[:c]], df.iloc[idx[c:]]
    def dom_split(df):
        d = df.rd.unique(); tr_d = set(rng.choice(d, int(len(d) * 0.75), replace=False)); m = df.rd.isin(tr_d).values; return df[m], df[~m]
    settings = {"A_oversample_THEN_random_split (leaky)": lambda: row_split(oversample(pool)), "B_random_split_no_oversampling": lambda: row_split(pool),
                "C_domain_split_no_oversampling": lambda: dom_split(pool), "D_domain_split_then_oversample_train_only (correct)": lambda: (lambda a, b: (oversample(a), b))(*dom_split(pool))}
    rows = []
    for name, f in settings.items():
        tr, te = f(); t0 = time.time(); m = UrlModel("tfidf_lr", strip_www=True, seed=42, max_features=100000).fit(tr.url.values, tr.label.values.astype(int))
        p = m.predict_proba(te.url.values); y = te.label.values.astype(int); c = classification_metrics(y, p, 0.5)
        rows.append(dict(setting=name, n_train=len(tr), n_test=len(te), test_pos_frac=float(y.mean()), accuracy=c["accuracy"], precision=c["precision"], recall=c["recall"], f1=c["f1"], pr_auc=c["pr_auc"],
                         test_urls_also_in_train=float(te.url.isin(set(tr.url)).mean()), test_domains_also_in_train=float(te.rd.isin(set(tr.rd)).mean()), fit_seconds=time.time() - t0))
        print("e4", name, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "e4_leakage_demo.csv", index=False)

# ------------------------------------------------------------------------------------------------------------------ models for e5/e6
def load_cnn():
    from src.models.charcnn import CharCNN
    info = json.load(open(V2 / f"{PROTO}__charcnn_pw2.json")); c = info["config"]
    m = CharCNN(max_len=c["max_len"], emb=c["emb"], widths=tuple(c["widths"]), n_filters=c["n_filters"], hidden=c["hidden"], seed=11)
    z = np.load(ART / "v2" / f"{PROTO}__charcnn_pw2.model.npz"); m.P = {k: z[k] for k in z.files}; return m

class Scorer:
    """Score arbitrary URL lists with every model; thresholds come from POLICY-partition predictions of the same model."""
    def __init__(self, S, parts):
        self.xl_full = joblib.load(ART / "v2" / f"{PROTO}__xlin_pw2_full.joblib"); self.xl_base = joblib.load(ART / "v2" / f"{PROTO}__xlin_pw2_base.joblib"); self.xl_aug = joblib.load(ART / "v2" / f"{PROTO}__xlin_pw2_aug.joblib")
        self.v1 = joblib.load(ART / "models" / f"{PROTO}__norm__hybrid_lr.joblib"); self.cnn = load_cnn(); self.ens = json.load(open(OUT / "ensemble.json"))
        self.thr = {n: op_thresholds(S[n], parts)["fpr1%"] for n in ["v1_hybrid_lr", "xlin_pw2_base", "xlin_pw2_aug", "xlin_pw2_full", "charcnn", ENS]}
    def scores(self, urls):
        urls = list(urls); s = {"v1_hybrid_lr": self.v1.predict_proba(urls), "xlin_pw2_base": self.xl_base.predict_proba(urls), "xlin_pw2_aug": self.xl_aug.predict_proba(urls), "xlin_pw2_full": self.xl_full.predict_proba(urls), "charcnn": self.cnn.predict_proba(urls)}
        z = np.column_stack([logit(_clip(s[m])) for m in self.ens["components"]]) @ np.array(self.ens["coef"]) + self.ens["intercept"]; s[ENS] = expit(z); return s

# ------------------------------------------------------------------------------------------------------------------ E5
def segments(text, start=0):
    return [(m.start(), m.end()) for m in re.finditer(r"[a-z0-9]+", text) if m.start() >= start]

def e5(parts, S):
    sc = Scorer(S, parts); te = parts["test"]; rng = np.random.default_rng(0); urls = te[te.label.astype(int) == 1].url.sample(300, random_state=0).values
    xl, cnn = sc.xl_full, sc.cnn; res = {"n_urls": int(len(urls))}
    t0 = time.perf_counter(); [xl.explain(u) for u in urls]; res["xlin_exact_explanation_ms_per_url"] = (time.perf_counter() - t0) / len(urls) * 1000
    t0 = time.perf_counter(); [xl.char_attribution(u) for u in urls]; res["xlin_char_attribution_ms_per_url"] = (time.perf_counter() - t0) / len(urls) * 1000
    t0 = time.perf_counter(); [cnn.explain_chars(u) for u in urls]; res["charcnn_saliency_ms_per_url"] = (time.perf_counter() - t0) / len(urls) * 1000
    t0 = time.perf_counter(); [xl.predict_proba([u]) for u in urls]; res["xlin_prediction_only_ms_per_url"] = (time.perf_counter() - t0) / len(urls) * 1000
    # occlusion baseline: remove each alphanumeric segment and re-score (batched per URL)
    occ_drops, exact_drops, rand_drops, cnn_drops, cnn_best = [], [], [], [], []; t_occ = 0.0; seg_counts = []
    for u in urls:
        text = xl.char_attribution(u)[0]; off = (re.match(r"^[a-z]+://\s*", text).end() if re.match(r"^[a-z]+://\s*", text) else 0); segs = segments(text, off)
        if len(segs) < 2: continue
        seg_counts.append(len(segs)); variants = [text[:s] + text[e:] for s, e in segs]
        t0 = time.perf_counter(); z0 = float(xl.decision_function([u])[0]); drops = z0 - xl.decision_function(variants); t_occ += time.perf_counter() - t0
        _, cs = xl.char_attribution(u); ex = np.array([cs[s:e].sum() for s, e in segs]); occ_drops.append(drops.max()); exact_drops.append(drops[int(np.argmax(ex))]); rand_drops.append(float(drops.mean()))
        t2, cscore, _ = cnn.explain_chars(u); s2 = segments(t2);
        if len(s2) >= 2:
            z0c = logit(_clip(cnn.predict_proba([u])))[0]; v2 = ["http://" + t2[:a] + t2[b:] for a, b in s2]; dc = z0c - logit(_clip(cnn.predict_proba(v2)))
            cnn_drops.append(dc[int(np.argmax([cscore[a:b].sum() for a, b in s2]))]); cnn_best.append(dc.max())
    res.update(xlin_occlusion_ms_per_url=t_occ / max(1, len(seg_counts)) * 1000, mean_segments_per_url=float(np.mean(seg_counts)), n_urls_evaluated=len(seg_counts),
               xlin_logit_drop_removing_top_segment_by_exact_attribution=float(np.mean(exact_drops)), xlin_logit_drop_removing_best_segment_by_occlusion=float(np.mean(occ_drops)),
               xlin_logit_drop_removing_random_segment=float(np.mean(rand_drops)), charcnn_logit_drop_removing_top_segment_by_saliency=float(np.mean(cnn_drops)), charcnn_logit_drop_best_single_segment=float(np.mean(cnn_best)))
    # additivity check on the test sample
    errs = [abs(xl.explain(u)["logit"] - float(xl.decision_function([u])[0])) for u in urls[:100]]; res["max_abs_error_additive_decomposition_vs_logit"] = float(max(errs))
    ex = xl.explain(urls[0], k=6); res["example"] = {"url": urls[0], "risk": ex["risk"], "toward_phishing": ex["toward_phishing"], "toward_legitimate": ex["toward_legitimate"]}
    write_json(res, OUT / "e5_explanations.json"); print("e5 done", flush=True)

# ------------------------------------------------------------------------------------------------------------------ E6
_HG = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5"}

def homoglyph_attack(url, rd, rng):
    sld = rd.split(".")[0]; el = [i for i, c in enumerate(sld) if c in _HG]
    if not el or sld not in url.lower(): return None
    pick = set(rng.choice(el, min(2, len(el)), replace=False)); new = "".join(_HG[c] if i in pick else c for i, c in enumerate(sld))
    i = url.lower().find(sld); return url[:i] + new + url[i + len(sld):]

def mutate_brand(rd, kind, rng):
    sld, _, suf = rd.partition(".")
    if kind == "homoglyph_digit":
        el = [i for i, c in enumerate(sld) if c in _HG]
        if not el: return None
        i = int(rng.choice(el)); return sld[:i] + _HG[sld[i]] + sld[i + 1:] + "." + suf
    if kind == "delete_char": i = int(rng.integers(1, len(sld) - 1)); return sld[:i] + sld[i + 1:] + "." + suf
    if kind == "insert_char": i = int(rng.integers(1, len(sld))); return sld[:i] + "abcdefghijklmnopqrstuvwxyz"[int(rng.integers(0, 26))] + sld[i:] + "." + suf
    if kind == "transpose": i = int(rng.integers(1, len(sld) - 2)); return sld[:i] + sld[i + 1] + sld[i] + sld[i + 2:] + "." + suf
    if kind == "hyphenate": i = int(rng.integers(2, len(sld) - 1)); return sld[:i] + "-" + sld[i:] + "." + suf
    if kind == "tld_swap": return sld + "." + str(rng.choice(["xyz", "top", "info", "online", "site"]))
    if kind == "brand_as_subdomain": return f"{rd}.secure-verify-{int(rng.integers(100, 999))}.{rng.choice(['xyz', 'top', 'info'])}"
    raise ValueError(kind)

def e6(parts, rec, S):
    sc = Scorer(S, parts); te = parts["test"]; rng = np.random.default_rng(0); thr = sc.thr
    ph = te[te.label.astype(int) == 1].sample(frac=1, random_state=0); rows = []; urls0, urls1 = [], []
    for u, rd in zip(ph.url.values, ph.rd.values):
        a = homoglyph_attack(u, rd, rng)
        if a: urls0.append(u); urls1.append(a)
        if len(urls0) >= 6000: break
    b0, b1 = sc.scores(urls0), sc.scores(urls1)
    # evasion 2 (as in v1): benign-looking path padding
    PAD = "/wiki/Main_Page/news/article/index.html"; b2 = sc.scores([u + PAD * 2 for u in urls0])
    PAD2 = "/blog/2021/travel/photos/gallery/summer/holiday/index.html"; b3 = sc.scores([u.rstrip("/") + PAD2 for u in urls0])      # vocabulary NOT used by any augmentation
    for m in thr: rows.append(dict(attack="homoglyph digits in registered-domain label (SIMULATED)", model=m, n=len(urls0), recall_clean=float((b0[m] >= thr[m]).mean()), recall_attacked=float((b1[m] >= thr[m]).mean())))
    for m in thr: rows.append(dict(attack="benign path padding x2 (SIMULATED)", model=m, n=len(urls0), recall_clean=float((b0[m] >= thr[m]).mean()), recall_attacked=float((b2[m] >= thr[m]).mean())))
    for m in thr: rows.append(dict(attack="benign path padding, unseen vocabulary (SIMULATED)", model=m, n=len(urls0), recall_clean=float((b0[m] >= thr[m]).mean()), recall_attacked=float((b3[m] >= thr[m]).mean())))
    pd.DataFrame(rows).to_csv(OUT / "e6_obfuscation_attacks.csv", index=False)
    # synthetic typosquat set around TRAIN-derived brand domains; controls = the genuine brand domains with the same path
    brands = list(sc.xl_full.brand.brand_rd); rng2 = np.random.default_rng(1); kinds = ["homoglyph_digit", "delete_char", "insert_char", "transpose", "hyphenate", "tld_swap", "brand_as_subdomain"]
    recs = []; PATH = "/"       # neutral path: the host alone must carry the signal (an earlier draft used /login/verify-account, which flagged every genuine brand too)
    for kind in kinds:
        for rd in rng2.choice(brands, 400, replace=False):
            if len(rd.split(".")[0]) < 6: continue
            host = mutate_brand(rd, kind, rng2)
            if host: recs.append((kind, "http://" + host + PATH, host))
    ctrl = ["http://" + rd + PATH for rd in rng2.choice(brands, 1500, replace=False)]; tsc = sc.scores([r[1] for r in recs]); csc = sc.scores(ctrl); K = np.array([r[0] for r in recs]); rows = []
    unseen = parts["test"][parts["test"].label.astype(int) == 0].rd.drop_duplicates().sample(3000, random_state=0).values; usc = sc.scores(["http://" + rd + PATH for rd in unseen])   # legit domains never seen in training
    for m in thr:
        for kind in kinds + ["ALL"]:
            sel = np.ones(len(K), bool) if kind == "ALL" else K == kind
            thr_c = float(np.nextafter(np.quantile(usc[m], 0.95, method="higher"), np.inf))      # threshold at which only 5% of unseen legitimate bare domains are flagged
            rows.append(dict(model=m, mutation=kind, n=int(sel.sum()), detection_rate=float((tsc[m][sel] >= thr[m]).mean()), detection_at_5pct_control_flag_rate=float((tsc[m][sel] >= thr_c).mean()), control_genuine_brand_flag_rate=float((csc[m] >= thr[m]).mean()) if kind == "ALL" else None, control_unseen_legit_domain_flag_rate=float((usc[m] >= thr[m]).mean()) if kind == "ALL" else None))
    pd.DataFrame(rows).to_csv(OUT / "e6_typosquat_synthetic.csv", index=False); write_json({"thresholds_policy_fpr1%": thr, "n_typosquats": len(recs), "n_controls_brand": len(ctrl), "n_controls_unseen_legit": len(unseen)}, OUT / "e6_info.json")
    print("e6 done", flush=True)

# ------------------------------------------------------------------------------------------------------------------ E7
def e7():
    """Replicate the main comparison on the three registered-domain-disjoint split seeds (11, 22, 33)."""
    global PROTO
    rows = []
    for seed in [11, 22, 33]:
        PROTO = f"regdomain_seed{seed}"
        try: parts, rec, S, novel = load_all(write=False)
        except FileNotFoundError as e: print("e7 skipping seed", seed, e); continue
        yt, rdt = y_of(parts, "test"), parts["test"].rd.values
        for name, P in S.items():
            th = op_thresholds(P, parts); p = P["p_test"]
            for tn in ["fpr1%", "recall95%", "recall99%"]:
                flag = (p >= th[tn]).astype(float)
                rows.append(dict(seed=seed, model=name, operating_point=tn, pr_auc=float(average_precision_score(yt, p)), roc_auc=float(roc_auc_score(yt, p)), pauc_fpr5=RC.partial_auc(yt, p, 0.05),
                                 recall=float(flag[yt == 1].mean()), fpr=float(flag[yt == 0].mean()),
                                 domain_recall=float(pd.Series(flag[yt == 1]).groupby(rdt[yt == 1]).mean().mean()), domain_fpr=float(pd.Series(flag[yt == 0]).groupby(rdt[yt == 0]).mean().mean())))
        print("e7 seed done", seed, flush=True)
    PROTO = "regdomain_seed11"; pd.DataFrame(rows).to_csv(OUT / "e7_seed_replication.csv", index=False)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=["e1", "e2", "e3", "e4", "e5", "e6", "e7"]); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    need = set(a.only) - {"e4", "e7"}
    if need: parts, rec, S, novel = load_all()
    if "e1" in a.only: e1(parts, rec, S, novel)
    if "e2" in a.only: e2(parts, rec, S, novel)
    if "e3" in a.only: e3(parts, rec, S, novel)
    if "e4" in a.only: e4()
    if "e5" in a.only: e5(parts, S)
    if "e6" in a.only: e6(parts, rec, S)
    if "e7" in a.only: e7()

if __name__ == "__main__": main()
