"""Safe-adaptation simulation (OFFLINE). Compares: static | ungated | ungated+rollback | gated | gated+rollback
under (A) an observed external shift ('recent' set) and (B) simulated input-preserving concept drift, with and without
a simulated label-poisoning attack. The gates never see the final-evaluation set."""
import _bootstrap, argparse, copy, json
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.common import ROOT, load_config, write_json
from src.evaluation.predictions import partitions
from src.features.lexical import strip_www_url, FEATURE_NAMES
from src.data.splits import group_assign
from src.decision.costs import CostModel
from src.adaptation.safe_update import GateConfig, ModelRegistry, evaluate_candidate, cost_per_1k

HV = HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=2 ** 18, alternate_sign=False, norm="l2", lowercase=True, dtype=np.float32)

class Feats:
    def __init__(self, scaler): self.sc = scaler
    def __call__(self, urls, lex):
        Xt = HV.transform([strip_www_url(u) for u in urls]); Xl = self.sc.transform(np.log1p(np.abs(lex))).astype(np.float32) * 0.3
        return sp.hstack([Xt, sp.csr_matrix(Xl)]).tocsr()

def fit_base(X, y, seed, epochs=4):
    m = SGDClassifier(loss="log_loss", alpha=2e-6, random_state=seed); classes = np.array([0, 1]); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        o = rng.permutation(X.shape[0]); m.partial_fit(X[o], y[o], classes=classes)
    return m

score = lambda m, X: m.predict_proba(X)[:, 1]

def make_candidate(inc, Xfb, yfb, Xrep, yrep, seed, passes=5, w=3.0):
    cand = copy.deepcopy(inc); rng = np.random.default_rng(seed)
    X = sp.vstack([Xfb, Xrep]).tocsr(); y = np.r_[yfb, yrep]; sw = np.r_[np.full(len(yfb), w), np.ones(len(yrep))]
    for _ in range(passes):
        o = rng.permutation(X.shape[0]); cand.partial_fit(X[o], y[o], sample_weight=sw[o])
    return cand

def simulate(strategy, seed, base, T, ref, evalset, cfg_s, gate, cm, thr, rounds_poison):
    """T: dict with X (sparse), y (observed-true labels for the target stream), batches idx list."""
    rng = np.random.default_rng(seed); reg = ModelRegistry(); reg.commit(base, {"note": "v0 deployed", "round": -1}); hist = []
    gated = "gated" in strategy and not strategy.startswith("ungated"); use_rb = "rollback" in strategy
    for r, bidx in enumerate(T["batches"]):
        if strategy != "static":
            fb = rng.choice(bidx, cfg_s["n_feedback"], replace=False); yfb = T["y"][fb].copy()
            flip = rng.random(len(fb)) < cfg_s["analyst_noise"]; yfb = np.where(flip, 1 - yfb, yfb)
            if r in rounds_poison and cfg_s["poison"] != "none":
                if cfg_s["poison"] == "flip_phish_to_legit": m = (yfb == 1) & (rng.random(len(fb)) < 0.6); yfb[m] = 0
                elif cfg_s["poison"] == "random_flip": m = rng.random(len(fb)) < 0.3; yfb = np.where(m, 1 - yfb, yfb)
            k = int(0.7 * len(fb)); fit_i, val_i = np.arange(k), np.arange(k, len(fb))
            # post-commit audit of the active version on this round's fresh feedback (rollback monitor)
            if use_rb and reg.active > 0:
                prev, act = reg.versions[reg.active - 1]["model"], reg.model
                live_a, live_p = cost_per_1k(yfb, score(act, T["X"][fb]), thr, cm.fn, cm.fp), cost_per_1k(yfb, score(prev, T["X"][fb]), thr, cm.fn, cm.fp)
                ref_a, ref_p = roc_auc_score(ref["y"], score(act, ref["X"])), roc_auc_score(ref["y"], score(prev, ref["X"]))
                if live_a > live_p * (1 + gate.rollback_cost_margin) + 1.0 or (ref_p - ref_a) > gate.max_reference_regression:
                    reg.rollback(f"round{r}: live_cost {live_a:.1f} vs prev {live_p:.1f}; ref_auc {ref_a:.4f} vs prev {ref_p:.4f}")
            rep = rng.choice(ref["Xtr_n"], 1500, replace=False)
            cand = make_candidate(reg.model, T["X"][fb[fit_i]], yfb[fit_i], ref["Xtr"][rep], ref["ytr"][rep], seed + r)
            if gated:
                ok, reasons, d = evaluate_candidate(gate, reg.model, cand, T["X"][fb[fit_i]], yfb[fit_i], T["X"][fb[val_i]], yfb[val_i], ref["X"], ref["y"], thr, cm.fn, cm.fp, score)
            else: ok, reasons, d = True, ["accepted_without_gates"], {}
            if ok: reg.commit(cand, {"round": r, "strategy": strategy, "reasons": reasons, **{k_: float(v) for k_, v in d.items()}})
            else: reg._log("reject", round=r, reasons=reasons, **{k_: float(v) for k_, v in d.items()})
        m = reg.model; pe = score(m, evalset["X"])
        hist.append(dict(round=r, strategy=strategy, active_version=reg.active, eval_cost_per_1k=cost_per_1k(evalset["y"], pe, thr, cm.fn, cm.fp), eval_auc=float(roc_auc_score(evalset["y"], pe)),
                         ref_auc=float(roc_auc_score(ref["y"], score(m, ref["X"])))))
    ev = [e["event"] for e in reg.log]
    return hist, dict(accepted=ev.count("commit") - 1, rejected=ev.count("reject"), rollbacks=ev.count("rollback")), reg.log

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true"); a = ap.parse_args()
    cfg = load_config(); c = cfg["cost"]; cm = CostModel(c["fn"], c["fp"], c["inspect"]); thr = cm.bayes_threshold(); proto = f"regdomain_seed{cfg['seeds'][0]}"
    parts, rec = partitions(proto, cfg); proc = ROOT / cfg["processed_dir"]
    lexp, lexr = np.load(proc / "lex_norm_pool.npy"), np.load(proc / "lex_norm_rec.npy"); rng0 = np.random.default_rng(0)
    ntr = 30000 if a.fast else 200000
    tr = parts["train"]; tri = rng0.choice(len(tr), min(ntr, len(tr)), replace=False)
    sc = StandardScaler().fit(np.log1p(np.abs(lexp[tr["pos"].values])))
    F = Feats(sc); Xtr = F(tr["url"].values[tri], lexp[tr["pos"].values[tri]]); ytr = tr["label"].values[tri].astype(int)
    va = parts["val"]; vi = rng0.choice(len(va), min(10000, len(va)), replace=False)
    ref = {"X": F(va["url"].values[vi], lexp[va["pos"].values[vi]]), "y": va["label"].values[vi].astype(int), "Xtr": Xtr, "ytr": ytr, "Xtr_n": np.arange(len(ytr))}
    seeds = [0] if a.fast else [0, 1, 2]; S = dict(rounds=4 if a.fast else 8, batch=2000, n_feedback=1000, analyst_noise=0.05)
    gate = GateConfig(); poison_rounds = {1, 2} if a.fast else {2, 3, 4}
    scen = {}
    # (A) observed shift: external 'recent' set, split by registered domain into adaptation stream / final-evaluation set
    sp_ = group_assign(rec["rd"], 7, {"train": 0.6, "val": 0.0, "policy": 0.0, "test": 0.4})
    A_ad, A_ev = np.flatnonzero(sp_ == "train"), np.flatnonzero(sp_ == "test")
    # (B) simulated concept drift on the pool's test partition (inputs unchanged; labels flipped for host digits>=2)
    te = parts["test"]; ti = rng0.choice(len(te), min(40000, len(te)), replace=False); tee = te.iloc[ti]; Xl = lexp[tee["pos"].values]
    flipm = Xl[:, FEATURE_NAMES.index("n_digits_host")] >= 2; yB = np.where(flipm, 1 - tee["label"].values, tee["label"].values).astype(int)
    grpB = group_assign(tee["rd"].reset_index(drop=True), 9, {"train": 0.6, "val": 0.0, "policy": 0.0, "test": 0.4}); B_ad, B_ev = np.flatnonzero(grpB == "train"), np.flatnonzero(grpB == "test")
    def build(urls, lex, y, ad, ev, seed):
        r = np.random.default_rng(seed); ad = r.permutation(ad)[: S["rounds"] * S["batch"]]
        Xa = F(urls[ad], lex[ad]); batches = [np.arange(i * S["batch"], (i + 1) * S["batch"]) for i in range(S["rounds"]) if (i + 1) * S["batch"] <= len(ad)]
        return {"X": Xa, "y": y[ad].astype(int), "batches": batches}, {"X": F(urls[ev], lex[ev]), "y": y[ev].astype(int)}
    scen["observed_external(recent)"] = build(rec["url"].values, lexr, rec["label"].values, A_ad, A_ev, 1)
    scen["simulated_concept_drift"] = build(tee["url"].values, Xl, yB, B_ad, B_ev, 2)
    strategies = ["static", "ungated", "ungated+rollback", "gated", "gated+rollback"]; poisons = ["none", "flip_phish_to_legit", "random_flip"]
    rows, audits, summ = [], {}, []
    for seed in seeds:
        base = fit_base(Xtr, ytr, seed)
        for sn, (T, E) in scen.items():
            for ps in poisons:
                for st in strategies:
                    if st == "static" and ps != "none": continue
                    h, cnt, log = simulate(st, seed, base, T, ref, E, dict(S, poison=ps), gate, cm, thr, poison_rounds)
                    for e in h: e.update(scenario=sn, poison=ps, seed=seed)
                    rows += h; summ.append(dict(scenario=sn, poison=ps, strategy=st, seed=seed, final_eval_cost_per_1k=h[-1]["eval_cost_per_1k"], final_eval_auc=h[-1]["eval_auc"], final_ref_auc=h[-1]["ref_auc"], initial_eval_cost_per_1k=None, **cnt))
                    if seed == seeds[0] and st.startswith("gated") and ps != "random_flip": audits[f"{sn}|{ps}|{st}"] = log
        print("seed", seed, "done", flush=True)
    H, Sm = pd.DataFrame(rows), pd.DataFrame(summ)
    H.to_csv(ROOT / "results" / "adaptation_rounds.csv", index=False); Sm.to_csv(ROOT / "results" / "adaptation_summary.csv", index=False)
    write_json(audits, ROOT / "results" / "adaptation_audit_log.json"); write_json({"gate": gate.__dict__, "sim": S, "poison_rounds": sorted(poison_rounds), "threshold": thr, "base_model": "SGD logistic (hashed char 2-4grams + lexical)", "n_base_train": int(len(ytr)), "fast": bool(a.fast)}, ROOT / "results" / "adaptation_run_info.json")
    figs = ROOT / "results" / "figures"
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    for i, sn in enumerate(scen):
        for j, ps in enumerate(poisons):
            ax = axs[i, j]
            for st in strategies:
                d = H[(H.scenario == sn) & (H.poison == ps) & (H.strategy == st)]
                if len(d): g = d.groupby("round").eval_cost_per_1k.mean(); ax.plot(g.index, g.values, "o-", ms=3, label=st)
            ax.set_title(f"{sn} | poison={ps}", fontsize=8); ax.set_xlabel("round"); ax.set_ylabel("final-eval cost/1k")
    axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(figs / "adaptation_rounds.png", dpi=110); plt.close(fig)
    write_report(Sm, H, gate, S, poison_rounds)

def write_report(Sm, H, gate, S, pr):
    g = Sm.groupby(["scenario", "poison", "strategy"]).agg(n=("seed", "count"), cost=("final_eval_cost_per_1k", "mean"), cost_sd=("final_eval_cost_per_1k", "std"), auc=("final_eval_auc", "mean"), ref_auc=("final_ref_auc", "mean"), acc=("accepted", "mean"), rej=("rejected", "mean"), rb=("rollbacks", "mean")).reset_index()
    L = ["# Safe-adaptation results (offline simulation)", "", "_Generated by `scripts/run_adaptation_experiments.py` from `results/adaptation_summary.csv`. Status: **Simulated** (no deployed system; feedback, noise and poisoning are simulated)._", "",
         f"Setup: base model = SGD logistic regression on hashed char 2-4-grams + lexical features trained on the primary training partition; {S['rounds']} rounds × {S['batch']} arriving URLs; {S['n_feedback']} labelled feedback items per round (70% fit / 30% fresh feedback-validation), analyst noise {S['analyst_noise']}; poisoning window rounds {sorted(pr)}. Final evaluation uses a held-out, registered-domain-disjoint target set with true labels that **no gate sees**. Reference set = trusted clean validation partition. Costs ASSUMED (fn=10, fp=1), fixed Bayes threshold.", "",
         f"Gates: min feedback {gate.min_feedback}, min per class {gate.min_per_class}, confident-disagreement ≤ {gate.max_confident_disagreement}, candidate must beat incumbent on fresh feedback-validation cost, reference AUC drop ≤ {gate.max_reference_regression}, reference cost increase ≤ {gate.max_reference_cost_increase:.0%}; rollback if live-window cost > {gate.rollback_cost_margin:.0%} above previous version or reference AUC regresses.", "",
         "Threat model (poisoning): an attacker controls part of the verified-label feedback in the poison window. `flip_phish_to_legit`: 60% of phishing feedback labelled legitimate (targeted evasion). `random_flip`: 30% of labels flipped at random. No claim is made about adaptive attackers or other attack types.", "",
         "| scenario | poison | strategy | seeds | final eval cost/1k (mean ± sd) | final eval AUC | reference AUC | accepted | rejected | rollbacks |", "|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for _, r in g.iterrows(): L.append(f"| {r.scenario} | {r.poison} | {r.strategy} | {r.n} | {r.cost:.1f} ± {0 if pd.isna(r.cost_sd) else r.cost_sd:.1f} | {r.auc:.4f} | {r.ref_auc:.4f} | {r.acc:.1f} | {r.rej:.1f} | {r.rb:.1f} |")
    L += ["", "Per-round trajectories: `results/adaptation_rounds.csv`, `results/figures/adaptation_rounds.png`; audit logs (accept/reject reasons, rollbacks): `results/adaptation_audit_log.json`.", "",
          "Interpretation is written in `reports/final_research_report.md` after reading these numbers; gates are hand-set (not tuned on the final-evaluation set) and are conventional, so results should be read as a feasibility simulation, not as validation of a production safeguard."]
    (ROOT / "reports" / "safe_adaptation_results.md").write_text("\n".join(L) + "\n")

if __name__ == "__main__": main()
