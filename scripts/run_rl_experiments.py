"""Cost-sensitive policy comparison (static) + sequential RL experiments. Outputs results/cost_*.csv, results/rl_*.csv, figures, reports."""
import _bootstrap, argparse, time, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from src.common import ROOT, load_config, write_json
from src.evaluation.predictions import load_calibrated
from src.evaluation.metrics import threshold_for_cost
from src.decision.costs import CostModel
from src.decision import policies as P
from src.rl.env import PhishInspectionEnv, Scenario, make_episode, run_episode
from src.rl import agents as A

def pick_classifier(cfg, proto, variant="norm"):
    best = None
    for k in ["lr_lex", "hgb_lex", "tfidf_lr", "hybrid_lr", "rf_lex", "stack"]:
        try: D = load_calibrated(proto, variant, k, cfg)
        except FileNotFoundError: continue
        from sklearn.metrics import average_precision_score as ap
        s = ap(D["parts"]["val"]["label"].values, D["cal_val"])
        if best is None or s > best[0]: best = (s, k, D)
    return best

def scen_list(cfg):
    return [Scenario(n, d, z) for n in ["stationary", "bursty"] for d in cfg["rl"]["delays"] for z in cfg["rl"]["noises"]]

def sname(s): return f"{s.name}_d{s.delay}_n{s.noise}"

def run_scenario(sc, cfg, cm, T, K, p_tr, y_tr, p_ev, y_ev, prior, thr_tuned, train_sc=None, tag="matched"):
    R = cfg["rl"]; train_sc = train_sc or sc
    env = PhishInspectionEnv(cm, T, K, prior)
    ib_tr = (np.flatnonzero(y_tr == 0), np.flatnonzero(y_tr == 1)); ib_ev = (np.flatnonzero(y_ev == 0), np.flatnonzero(y_ev == 1))
    mk_tr = lambda e: make_episode(p_tr, y_tr, train_sc, T, 1000 + e, ib_tr)
    eps_ev = [make_episode(p_ev, y_ev, sc, T, 900000 + i, ib_ev) for i in range(R["eval_episodes"])]
    # tune uncertainty band on training-environment episodes of the matched scenario
    tune_eps = [mk_tr(5000 + i) for i in range(25)]
    best = None
    for lo in (0.02, 0.05, 0.1, 0.2, 0.3):
        for hi in (0.5, 0.6, 0.7, 0.8, 0.9):
            c = np.mean([run_episode(env, e, A.pol_uncertainty(cm, lo, hi))["cost"] for e in tune_eps])
            if best is None or c < best[0]: best = (c, lo, hi)
    pols = {"fixed_0.5": A.pol_fixed(0.5), "val_tuned_threshold": A.pol_fixed(thr_tuned), "bayes_unmasked": A.pol_bayes(cm, False),
            "greedy_masked(bayes)": A.pol_bayes(cm, True), f"uncertainty_band[{best[1]},{best[2]}]": A.pol_uncertainty(cm, best[1], best[2]),
            "pacing_heuristic": A.pol_pacing(cm, p_tr, T, K), "pacing+prior_adapt": A.pol_pacing(cm, p_tr, T, K, True, prior)}
    rows = []
    def ev(name, pol, seed):
        for i, e in enumerate(eps_ev):
            m = run_episode(env, e, pol); rows.append(dict(scenario=sname(sc), train_scenario=sname(train_sc), mode=tag, policy=name, seed=seed, episode=i, **{k: m[k] for k in ["cost", "fn", "fp", "inspections", "violations", "analyst_errors", "hold_cost", "phishing_arrivals"]}))
    for n, pl in pols.items(): ev(n, pl, -1)
    for i in range(len(eps_ev)): pass
    ev("random_valid", A.pol_random(0), -1)
    curves = []
    for sd in R["agent_seeds"]:
        for name, mode, gamma in [("RL_Q(full,g=0.97)", "full", R["gamma"]), ("RL_Q(budget_only,g=0.97)", "budget_only", R["gamma"]), ("bandit(full,g=0)", "full", 0.0)]:
            ag = A.QAgent(mode, gamma, R["alpha"], seed=sd)
            cv = ag.train(env, mk_tr, R["train_episodes"], R["eps_start"], R["eps_end"])
            curves += [dict(scenario=sname(train_sc), agent=name, seed=sd, episode=e, train_cost=c, eps=ep) for e, c, ep in cv]
            ev(name, ag.policy(T, K), sd)
            if sd == R["agent_seeds"][0] and name.startswith("RL_Q(full") and tag == "matched" and sc.name == "bursty" and sc.delay == 10 and sc.noise == 0.1:
                np.savez_compressed(ROOT / "artifacts" / "q_table_bursty_d10_n0.1.npz", Q=ag.Q, visits=ag.visits)
    return rows, curves, dict(band=best[1:], scenario=sname(sc))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true"); ap.add_argument("--classifier", default=None); args = ap.parse_args()
    cfg = load_config()
    if args.fast: cfg["rl"].update(train_episodes=60, eval_episodes=10, agent_seeds=[0], delays=[0, 10], noises=[0.1])
    c = cfg["cost"]; cm = CostModel(c["fn"], c["fp"], c["inspect"], 0.01); T = cfg["rl"]["episode_len"]; K = int(round(c["capacity_frac"] * T))
    proto = f"regdomain_seed{cfg['seeds'][0]}"
    out = ROOT / "results"; figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    classifiers = [args.classifier] if args.classifier else None
    sel = pick_classifier(cfg, proto)
    ap_val, kind, D = sel
    weak = load_calibrated(proto, "norm", "lr_lex", cfg)
    info = {"classifier": kind, "val_average_precision": ap_val, "calibrator": D["calibrator_name"], "protocol": proto, "variant": "norm", "costs": c, "T": T, "K": K, "weak_classifier": "lr_lex"}
    # ---------- Section A: static cost-sensitive policies on the whole test stream ----------
    stat_rows = []
    for label, DD in [("primary:" + kind, D), ("weak:lr_lex", weak)]:
        yv, pv = DD["parts"]["val"]["label"].values, DD["cal_val"]
        thr_t = threshold_for_cost(yv, pv, cm.fn, cm.fp)
        yp, pp = DD["parts"]["policy"]["label"].values, DD["cal_policy"]
        Kp = int(round(c["capacity_frac"] * len(yp))); band = P.tune_uncertainty_band(pp, yp, cm, Kp)
        bandit = P.ContextualBandit(cm, seed=0).fit(pp, yp, passes=1)
        yt, pt = DD["parts"]["test"]["label"].values, DD["cal_test"]
        order = np.random.default_rng(0).permutation(len(yt)); yt, pt = yt[order], pt[order]; Kt = int(round(c["capacity_frac"] * len(yt)))
        res = P.evaluate_all_static(pt, yt, cm, Kt, thr_t, band, extra=[bandit.evaluate(pt, yt, Kt)])
        for r in res: r["classifier"] = label; r["tuning"] = f"thr_tuned_on_val={thr_t:.4f}; band tuned on policy split={band}"; stat_rows.append(r)
    pd.DataFrame(stat_rows).to_csv(out / "cost_static_policies.csv", index=False)
    # ---------- Section B: sequential experiments ----------
    p_tr, y_tr = D["cal_policy"], D["parts"]["policy"]["label"].values.astype(int)
    p_ev, y_ev = D["cal_test"], D["parts"]["test"]["label"].values.astype(int)
    prior = float(y_tr.mean()); thr_tuned = threshold_for_cost(D["parts"]["val"]["label"].values, D["cal_val"], cm.fn, cm.fp)
    scs = scen_list(cfg); t0 = time.time()
    jobs = [delayed(run_scenario)(s, cfg, cm, T, K, p_tr, y_tr, p_ev, y_ev, prior, thr_tuned) for s in scs]
    # mismatch: trained on stationary_d0_n0, evaluated on bursty_d10_n0.1 (or first bursty scenario in fast mode)
    tgt = [s for s in scs if s.name == "bursty"][-1 if args.fast else 3]
    jobs.append(delayed(run_scenario)(tgt, cfg, cm, T, K, p_tr, y_tr, p_ev, y_ev, prior, thr_tuned, Scenario("stationary", 0, 0.0), "mismatch(train=stationary_d0_n0)"))
    res = Parallel(n_jobs=2)(jobs)
    rows = [r for rr, _, _ in res for r in rr]; curves = [r for _, cc, _ in res for r in cc]; bands = [b for _, _, b in res]
    E = pd.DataFrame(rows); E.to_csv(out / "rl_episode_results.csv", index=False); pd.DataFrame(curves).to_csv(out / "rl_training_curves.csv", index=False)
    info.update(bands=bands, wall_seconds=time.time() - t0, rl=cfg["rl"], fast=bool(args.fast)); write_json(info, out / "rl_run_info.json")
    plot_curves(pd.DataFrame(curves), figs)
    print("done", time.time() - t0)

def plot_curves(C, figs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (sc, ag), g in C.groupby(["scenario", "agent"]):
        if sc != "bursty_d10_n0.1": continue
        m = g.groupby("episode").train_cost.mean(); ax.plot(m.index, m.values, label=ag)
    ax.set_xlabel("training episode"); ax.set_ylabel("mean training-episode cost (eps-greedy)"); ax.set_title("Learning curves, bursty_d10_n0.1"); ax.legend(); fig.tight_layout(); fig.savefig(figs / "rl_learning_curves.png", dpi=110); plt.close(fig)

if __name__ == "__main__": main()
