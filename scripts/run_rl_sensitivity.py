"""Sensitivity of the sequential comparison to the ASSUMED cost matrix and inspection capacity (so that no single favourable
cost matrix drives the conclusion). For each variant the learners are re-trained under that variant's costs/capacity.
Outputs results/rl_sensitivity_episodes.csv and results/rl_sensitivity_summary.csv (mean episode cost per policy)."""
import _bootstrap, argparse, time, copy
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.common import ROOT, load_config
from src.evaluation.metrics import threshold_for_cost
from src.decision.costs import CostModel
from src.rl.env import Scenario
from run_rl_experiments import run_scenario, pick_classifier, sname

VARIANTS = {"base(fn=10,insp=0.5,cap=5%)": (10, 1, 0.5, 0.05), "fn=5": (5, 1, 0.5, 0.05), "fn=20": (20, 1, 0.5, 0.05),
            "insp=0.25": (10, 1, 0.25, 0.05), "insp=1.0": (10, 1, 1.0, 0.05), "cap=2%": (10, 1, 0.5, 0.02), "cap=10%": (10, 1, 0.5, 0.10)}
SCENS = [Scenario("stationary", 0, 0.1), Scenario("bursty", 10, 0.1)]

def one(vname, v, sc, cfg, D):
    fn, fp, insp, cap = v; cfg = copy.deepcopy(cfg); cfg["cost"].update(fn=fn, fp=fp, inspect=insp, capacity_frac=cap)
    cm = CostModel(fn, fp, insp, 0.01); T = cfg["rl"]["episode_len"]; K = int(round(cap * T))
    p_tr, y_tr = D["cal_policy"], D["parts"]["policy"]["label"].values.astype(int); p_ev, y_ev = D["cal_test"], D["parts"]["test"]["label"].values.astype(int)
    thr = threshold_for_cost(D["parts"]["val"]["label"].values, D["cal_val"], fn, fp)
    rows, _, _ = run_scenario(sc, cfg, cm, T, K, p_tr, y_tr, p_ev, y_ev, float(y_tr.mean()), thr)
    E = pd.DataFrame(rows); E["variant"] = vname; return E

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true"); a = ap.parse_args()
    cfg = load_config()
    if a.fast: cfg["rl"].update(train_episodes=60, eval_episodes=10, agent_seeds=[0])
    proto = f"regdomain_seed{cfg['seeds'][0]}"; _, kind, D = pick_classifier(cfg, proto)
    t0 = time.time(); res = Parallel(n_jobs=2)(delayed(one)(vn, v, sc, cfg, D) for vn, v in VARIANTS.items() for sc in SCENS)
    E = pd.concat(res); E["policy_g"] = E.policy.str.replace(r"uncertainty_band\[.*\]", "uncertainty_band", regex=True)
    E.to_csv(ROOT / "results" / "rl_sensitivity_episodes.csv", index=False)
    pe = E.groupby(["scenario", "variant", "policy_g", "episode"]).cost.mean().reset_index()
    S = pe.groupby(["scenario", "variant", "policy_g"]).cost.mean().unstack("policy_g").reset_index()
    keep = ["scenario", "variant", "val_tuned_threshold", "greedy_masked(bayes)", "uncertainty_band", "pacing_heuristic", "pacing+prior_adapt", "bandit(full,g=0)", "RL_Q(budget_only,g=0.97)", "RL_Q(full,g=0.97)"]
    S = S[[c for c in keep if c in S.columns]]; S.to_csv(ROOT / "results" / "rl_sensitivity_summary.csv", index=False)
    print("done", time.time() - t0, "fast" if a.fast else "full"); print(S.round(0).to_string())

if __name__ == "__main__": main()
