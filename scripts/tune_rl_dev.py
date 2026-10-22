"""RL training-budget / learning-rate selection on DEVELOPMENT episodes built from the *policy* split only
(never the test split). Writes results/rl_dev_tuning.csv; chosen values are then fixed in config/config.yaml."""
import _bootstrap, itertools, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.common import ROOT, load_config
from src.evaluation.predictions import load_calibrated
from src.decision.costs import CostModel
from src.rl.env import PhishInspectionEnv, Scenario, make_episode, run_episode
from src.rl import agents as A

def trial(cfg, p, y, sc, episodes, alpha_mode, gamma, seed, mode="full"):
    cm = CostModel(**{k: cfg["cost"][k] for k in ("fn", "fp", "inspect")}); T = cfg["rl"]["episode_len"]; K = int(round(cfg["cost"]["capacity_frac"] * T)); env = PhishInspectionEnv(cm, T, K, float(y.mean()))
    ib = (np.flatnonzero(y == 0), np.flatnonzero(y == 1)); ag = A.QAgent(mode, gamma, 0.1, seed=seed, alpha_mode=alpha_mode)
    ag.train(env, lambda e: make_episode(p, y, sc, T, 1000 + e, ib), episodes)
    dev = [make_episode(p, y, sc, T, 700000 + i, ib) for i in range(60)]   # development episodes (policy split, disjoint seeds from training)
    rl = np.mean([run_episode(env, e, ag.policy(T, K))["cost"] for e in dev]); pace = np.mean([run_episode(env, e, A.pol_pacing(cm, p, T, K))["cost"] for e in dev])
    return dict(scenario=f"{sc.name}_d{sc.delay}_n{sc.noise}", mode=mode, episodes=episodes, alpha_mode=alpha_mode, gamma=gamma, seed=seed, dev_cost_rl=rl, dev_cost_pacing=pace)

if __name__ == "__main__":
    cfg = load_config(); proto = f"regdomain_seed{cfg['seeds'][0]}"; D = load_calibrated(proto, "norm", "hybrid_lr", cfg)
    p, y = D["cal_policy"], D["parts"]["policy"]["label"].values.astype(int)
    scs = [Scenario("stationary", 0, 0.0), Scenario("bursty", 10, 0.1)]
    import sys; rnd = sys.argv[1] if len(sys.argv) > 1 else "2"
    if rnd == "1": grid = [(e, am, g, "full") for e, am, g in itertools.product([800, 2500], ["const", "visit"], [0.97, 0.99])]
    else: grid = [(3000, am, g, m) for am, g, m in itertools.product(["const", "visit"], [0.99, 0.995, 0.999], ["full", "budget_only"])]   # round 2: longer horizons (budget is spent over ~1000 steps)
    jobs = [delayed(trial)(cfg, p, y, sc, e, am, g, 0, m) for sc in scs for e, am, g, m in grid]
    R = pd.DataFrame(Parallel(n_jobs=2)(jobs)); R.to_csv(ROOT / "results" / f"rl_dev_tuning_round{rnd}.csv", index=False); print(R.to_string())
