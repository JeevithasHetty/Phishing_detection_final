"""Master entry point:  python scripts/run_all.py [--fast] [--from STAGE] [--only STAGE ...] [--skip-tests]

Runs the pipeline stages in order. Every stage is a subprocess whose exit status, wall time and log file are recorded in
results/run_status.json. A failing stage aborts the run with a non-zero exit code (no 'success' report is produced).
Stages are resumable: model training reuses cached predictions only if the cache key (code hash, dataset hash, split
checksum, config) still matches (see scripts/train_models.py); the other stages are re-run (they are comparatively cheap).

--fast  : reduced-budget RL / adaptation runs (labelled FAST in run_status.json and in the reports; NOT a full run).
"""
import _bootstrap, argparse, json, subprocess, sys, time, datetime
from pathlib import Path
from src.common import ROOT, load_config, sha256_file, sha256_text, data_dir, env_info

PY = sys.executable
STAGES = [
    ("audit",       [PY, "scripts/audit_dataset.py"],                       "1 dataset audit"),
    ("prepare",     [PY, "scripts/prepare_data.py"],                        "2 data preparation + split/leakage verification"),
    ("train",       [PY, "scripts/train_models.py"],                        "3 supervised baselines (all protocols)"),
    ("evaluate",    [PY, "scripts/evaluate_models.py"],                     "4-5 validation calibration, test evaluation, CIs"),
    ("bundle",      [PY, "scripts/build_inference_bundle.py"],              "inference bundle (best model by validation AP)"),
    ("rl",          [PY, "scripts/run_rl_experiments.py"],                  "6-7 cost-sensitive policies + sequential RL"),
    ("rl_sens",     [PY, "scripts/run_rl_sensitivity.py"],                  "6-7b RL sensitivity to cost matrix / capacity"),
    ("shift",       [PY, "scripts/run_shift_experiments.py"],               "8 distribution-shift experiments"),
    ("adaptation",  [PY, "scripts/run_adaptation_experiments.py"],          "9 safe-adaptation simulation"),
    ("reports",     [PY, "scripts/make_reports.py"],                        "10 report + plot generation from saved outputs"),
    ("tests",       [PY, "-m", "unittest", "discover", "-s", "tests", "-v"],  "automated tests"),
    ("integrity",   [PY, "scripts/verify_integrity.py"],                    "11 final integrity checks"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true"); ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    cfg = load_config(); logs = ROOT / "logs"; logs.mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    names = [s[0] for s in STAGES]
    if a.start and a.start not in names: raise SystemExit(f"unknown stage {a.start}; choose from {names}")
    selected = [s for s in STAGES if (not a.only or s[0] in a.only) and (not a.start or names.index(s[0]) >= names.index(a.start))]
    if a.skip_tests: selected = [s for s in selected if s[0] != "tests"]
    status = {"started": datetime.datetime.now().isoformat(timespec="seconds"), "mode": "FAST (reduced budget)" if a.fast else "FULL",
              "env": env_info(), "config_sha256": sha256_text((ROOT / "config" / "config.yaml").read_text()),
              "dataset_master_sha256": sha256_file(data_dir(cfg) / cfg["data"]["files"]["master"]), "stages": []}
    path = ROOT / "results" / "run_status.json"
    prev = {}
    if path.exists() and (a.only or a.start):  # keep earlier stage records when doing a partial run
        try: prev = {s["stage"]: s for s in json.loads(path.read_text()).get("stages", [])}
        except Exception: prev = {}
    for name, cmd, desc in selected:
        cmd = list(cmd)
        if a.fast and name in ("rl", "rl_sens", "adaptation"): cmd.append("--fast")
        t0 = time.time(); print(f"[run_all] >>> {name}: {desc}", flush=True)
        with open(logs / f"stage_{name}.log", "w") as lf:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode
        rec = {"stage": name, "description": desc, "command": " ".join(cmd[1:]), "exit_status": rc, "seconds": round(time.time() - t0, 1),
               "log": f"logs/stage_{name}.log", "mode": "fast" if (a.fast and name in ("rl", "rl_sens", "adaptation")) else "full"}
        prev[name] = rec
        status["stages"] = [prev[n] for n in names if n in prev]
        status["finished"] = datetime.datetime.now().isoformat(timespec="seconds"); status["all_ok"] = all(s["exit_status"] == 0 for s in status["stages"])
        path.write_text(json.dumps(status, indent=2))
        print(f"[run_all]     exit={rc} in {rec['seconds']}s", flush=True)
        if rc != 0:
            print(f"[run_all] STAGE FAILED: {name} (see {rec['log']}). Aborting; no final report is valid.", file=sys.stderr)
            tail = (logs / f"stage_{name}.log").read_text().splitlines()[-15:]; print("\n".join(tail), file=sys.stderr); sys.exit(1)
    print("[run_all] all requested stages finished with exit status 0")

if __name__ == "__main__": main()
