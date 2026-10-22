"""Final integrity checks. Exits non-zero if any check fails. Writes results/integrity_report.json and reports/integrity_report.md.
Checks: raw data hashes unchanged; predictions belong to the supplied dataset hash and split checksums; metrics regenerate from saved
predictions; reports contain the numbers in the machine-readable outputs; model bundle loads and predicts; no placeholders/empty files;
pipeline stage statuses."""
import _bootstrap, json, os, re, sys
import numpy as np, pandas as pd
from pathlib import Path
from src.common import ROOT, load_config, data_dir, sha256_file, read_json
from src.evaluation.predictions import load_calibrated
from src.evaluation.metrics import classification_metrics

cfg = load_config(); checks = []
def check(name, ok, detail=""):
    checks.append({"check": name, "passed": bool(ok), "detail": str(detail)[:600]}); print(("PASS " if ok else "FAIL ") + name, ("- " + str(detail)[:200]) if detail and not ok else "", flush=True)

def main():
    dd = data_dir(cfg); A = read_json(ROOT / "results/dataset_audit.json")
    # 1 raw data hashes unchanged vs the audit (which hashed them at audit time)
    bad = [n for n, v in A["data_dir_files"].items() if not (dd / n).exists() or sha256_file(dd / n) != v["sha256"]]
    check("raw data files match hashes recorded by the audit", not bad, bad)
    z = os.environ.get("PHISH_ORIGINAL_ZIP")
    if z and Path(z).exists(): check("original ZIP hash matches config", sha256_file(z) == cfg["data"]["expected_zip_sha256"], sha256_file(z))
    else: print("SKIP original-ZIP hash check (set PHISH_ORIGINAL_ZIP to enable)")
    dh = sha256_file(dd / cfg["data"]["files"]["master"])
    # 2 predictions belong to dataset hash + split checksum
    sr = read_json(ROOT / "results/split_report.json"); metas = sorted((ROOT / "results/predictions").glob("*.json"))
    badm = []
    for m in metas:
        j = read_json(m); proto = j["protocol"]
        if j.get("dataset_sha256_master") != dh or j.get("split_checksum") != sr["protocols"][proto]["checksum"]: badm.append(m.name)
    check(f"all {len(metas)} prediction sets carry the current dataset hash and split checksum", metas and not badm, badm[:5])
    # 3 metrics regenerate from saved predictions
    M = pd.read_csv(ROOT / "results/supervised_metrics.csv"); prim = f"regdomain_seed{cfg['seeds'][0]}"; errs = []; n = 0
    for kind in ["lr_lex", "tfidf_lr", "hybrid_lr", "hgb_lex", "stack"]:
        for variant in ["norm", "full"]:
            if not (ROOT / f"results/predictions/{prim}__{variant}__{kind}.npz").exists(): continue
            D = load_calibrated(prim, variant, kind, cfg); y = D["parts"]["test"]["label"].values
            r = classification_metrics(y, D["raw_test"], 0.5)
            row = M[(M.protocol == prim) & (M.variant == variant) & (M.kind == kind) & (M.eval_set == "test") & (M.calibration == "raw") & (M.thr_name == "fixed_0.5")].iloc[0]
            n += 1
            for k in ["accuracy", "roc_auc", "pr_auc", "brier", "ece"]:
                if abs(r[k] - row[k]) > 1e-9: errs.append((kind, variant, k, r[k], row[k]))
    check(f"supervised metrics regenerated from saved predictions for {n} model/variant pairs", n > 0 and not errs, errs[:3])
    # 4 reports contain machine-readable numbers
    rep = (ROOT / "reports/supervised_results.md").read_text(); miss = []
    for kind in ["lr_lex", "hybrid_lr", "tfidf_lr"]:
        row = M[(M.protocol == prim) & (M.variant == "norm") & (M.kind == kind) & (M.eval_set == "test") & (M.calibration == "cal") & (M.thr_name == "val_f1")].iloc[0]
        for k in ["roc_auc", "pr_auc", "f1"]:
            if f"{row[k]:.4f}" not in rep: miss.append((kind, k, f"{row[k]:.4f}"))
    check("supervised_results.md contains the values from supervised_metrics.csv", not miss, miss[:3])
    T = pd.read_csv(ROOT / "results/rl_summary_table.csv"); rl = (ROOT / "reports/sequential_rl_results.md").read_text()
    miss = [(r.scenario, r.policy) for _, r in T.head(12).iterrows() if f"{r.cost:.0f}" not in rl]
    check("sequential_rl_results.md contains values from rl_summary_table.csv", not miss, miss[:3])
    C = pd.read_csv(ROOT / "results/cost_static_policies.csv"); cs = (ROOT / "reports/cost_sensitive_results.md").read_text()
    miss = [r.policy for _, r in C.iterrows() if f"{r.cost_per_1k:.1f}" not in cs]
    check("cost_sensitive_results.md contains values from cost_static_policies.csv", not miss, miss[:3])
    # 5 model bundle loads and predicts
    try:
        from src.inference.predict import Detector
        d = Detector.load(ROOT / "artifacts/deployable_bundle.joblib"); out = d.predict(["http://example.com/a", "http://secure-login-verify.xyz/account/update.php?id=1"])
        ok = all(o["action"] in ("ALLOW", "BLOCK", "INSPECT") and 0 <= o["risk_score"] <= 1 for o in out)
        check("deployable bundle loads and predicts", ok, out)
    except Exception as e: check("deployable bundle loads and predicts", False, e)
    # 6 placeholders / empty files / required reports
    req = ["dataset_audit", "literature_review", "research_gap_analysis", "research_questions", "experimental_protocol", "supervised_results", "cost_sensitive_results", "sequential_rl_results", "distribution_shift_results", "safe_adaptation_results", "limitations", "final_research_report"]
    missing = [r for r in req if not (ROOT / "reports" / f"{r}.md").exists() or (ROOT / "reports" / f"{r}.md").stat().st_size < 500]
    check("all required reports exist and are non-trivial", not missing, missing)
    check("references.bib exists", (ROOT / "references.bib").exists() and (ROOT / "references.bib").stat().st_size > 1000)
    empties = [str(p.relative_to(ROOT)) for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")) if p.stat().st_size == 0 and p.name != "__init__.py"]
    check("no empty source files", not empties, empties)
    pat = re.compile(r"TODO|FIXME|NotImplementedError|XXX|not yet written|lorem ipsum", re.I); hits = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "reports").glob("*.md")):
        if p.name == "verify_integrity.py": continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines()):
            if pat.search(line): hits.append(f"{p.relative_to(ROOT)}:{i + 1}")
    check("no placeholder markers in source/reports", not hits, hits[:5])
    # 7 stage statuses
    st = read_json(ROOT / "results/run_status.json")
    others = [s for s in st["stages"] if s["stage"] != "integrity"]   # this stage cannot attest to its own (previous) record
    check("run_status: every recorded stage (except this one) exited 0", all(s["exit_status"] == 0 for s in others), [s["stage"] for s in others if s["exit_status"]])
    fast = [s["stage"] for s in st["stages"] if s.get("mode") == "fast"]
    check("no stage was run in FAST (reduced-budget) mode", not fast, fast)
    ok = all(c["passed"] for c in checks)
    (ROOT / "results/integrity_report.json").write_text(json.dumps({"all_passed": ok, "checks": checks}, indent=2))
    L = ["# Integrity report", "", f"_Generated by `scripts/verify_integrity.py`. Overall: **{'ALL PASSED' if ok else 'FAILURES PRESENT'}**._", "", "| check | result | detail |", "|---|---|---|"]
    L += [f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail'][:160].replace('|', '/') if not c['passed'] else ''} |" for c in checks]
    (ROOT / "reports/integrity_report.md").write_text("\n".join(L) + "\n")
    sys.exit(0 if ok else 1)

if __name__ == "__main__": main()
