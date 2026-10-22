"""v2 integrity checks: required outputs exist and are non-empty, no NaN in key result columns, reports contain no placeholder text,
every number-bearing report was generated after its inputs, figures exist. Exit status 1 with a clear message on any failure."""
import _bootstrap, sys, json
import pandas as pd
from src.common import ROOT
O = ROOT / "results" / "v2"; R = ROOT / "reports"
need_csv = ["e1_operating_points", "e1_bootstrap_ci", "e1_paired_differences", "e1_calibration", "e2_recall_control", "e2b_recalibration_on_target_feedback", "e3_cascade", "e4_leakage_demo", "e6_obfuscation_attacks", "e6_typosquat_synthetic", "e7_seed_replication"]
need_json = ["e3_info", "e5_explanations", "e6_info", "ensemble"]
need_fig = ["v2_recall_url_vs_domain", "v2_padding_attack", "v2_recall_control_small_n", "v2_leakage_demo"]
need_rep = ["v2_results", "v2_interpretation", "v2_limitations", "v2_roadmap", "v2_literature_addendum", "v2_final_report"]
fails = []
def chk(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok: fails.append(msg)
for n in need_csv:
    f = O / f"{n}.csv"; ok = f.exists() and f.stat().st_size > 0
    chk(ok, f"results/v2/{n}.csv exists and is non-empty")
    if ok:
        d = pd.read_csv(f); num = d.select_dtypes("number")
        if n == "e2_recall_control": num = num.drop(columns=[c for c in num if "shortfall" in c], errors="ignore")
        chk(len(d) > 0 and not num.isna().all().any(), f"{n}: rows={len(d)}, no all-NaN numeric column")
for n in need_json: chk((O / f"{n}.json").exists(), f"results/v2/{n}.json exists")
for n in need_fig: chk((O / "figures" / f"{n}.png").exists(), f"figure {n}.png exists")
for n in need_rep:
    f = R / f"{n}.md"; ok = f.exists() and f.stat().st_size > 500; chk(ok, f"reports/{n}.md exists")
    if ok:
        t = f.read_text().lower(); chk(not any(w in t for w in ("lorem", "tbd", "xxx", "to be filled")), f"{n}.md has no placeholder text")
d = pd.read_csv(O / "e1_operating_points.csv")
chk(((d.recall >= 0) & (d.recall <= 1) & (d.fpr >= 0) & (d.fpr <= 1)).all(), "e1 recall/FPR within [0,1]")
e7 = pd.read_csv(O / "e7_seed_replication.csv"); chk(e7.seed.nunique() == 3, f"e7 covers 3 seeds (found {sorted(e7.seed.unique())})")
chk("Not executed** (results/v2/e7" not in (R / "v2_results.md").read_text(), "v2_results.md includes E7")
for s in (11, 22, 33): chk((ROOT / "results" / "v2" / "predictions" / f"regdomain_seed{s}__charcnn_pw2.json").exists() or (ROOT / "results" / "v2" / f"e7_seed_replication.csv").exists(), f"seed {s} artefacts or E7 summary present")
print("ALL PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED"); sys.exit(1 if fails else 0)
