"""Figures for the v2 results, drawn from results/v2/*.csv only (no numbers typed in).
Palette: first three categorical slots of the validated reference palette (blue, orange, aqua); text in ink tokens; value labels on marks."""
import _bootstrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.common import ROOT

OUT = ROOT / "results" / "v2"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"; C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
NICE = {"v1_hybrid_lr": "v1 hybrid LR", "v1_stack": "v1 stacked", "v1_tfidf_lr": "v1 char TF-IDF", "xlin_pw1_base": "XLin (no class weight)", "xlin_pw2_base": "XLin base (w=2)",
        "xlin_pw2_canon": "XLin + canonical host", "xlin_pw2_full": "XLin + typosquat", "xlin_pw2_aug": "XLin + padding aug.", "charcnn": "Char-CNN", "ens_xlin+cnn": "Ensemble XLin+CNN"}

def style(ax):
    ax.set_facecolor(SURF); [ax.spines[s].set_visible(False) for s in ("top", "right", "left")]; ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)

def hbars(ax, labels, series, names, colors, fmt="{:.2f}", xlim=(0, 1)):
    n = len(series); h = 0.8 / n; y = np.arange(len(labels))
    for i, (vals, nm, c) in enumerate(zip(series, names, colors)):
        yy = y + (i - (n - 1) / 2) * h; ax.barh(yy, vals, height=h * 0.85, color=c, label=nm)
        for v, yv in zip(vals, yy): ax.text(v + 0.008, yv, fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, color=INK); ax.invert_yaxis(); ax.set_xlim(*xlim); style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1 if n>2 else 2, labelcolor=INK2)

def fig_url_vs_domain():
    d = pd.read_csv(OUT / "e1_operating_points.csv"); t = d[(d.split == "test") & (d.operating_point == "fpr1%")].set_index("model")
    ms = ["v1_hybrid_lr", "v1_stack", "xlin_pw2_base", "xlin_pw2_aug", "charcnn", "ens_xlin+cnn"]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURF)
    hbars(ax, [NICE[m] for m in ms], [t.loc[ms, "recall"].values, t.loc[ms, "domain_recall"].values], ["URL-level recall", "Domain-level recall"], [C1, C2])
    ax.set_title("Recall at the validation-derived ~1% FPR threshold (domain-disjoint test)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("recall  (domain-level: each phishing registered domain counts once)", color=INK2, fontsize=9); fig.tight_layout(); fig.savefig(FIG / "v2_recall_url_vs_domain.png", dpi=140, facecolor=SURF); plt.close(fig)

def fig_padding():
    a = pd.read_csv(OUT / "e6_obfuscation_attacks.csv"); ms = ["v1_hybrid_lr", "xlin_pw2_base", "xlin_pw2_aug", "xlin_pw2_full", "charcnn", "ens_xlin+cnn"]
    g = lambda key: [a[(a.model == m) & (a.attack.str.startswith(key))].iloc[0] for m in ms]
    clean = [r.recall_clean for r in g("benign path padding x2")]; fixed = [r.recall_attacked for r in g("benign path padding x2")]; unseen = [r.recall_attacked for r in g("benign path padding, unseen")]
    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=SURF)
    hbars(ax, [NICE[m] for m in ms], [clean, fixed, unseen], ["clean phishing URLs", "padded (vocabulary seen in augmentation style)", "padded (unseen vocabulary)"], [C1, C2, C3])
    ax.set_title("Phishing recall under benign-path padding (SIMULATED attack, same thresholds)", fontsize=10, color=INK, loc="left"); ax.set_xlabel("recall at the validation-derived ~1% FPR threshold (URL-level)", color=INK2, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "v2_padding_attack.png", dpi=140, facecolor=SURF); plt.close(fig)

def fig_recall_control():
    d = pd.read_csv(OUT / "e2_recall_control.csv"); m = d[(d.model == "ens_xlin+cnn") & (d.alpha == 0.05)].sort_values("n_cal")
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURF)
    for meth, c, nm in [("naive_quantile", C2, "naive empirical quantile"), ("ltt_delta0.1", C1, "finite-sample bound (delta = 0.1)")]:
        s = m[m.method == meth]; ax.plot(s.n_cal, s.test_violation_rate, color=c, lw=2, marker="o", ms=6, label=nm)
        for x, y in zip(s.n_cal, s.test_violation_rate): ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK)
    ax.axhline(0.1, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(ax.get_xlim()[0] * 1.05, 0.11, "delta = 0.1", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.1); ax.set_xlabel("labelled phishing calibration domains (one URL per domain)", color=INK2, fontsize=9)
    ax.set_ylabel("share of calibration draws that miss the 95% recall target", color=INK2, fontsize=9); style(ax); ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_title("Recall target 95%: how often the chosen threshold misses it (test, domain-level)", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    fig.tight_layout(); fig.savefig(FIG / "v2_recall_control_small_n.png", dpi=140, facecolor=SURF); plt.close(fig)

def fig_leakage():
    d = pd.read_csv(OUT / "e4_leakage_demo.csv"); short = ["A: oversample, then random split\n(leaky)", "B: random split", "C: domain-disjoint split", "D: domain split, oversample\ntrain only (correct)"]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF)
    hbars(ax, short, [d.recall.values, d.test_urls_also_in_train.values], ["recall on held-out part (threshold 0.5)", "share of test URLs that also appear in train"], [C1, C2])
    ax.set_title("Evaluation protocol changes the headline number (same model, 150k-URL sample)", fontsize=10, color=INK, loc="left"); fig.tight_layout(); fig.savefig(FIG / "v2_leakage_demo.png", dpi=140, facecolor=SURF); plt.close(fig)

if __name__ == "__main__":
    for f in (fig_url_vs_domain, fig_padding, fig_recall_control, fig_leakage): f(); print("made", f.__name__)
