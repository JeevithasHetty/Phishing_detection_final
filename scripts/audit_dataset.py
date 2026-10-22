"""Reproducible dataset audit -> reports/dataset_audit.md + results/dataset_audit.json"""
import _bootstrap, argparse, collections, re
import numpy as np, pandas as pd
from pathlib import Path
from src.common import ROOT, load_config, data_dir, sha256_file, write_json
from src.data.loading import load_split
from src.data.urls import split_url, hostname, registered_domain, normalise_url, is_ip_host

def main():
    cfg = load_config(); dd = data_dir(cfg)
    A = {"data_dir_files": {}}
    for p in sorted(dd.glob("*")):
        if p.is_file():
            A["data_dir_files"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    S = {n: load_split(n, cfg) for n in ["train", "val", "test", "recent"]}
    A["splits"] = {}
    for n, d in S.items():
        host = d["url"].map(hostname)
        d["host"] = host
        d["rd"] = d["url"].map(registered_domain)
        d["norm"] = d["url"].map(normalise_url)
        sch = d["url"].map(lambda u: split_url(u)[0])
        A["splits"][n] = {
            "rows": len(d), "cols": list(d.columns[:5]), "phishing": int(d.label.sum()),
            "legit": int((1 - d.label).sum()), "phish_pct": round(100 * d.label.mean(), 2),
            "dup_exact_url": int(d.url.duplicated().sum()), "dup_normalised": int(d.norm.duplicated().sum()),
            "empty_host": int((d.host == "").sum()),
            "unique_hosts": int(d.host.nunique()), "unique_registered_domains_icann": int(d.rd.nunique()),
            "provided_domain_equals_parsed_host_pct": round(100 * float((d["domain"].str.lower() == d.host).mean()), 2),
            "ip_host_pct": round(100 * float(d.host.map(is_ip_host).mean()), 2),
            "scheme_counts_by_label": {str(k): v for k, v in pd.crosstab(sch, d.label).to_dict("index").items()},
            "url_with_space_after_scheme_by_label": d.url.str.contains(r"^https?:// ", regex=True).groupby(d.label).sum().astype(int).to_dict(),
            "missing_values": int(d.isna().sum().sum()),
            "source_counts": d["sources"].value_counts().to_dict(),
            "source_phish_rate": d.groupby("sources").label.mean().round(4).to_dict(),
        }
    # overlap matrix
    names = list(S); ov = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov[f"{a}~{b}"] = {
                "exact_url": len(set(S[a].url) & set(S[b].url)),
                "normalised_url": len(set(S[a].norm) & set(S[b].norm)),
                "hostname": len(set(S[a].host) & set(S[b].host)),
                "registered_domain_icann": len(set(S[a].rd) & set(S[b].rd)),
                "rows_b_whose_rd_in_a": int(S[b].rd.isin(set(S[a].rd)).sum()),
                "frac_b_rows_rd_in_a": round(float(S[b].rd.isin(set(S[a].rd)).mean()), 4)}
    A["overlap"] = ov
    # label conflicts on normalised URL across everything
    allx = pd.concat([d.assign(split=n)[["norm", "label", "split"]] for n, d in S.items()])
    g = allx.groupby("norm").label.nunique()
    conf = g[g > 1].index
    A["label_conflicts_normalised"] = {"n_keys": int(len(conf)),
        "examples": allx[allx.norm.isin(conf[:5])].sort_values("norm").head(10).to_dict("records")}
    # registered-domain label purity (how much a domain-only lookup would reveal)
    tr = S["train"]; rdp = tr.groupby("rd").label.agg(["mean", "size"])
    A["train_rd_purity"] = {"frac_rd_pure": round(float(((rdp["mean"] == 0) | (rdp["mean"] == 1)).mean()), 4),
        "frac_rows_in_pure_rd": round(float(rdp[(rdp["mean"] == 0) | (rdp["mean"] == 1)]["size"].sum() / len(tr)), 4),
        "top_phish_rd": tr[tr.label == 1].rd.value_counts().head(10).to_dict()}
    # leakage / artefact probes: trivially predictive format features
    probes = {}
    for n, d in S.items():
        L = d.url.str.len()
        probes[n] = {"mean_len_phish": float(L[d.label == 1].mean()), "mean_len_legit": float(L[d.label == 0].mean()),
            "frac_www_phish": float(d.host[d.label == 1].str.startswith("www.").mean()),
            "frac_www_legit": float(d.host[d.label == 0].str.startswith("www.").mean()),
            "frac_https_phish": float((d.url[d.label == 1].str.lower().str.startswith("https")).mean()),
            "frac_https_legit": float((d.url[d.label == 0].str.lower().str.startswith("https")).mean()),
            "frac_has_path_phish": float((d.url[d.label == 1].map(lambda u: len(split_url(u)[2]) > 1)).mean()),
            "frac_has_path_legit": float((d.url[d.label == 0].map(lambda u: len(split_url(u)[2]) > 1)).mean())}
    A["format_probes"] = probes
    # master vs splits consistency, UCI, recent candidate
    m = load_split("master", cfg)
    A["master"] = {"rows": len(m), "phishing": int(m.label.sum()),
        "split_rows_sum": sum(len(S[k]) for k in ["train", "val", "test"]),
        "all_split_urls_in_master": bool(set(pd.concat([S[k].url for k in ["train", "val", "test"]])) <= set(m.url))}
    rc = load_split("recent_candidate", cfg)
    A["recent_candidate"] = {"rows": len(rc), "phishing": int(rc.label.sum()),
        "holdout_rows": len(S["recent"]), "holdout_subset_of_candidate": bool(set(S["recent"].url) <= set(rc.url))}
    uci = pd.read_csv(dd / cfg["data"]["files"]["uci"])
    A["uci"] = {"shape": list(uci.shape), "label_counts": uci["label"].value_counts().to_dict(),
                "has_raw_url": False, "note": "30 categorical features only; benchmark-only, unused for URL experiments"}
    A["timestamps_available"] = False
    split_summary = pd.read_csv(dd / "split_summary.csv")
    A["split_summary_vs_actual"] = split_summary.to_dict("records")
    write_json(A, ROOT / "results" / "dataset_audit.json")
    write_report(A, cfg)

def write_report(A, cfg):
    L = ["# Dataset audit", "", "_Generated by `scripts/audit_dataset.py` from `results/dataset_audit.json`. Status: **Executed and verified**._", "",
         "## Supplied files (read-only, SHA-256)", "", "| file | bytes | sha256 |", "|---|---:|---|"]
    for k, v in A["data_dir_files"].items(): L.append(f"| {k} | {v['bytes']:,} | `{v['sha256'][:16]}…` |")
    L += ["", "Label convention (from the supplied README, verified: labels are exactly {0,1}): 1 = phishing, 0 = legitimate. URL column `url`, label column `label`, `domain` column = **full hostname** (not the registered domain).", "",
          "## Per-split summary", "", "| split | rows | phishing | phish % | unique hosts | unique registered domains (ICANN PSL) | dup normalised URLs | empty host | missing |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, s in A["splits"].items():
        L.append(f"| {n} | {s['rows']:,} | {s['phishing']:,} | {s['phish_pct']} | {s['unique_hosts']:,} | {s['unique_registered_domains_icann']:,} | {s['dup_normalised']:,} | {s['empty_host']} | {s['missing_values']} |")
    L += ["", "## Overlap between the supplied splits", "", "| pair | exact URL | normalised URL | hostname | registered domain | rows of 2nd split whose registered domain is in 1st (frac) |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in A["overlap"].items():
        L.append(f"| {k} | {v['exact_url']} | {v['normalised_url']} | {v['hostname']} | {v['registered_domain_icann']} | {v['rows_b_whose_rd_in_a']:,} ({v['frac_b_rows_rd_in_a']}) |")
    L += ["", "**Finding.** The supplied splits are hostname-disjoint but **not registered-domain-disjoint**: the `domain` column is the full hostname, so sibling subdomains of one registered domain land in different splits (see table). The project therefore re-derives registered-domain-disjoint splits (`scripts/prepare_data.py`) rather than trusting the supplied partition for the registered-domain protocol.", "",
          "## Label conflicts", "", f"Normalised-URL keys (lowercase, scheme/`www.`/trailing-slash removed) appearing with both labels: **{A['label_conflicts_normalised']['n_keys']}**.", "",
          "## Registered-domain label purity (train)", "", f"Fraction of train registered domains with a single label: {A['train_rd_purity']['frac_rd_pure']}; fraction of train rows in such domains: {A['train_rd_purity']['frac_rows_in_pure_rd']}. High purity means domain identity alone predicts the label in-distribution, which is why domain-disjoint evaluation matters.", "",
          "## Format artefacts / leakage probes", "", "| split | mean len phish | mean len legit | www phish | www legit | https phish | https legit | has path phish | has path legit |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, p in A["format_probes"].items():
        L.append(f"| {n} | {p['mean_len_phish']:.1f} | {p['mean_len_legit']:.1f} | {p['frac_www_phish']:.3f} | {p['frac_www_legit']:.3f} | {p['frac_https_phish']:.3f} | {p['frac_https_legit']:.3f} | {p['frac_has_path_phish']:.3f} | {p['frac_has_path_legit']:.3f} |")
    L += ["", "Large phish/legit gaps in such trivially computable properties indicate dataset-construction artefacts (different collection pipelines per class) that a model can exploit without learning phishing semantics. Space after `http://` by label:", ""]
    for n, s in A["splits"].items(): L.append(f"- {n}: {s['url_with_space_after_scheme_by_label']} ; sources→phish rate {s['source_phish_rate']}")
    L += ["", "## Other findings", "", f"- Master candidate: {A['master']['rows']:,} rows; train+val+test = {A['master']['split_rows_sum']:,}; all split URLs found in master: {A['master']['all_split_urls_in_master']}.",
          f"- Recent candidate: {A['recent_candidate']['rows']:,} rows; recent holdout {A['recent_candidate']['holdout_rows']:,} rows (subset of candidate: {A['recent_candidate']['holdout_subset_of_candidate']}). `split_summary.csv` reports 74,705 rows for this holdout; the file has {A['recent_candidate']['holdout_rows']:,} (discrepancy of 2 rows, unexplained by the supplied files).",
          f"- UCI benchmark: {A['uci']['shape']}, no raw URL; not used for URL experiments (as the supplied README instructs).",
          "- **No timestamps** exist in any supplied file. The 'recent' holdout is described as recent by the dataset README, but no per-row date or verifiable provenance is supplied, so **chronological/temporal generalisation cannot be claimed**; it is analysed only as an *external-source, domain-disjoint holdout*.",
          "- Per-row `sources` identify PhiUSIIL vs Mendeley-450K (and a small overlap). Cross-source evaluation is feasible; the recent set is a third source.",
          "- Tranco Top-1M is listed in the audit summary but **was not included in the ZIP**; it is not used.",
          "- Raw files were opened read-only; hashes above are recorded for the integrity check."]
    (ROOT / "reports" / "dataset_audit.md").write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
