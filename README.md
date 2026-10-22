# Phishing URL detection: leakage-resistant evaluation, cost-aware decisions, sequential RL and safe adaptation

A reproducible research project built on the supplied dataset `Phishing_Detection_Final_Dataset_Adaptive.zip`
(SHA-256 `75498f52a089e4fef9dd11eed0f261a9dfa3c162a2fa62f07d78bc10c60180f6`). All reported numbers are produced by
the scripts in this repository from that data; nothing is typed in by hand except the clearly marked
interpretation section (`reports/interpretation.md`, shown as section 14 of the final report).

**Start with `reports/final_research_report.md`.** Limitations are in `reports/limitations.md`.

## Headline findings (details, uncertainty and caveats are in the reports)

- **Supplied "domain-disjoint" splits are only hostname-disjoint.** Thousands of registered domains are shared between
  train and test, so registered-domain-disjoint splits were re-derived (Public Suffix List based).
- **Format shortcuts inflate scores** (e.g. `www.` appears in about 99.9% of legitimate but about 28% of phishing URLs).
  Primary results use URLs with `www.` normalised away; results on the URLs as supplied are reported separately.
- **Domain-disjoint evaluation lowers PR-AUC for every model** compared with a random split, and calibration degrades
  sharply on the external "recent" set and on cross-source tests.
- **Sequential RL did not beat strong non-learning baselines** in the simulated inspection environment (13 of 13 scenario
  cells worse than the best baseline). This is reported as a negative result for this tabular agent and these assumptions.
- **The adaptation gate was over-conservative.** It blocked poisoned updates but also every clean update, because its
  suspicious-feedback check cannot separate genuine shift from poisoning.
- **No temporal claims.** No file has timestamps; the "recent" set is a different source, not a later period.

## Result status

| component | status |
|---|---|
| Dataset audit, splits, leakage checks, supervised baselines, calibration, bootstrap CIs | Executed and verified |
| Static cost-sensitive policies | Executed and verified (costs are assumptions) |
| Sequential RL environment and comparison | Simulated (full configured budget) |
| Distribution shift | Observed shifts executed and verified; `sim:` rows simulated |
| Safe adaptation | Simulated (offline, non-adaptive attacker) |
| Chronological evaluation | Not supported by the available data |
| XGBoost, neural URL models | Not executed (no package-index access in the build environment; HistGradientBoosting used instead) |

## Setup and reproduction

Python 3.13 (tested), CPU only, 2 cores / 7 GB RAM. Per-stage wall times of the recorded run are in `results/run_status.json` (training was partly resumed from cached predictions in that run, so the training time there is not a from-scratch figure).

```
pip install -r requirements.txt
mkdir -p data/raw && unzip /path/to/Phishing_Detection_Final_Dataset_Adaptive.zip -d data/raw   # dataset is NOT bundled
python scripts/run_all.py                      # all stages; logs/ and results/run_status.json record exit status
python scripts/run_all.py --only audit prepare tests integrity   # quick re-check
python -m unittest discover -s tests -v        # 60 tests, plain unittest (no pytest needed)
python scripts/predict_url.py http://example.com/login   # offline prediction: class, risk score, ALLOW/INSPECT/BLOCK
```

`run_all.py` aborts with a non-zero exit code if any stage fails. Trained prediction caches are reused only if the code,
dataset hash, split checksum and configuration match. `--fast` runs reduced-budget RL and adaptation stages and labels them as such.
Configuration, seeds and the **assumed** cost matrix are in `config/config.yaml`.

## Layout

`src/` library code (data, features, models, evaluation, decision, rl, monitoring, adaptation, inference) ·
`scripts/` stage entry points · `tests/` unit and end-to-end tests (synthetic fixtures are used for software tests only) ·
`reports/` generated reports · `results/` CSV/JSON/figures/predictions · `artifacts/` primary-split models and the
inference bundle · `MANIFEST.json` file hashes and execution status.

## Caveats

The prediction CLI is an offline research prototype that reads only the URL string. It makes no network requests and is not
production-validated. Literature evidence levels (full text / abstract / metadata only) are recorded per source in
`reports/literature_review.md`; no novelty is claimed.
