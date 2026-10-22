"""Write MANIFEST.json: SHA-256 of every packaged file, the supplied-dataset hashes, and the recorded run/integrity status.
Run after the pipeline; excludes the (non-bundled) dataset copy, processed caches and bytecode."""
import _bootstrap, json, datetime
from src.common import ROOT, sha256_file, read_json

EXCLUDE_DIRS = {"data/raw", "artifacts/processed", "__pycache__", "results/predictions_smoke"}

def packaged_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.name in ("MANIFEST.json",): continue
        rel = p.relative_to(ROOT).as_posix()
        if any(rel.startswith(d + "/") or f"/{d}/" in f"/{rel}" for d in EXCLUDE_DIRS) or p.suffix == ".pyc": continue
        yield rel, p

def main():
    audit = read_json(ROOT / "results" / "dataset_audit.json")
    st = read_json(ROOT / "results" / "run_status.json"); integ = read_json(ROOT / "results" / "integrity_report.json")
    files = {rel: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for rel, p in packaged_files()}
    m = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
         "supplied_zip_sha256": "75498f52a089e4fef9dd11eed0f261a9dfa3c162a2fa62f07d78bc10c60180f6",
         "note": "The supplied dataset is not bundled; extract it into data/raw (see README). Original files are read-only inputs.",
         "pipeline_mode": st.get("mode"), "pipeline_all_ok": st.get("all_ok"), "integrity_all_passed": integ.get("all_passed"),
         "stages": [{k: s[k] for k in ("stage", "exit_status", "seconds", "mode")} for s in st["stages"]],
         "n_files": len(files), "files": files}
    (ROOT / "MANIFEST.json").write_text(json.dumps(m, indent=1))
    print(f"MANIFEST.json written: {len(files)} files")

if __name__ == "__main__": main()
