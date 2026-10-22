"""Usage: python scripts/predict_url.py <url> [<url> ...]   (offline; no network requests are made)"""
import _bootstrap, sys, json
from src.common import ROOT
from src.inference.predict import Detector
if __name__ == "__main__":
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    d = Detector.load(ROOT / "artifacts" / "deployable_bundle.joblib")
    print(json.dumps({"model": d.meta, "predictions": d.predict(sys.argv[1:])}, indent=2))
