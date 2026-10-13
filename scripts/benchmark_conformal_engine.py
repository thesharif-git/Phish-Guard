"""Display recomputed development metrics, without novelty or prevention claims."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.security_pipeline import security_pipeline_id



def run_benchmark():
    path = ROOT / "models/conformal_evaluation_results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    if results.get("schema_version") != 2 or results.get("pipeline_id") != security_pipeline_id():
        raise RuntimeError("Legacy/stale benchmark. Run scripts.calibrate_conformal_engine first.")
    print(json.dumps(results, indent=2))
    print("Quarantine recall and warning recall are separate; neither is measured attack prevention.")
    print("Fusion replaces Pillar 4 only. Reliability constants are assigned priors, not learned uncertainty.")

if __name__ == "__main__":
    run_benchmark()
