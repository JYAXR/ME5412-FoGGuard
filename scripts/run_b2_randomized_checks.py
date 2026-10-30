"""Plan random real clips, run receiver checks and inventory exact delivered weights."""

import argparse
import json
from pathlib import Path

from fogguard.evaluation.randomized_plan import build_plan, file_hash
from fogguard.evaluation.randomized_receiver import run_receiver_trials

ROOT = Path(__file__).resolve().parents[1]


def inventory(bundle, direction_model):
    lock = json.loads((ROOT / "docs/a_group/TEST_LOCK.json").read_text("utf-8"))
    rows = []
    for name in ("ankle_history_rf", "bilateral_rf", "ankle_cnn", "current_direction"):
        path = (direction_model if name == "current_direction" else
                bundle / name / "model.joblib" if bundle else None)
        expected = lock["files_sha256"]["models/" + name]
        actual = file_hash(path) if path is not None and path.is_file() else None
        rows.append({"model": name, "expected_filename":
                     "current/model.joblib" if name == "current_direction" else name + "/model.joblib",
                     "status": "MISSING" if actual is None else "HASH_MATCH" if actual == expected
                     else "HASH_MISMATCH", "expected_sha256": expected, "actual_sha256": actual})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--direction-model", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Preserve evidence: use a new output directory")
    args.output_dir.mkdir(parents=True)
    protocol_path = ROOT / "config/b2_randomized_evaluation.json"
    protocol = json.loads(protocol_path.read_text("utf-8"))
    plan = build_plan(args.raw, protocol_path, ROOT / "config/fog_selected.json",
                      args.output_dir / "sample_plan")
    receiver = run_receiver_trials(args.output_dir / "receiver", seed=protocol["seed"],
                                   trials=protocol["receiver_trials"],
                                   observations=protocol["receiver_source_observations_per_record"])
    weights = inventory(args.bundle, args.direction_model)
    result = {"schema": "fogguard.b2.randomized_readiness.v1", "status": "PARTIAL",
              "sampling": {"status": "READY", "clips": plan["clips"],
                           "records": plan["records"], "scored_windows": plan["scored_windows"],
                           "positive_windows": plan["positive_windows"],
                           "scope": "fixed development sample; not a new blind TEST"},
              "receiver": receiver, "models": weights,
              "model_accuracy": {"status": "NOT_RUN", "metrics": None,
                                 "reason": "Learned inference has not run. Readiness and receiver checks do not measure accuracy."}}
    (args.output_dir / "RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return int(receiver["fail"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
