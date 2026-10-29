"""Read-only evaluation of locked 66D versus 78D bilateral RFs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from report_a1_author_cnn import event_summary
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_author import annotate, grouped_diagnostics, matched_frames, metrics
from fogguard.evaluation.a1_fixed_split import role_frame
from fogguard.modeling.a1_bilateral import anchored_gyro_windows
from fogguard.modeling.a1_coordination import CoordinationFoG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    output = run / "evaluation"
    if output.exists():
        parser.error("Use a run without prior evaluation")
    config, split, lock, protocol = [json.loads((run / n).read_text("utf-8")) for n in ("config.json", "split.json", "MODEL_LOCK.json", "protocol.json")]
    check_lock(run, lock)
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Raw data changed")
    for name, digest in protocol["source_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Source changed: {name}")
    reference = Path(config["reference_run"])
    check_lock(reference / "evaluation", json.loads((reference / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8")))
    output.mkdir()
    write_json(output / "VALIDATION_STARTED.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "model_lock_sha256": sha(run / "MODEL_LOCK.json")})
    raw = pd.read_csv(args.raw)
    train, validation = [role_frame(raw, split, role) for role in ("train", "validation")]
    del raw
    windows = anchored_gyro_windows(validation, require_right=True)
    frames = {name: annotate(windows, CoordinationFoG.load(run / name / "model.joblib").predict_windows(windows), "validation") for name in config["models"]}
    old = pd.read_csv(reference / "evaluation/bilateral_rf/predictions.csv", float_precision="round_trip")
    np.testing.assert_allclose(frames["bilateral_control"].fog_probability, old.fog_probability, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(frames["bilateral_control"].is_fog, old.is_fog)
    for name in ("previous_author", "reference_rf_context", "reference_rf_ankle", "left_natural"):
        frames[name] = pd.read_csv(reference / "evaluation" / name / "predictions.csv", float_precision="round_trip")
    rows = []
    for name, frame in frames.items():
        dest = output / name
        dest.mkdir()
        frame.to_csv(dest / "predictions.csv", index=False)
        frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(dest / "runtime.csv", index=False)
        rows.append({"model": name, "pool": "validation_own", **metrics(frame), **event_summary(validation, frame, dest, root)})
        for column in ("subject_id", "truth_activity", "task_id"):
            grouped_diagnostics(frame, column).to_csv(dest / f"by_{column}.csv", index=False)
    for name in config["models"]:
        frame = pd.read_csv(run / name / "oof_predictions.csv", float_precision="round_trip")
        dest = output / f"train_oof_{name}"
        dest.mkdir()
        rows.append({"model": name, "pool": "train_oof_calibration", **metrics(frame), **event_summary(train, frame, dest, root)})
    for name, frame in matched_frames(frames).items():
        rows.append({"model": name, "pool": "all_common", **metrics(frame)})
        frame.to_csv(output / f"common_{name}.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    own = table.loc[table.pool.eq("validation_own")].set_index("model")
    base, candidate = own.loc["bilateral_control"], own.loc["coordination_rf"]
    conditions = {"matched_f1_higher": bool(candidate.fog_f1 > base.fog_f1),
        "false_alarm_seconds_not_higher": bool(candidate.false_alarm_s <= base.false_alarm_s + 1e-6),
        "fog_time_recall_not_lower": bool(candidate.time_recall + 1e-9 >= base.time_recall),
        "decision_support_not_lost": bool(candidate.decision_support_fraction + 1e-9 >= base.decision_support_fraction)}
    gate = {"conditions": conditions, "passed": all(conditions.values()), "default_replaced": False,
            "old_66d_validation_reproduced": True, "scope": "Previously inspected development cohort, no new TEST"}
    write_json(output / "development_gate.json", gate)
    check_lock(run, lock)
    write_json(output / "EVALUATION_COMPLETED.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json"), "validation_people": split["validation"],
        "test_windows_constructed": 0, "validation_driven_refits": 0,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print(table[["model", "pool", "windows", "precision", "recall", "fog_f1", "false_alarm_s", "time_recall"]].to_string(index=False), flush=True)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
