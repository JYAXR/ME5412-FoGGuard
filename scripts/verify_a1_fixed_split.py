"""Audit frozen split, validation choices, hashes and all TEST stress outcomes."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json
from run_a1_fallback import validate_mixed_windows
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from verify_a1_event import old_interval_check

from fogguard.data.fogstar import iter_fogstar_windows, sensor_columns
from fogguard.evaluation.a1_fixed_split import (
    check_split,
    perturb_sensors,
    role_frame,
    training_noise_statistics,
)
from fogguard.modeling.a1_fallback import route_available_predictions
from fogguard.modeling.a1_fog import A1FogModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path)
    args = parser.parse_args()
    root, repository = args.run_dir, Path(__file__).resolve().parents[1]
    test_dir = args.test_dir or root / "test_results"
    config = json.loads((root / "frozen_config.json").read_text("utf-8"))
    lock = json.loads((root / "TRAINING_LOCK.json").read_text("utf-8"))
    completed = json.loads((test_dir / "TEST_COMPLETED.json").read_text("utf-8"))
    started = json.loads((test_dir / "TEST_EVALUATION_STARTED.json").read_text("utf-8"))
    check_lock(root, lock)
    if (sha(root / "TRAINING_LOCK.json") != started["training_lock_sha256"]
            or started["training_lock_sha256"] != completed["training_lock_sha256"]
            or datetime.fromisoformat(lock["locked_utc"]) > datetime.fromisoformat(started["created_utc"])):
        raise ValueError("Model lock did not precede testing or was modified")
    raw = pd.read_csv(args.raw)
    protocol = json.loads((root / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Wrong raw input")
    check_split(config["split"], raw.subjectID.unique())
    train = role_frame(raw, config["split"], "train")
    test = role_frame(raw, config["split"], "test")
    noise = training_noise_statistics(train, sensor_columns(("ankleL", "back")), config["split"]["train"])
    recorded = json.loads((root / "training_noise_stats.json").read_text("utf-8"))
    for name, value in noise.items():
        if abs(recorded[name] - value) > 1e-12:
            raise ValueError("Noise scale not derived from TRAIN")
    choices = {}
    grid = config["threshold_grid"]
    for model_id in config["models"]:
        directory = root / model_id
        detector = A1FogModel.load(directory / "model.joblib")
        if (set(detector.fit_participants) != set(config["split"]["train"])
                or set(detector.threshold_participants) != set(config["split"]["validation"])):
            raise ValueError("Wrong model training/validation people")
        validation = pd.read_csv(directory / "validation_predictions.csv", float_precision="round_trip")
        if set(validation.subject_id.astype(int)) != set(config["split"]["validation"]):
            raise ValueError("Wrong validation people")
        audit = json.loads((directory / "validation_selection.json").read_text("utf-8"))
        options = audit["threshold_scores"]
        if len(options) != grid["count"]:
            raise ValueError("Wrong validation threshold grid")
        for row, threshold in zip(options, np.linspace(grid["start"], grid["stop"], grid["count"]), strict=True):
            predicted = validation.fog_probability >= threshold
            if (abs(row["threshold"] - threshold) > 1e-12
                    or abs(row["fog_f1"] - f1_score(validation.truth_fog_label, predicted)) > 1e-12
                    or abs(row["balanced_accuracy"] - balanced_accuracy_score(validation.truth_fog_label, predicted)) > 1e-12):
                raise ValueError("Validation selection not backed by saved scores")
        selected = max(options, key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
        if selected != audit["selected"] or abs(detector.threshold - selected["threshold"]) > 1e-12:
            raise ValueError("Wrong frozen threshold")
        if abs(average_precision_score(validation.truth_fog_label, validation.fog_probability) - audit["average_precision"]) > 1e-12:
            raise ValueError("Wrong validation AP")
        choices[model_id] = audit
    eligible = [m for m in config["models"] if m != "rf_ankle"]
    winner = max(enumerate(eligible), key=lambda pair: (
        choices[pair[1]]["selected"]["fog_f1"], choices[pair[1]]["average_precision"], -pair[0]))[1]
    if winner != lock["selection"]["primary_model"] or winner != completed["selected_primary_unchanged"]:
        raise ValueError("Primary choice changed using TEST")
    rows = json.loads((test_dir / "all_metrics.json").read_text("utf-8"))
    expected_keys = {(s, m) for s in config["stress_scenarios"] for m in [*config["models"], "joint_validation_selected"]}
    if len(rows) != len(expected_keys) or {(r["scenario"], r["model"]) for r in rows} != expected_keys:
        raise ValueError("Missing declared test model/scenario")
    checked_rows = 0
    for scenario in config["stress_scenarios"]:
        predictions = {}
        transformed = perturb_sensors(test, scenario, recorded, config["stress_seed"])
        expected_support = {}
        for kind, positions in [("primary", ("ankleL", "back")), ("ankle", ("ankleL",))]:
            windows = list(iter_fogstar_windows(transformed, sensor_positions=positions, max_interpolation_gap_s=0))
            expected_support[kind] = pd.DataFrame({"subject_id": [int(w.subject_id) for w in windows],
                "session_id": [int(w.session_id) for w in windows], "task_id": [int(w.task_id) for w in windows],
                "window_start_s": [w.start_s for w in windows], "window_end_s": [w.end_s for w in windows]})
        for model_id in [*config["models"], "joint_validation_selected"]:
            directory = test_dir / scenario / model_id
            frame = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
            if (set(frame.subject_id.astype(int)) - set(config["split"]["test"])
                    or not frame.data_role.eq("test").all()
                    or not frame.stress_scenario.eq(scenario).all()
                    or not np.array_equal(frame.is_fog, frame.fog_probability >= frame.fog_threshold)
                    or not np.array_equal(frame.available_at_s, frame.window_end_s)):
                raise ValueError("Invalid TEST prediction/provenance")
            if model_id != "joint_validation_selected":
                cutoff = choices[model_id]["selected"]["threshold"]
                if not np.allclose(frame.fog_threshold, cutoff, atol=1e-12, rtol=0):
                    raise ValueError("Threshold tuned on TEST or stress")
                predictions[model_id] = frame
                keys = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s"]
                pd.testing.assert_frame_equal(frame[keys], expected_support["ankle" if model_id == "rf_ankle" else "primary"],
                                               atol=1e-6, rtol=0)
            else:
                rebuilt = route_available_predictions(predictions[winner], predictions["rf_ankle"])
                pd.testing.assert_frame_equal(rebuilt, frame, check_dtype=False)
            checked_rows += validate_mixed_windows(test, frame)
            expected = next(r for r in rows if r["scenario"] == scenario and r["model"] == model_id)
            actual = old_interval_check(test, frame.assign(event_fog=frame.is_fog), repository)
            actual.update({"precision": precision_score(frame.truth_fog_label, frame.is_fog, zero_division=0),
                           "recall": recall_score(frame.truth_fog_label, frame.is_fog, zero_division=0),
                           "fog_f1": f1_score(frame.truth_fog_label, frame.is_fog, zero_division=0)})
            for name, value in actual.items():
                if abs(value - expected[name]) > 1e-6:
                    raise ValueError(f"Independent TEST metric mismatch: {scenario}/{model_id}/{name}")
            runtime = pd.read_csv(directory / "runtime.csv", float_precision="round_trip")
            pd.testing.assert_frame_equal(runtime, frame.drop(columns=[c for c in frame if c.startswith("truth_")]))
    for name, digest in completed["files"].items():
        if sha(test_dir / name) != digest:
            raise ValueError(f"TEST artifact changed: {name}")
    check_lock(root, lock)
    result = {"status": "PASS", "train_people": config["split"]["train"],
              "validation_people": config["split"]["validation"], "test_people": config["split"]["test"],
              "selected_before_test": winner, "models": len(config["models"]),
              "test_model_scenarios": len(rows), "raw_test_records": test.groupby(["subjectID", "sessionID", "taskID"]).ngroups,
              "raw_window_rows_checked_including_repeated_stress_and_models": checked_rows,
              "frozen_training_files_verified": len(lock["files"]), "test_files_verified": len(completed["files"]),
              "model_hashes_unchanged_after_testing": True, "test_driven_refits": 0,
              "scope": "Saved artifacts, raw support and old interval cross-check; no independent retraining"}
    write_json(test_dir / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
