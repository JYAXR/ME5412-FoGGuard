"""Audit locked files, independent raw window inventory, roles and saved scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json
from sklearn.metrics import balanced_accuracy_score, f1_score

from fogguard.evaluation.a1_author import matched_frames, metrics
from fogguard.evaluation.a1_event import EventSupport
from fogguard.evaluation.a1_fixed_split import role_frame
from fogguard.modeling.a1_fog import A1FogModel

IDS = ["subject_id", "session_id", "task_id"]
RAW_IDS = ["subjectID", "sessionID", "taskID"]


def raw_inventory(raw, channels):
    """Enumerate eligible endpoints independently of all production iterators."""
    rows = []
    for key, group in raw.groupby(RAW_IDS, sort=False):
        t = group.timestamp.to_numpy(float)
        valid = np.isfinite(group[channels].to_numpy(float)).all(axis=1)
        run_start = None
        for i in range(len(t)):
            if not valid[i]:
                run_start = None
                continue
            if run_start is None or (i and t[i] - t[i - 1] > 1.5 / 60):
                run_start = i
            count = i - run_start + 1
            if count >= 120 and (count - 120) % 30 == 0:
                start = i - 119
                fraction = float(group.fog.iloc[start:i + 1].mean())
                rows.append((*map(int, key), t[start], t[i], int(fraction >= 0.5), fraction))
    return pd.DataFrame(rows, columns=[*IDS, "window_start_s", "window_end_s", "truth_fog_label", "truth_fog_fraction"])


def verify_frame(raw, frame, channels, people):
    expected = raw_inventory(raw, channels)
    if len(expected) != len(frame) or set(frame.subject_id.astype(int)) - set(people):
        raise ValueError("Wrong people or missing/extra eligible windows")
    for column in expected:
        if not np.allclose(expected[column].astype(float), frame[column].astype(float), atol=1e-6, rtol=0):
            raise ValueError(f"Raw window mismatch: {column}")
    if not np.allclose(frame.available_at_s, frame.window_end_s, atol=1e-9, rtol=0):
        raise ValueError("Backdated decision")
    if not np.array_equal(frame.is_fog, frame.fog_probability >= frame.fog_threshold):
        raise ValueError("Saved decisions differ from threshold")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    lock = json.loads((run / "MODEL_LOCK.json").read_text("utf-8"))
    report_lock = json.loads((run / "report/REPORT_COMPLETED.json").read_text("utf-8"))
    protocol = json.loads((run / "protocol.json").read_text("utf-8"))
    check_lock(run, lock)
    check_lock(run / "report", report_lock)
    if sha(args.raw) != protocol["raw_sha256"] or sha(run / "MODEL_LOCK.json") != report_lock["model_lock_sha256"]:
        raise ValueError("Raw or model lock changed")
    for name, digest in protocol["source_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Source changed after fitting: {name}")
    config = json.loads((run / "split_config.json").read_text("utf-8"))
    cnn_config = json.loads((run / "config.json").read_text("utf-8"))
    split = config["split"]
    if any(set(split[a]) & set(split[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("Data role overlap")
    if protocol["data_roles_constructed"] != ["train", "validation"] or lock["test_windows_constructed"] != 0:
        raise ValueError("Unexpected TEST construction")
    raw = pd.read_csv(args.raw)
    roles = {role: role_frame(raw, split, role) for role in ("train", "validation")}
    del raw
    names = ["rf_gyro_2s", "author_cnn", "reference_rf_context", "reference_rf_ankle", "train_oof_rf_context"]
    frames, checked = {}, 0
    for name in names:
        frame = pd.read_csv(run / name / "predictions.csv")
        role = "train" if name == "train_oof_rf_context" else "validation"
        if name in ("rf_gyro_2s", "author_cnn"):
            channels = [f"ankleL_gyro_{a}" for a in "xyz"]
        else:
            positions = ["ankleL"] if name == "reference_rf_ankle" else ["ankleL", "back"]
            channels = [f"{p}_{s}_{a}" for p in positions for s in ("acc", "gyro") for a in "xyz"]
        checked += verify_frame(roles[role], frame, channels, split[role])
        runtime = pd.read_csv(run / name / "runtime.csv")
        if any(c.startswith("truth_") for c in runtime) or not np.allclose(runtime.fog_probability, frame.fog_probability, atol=1e-12):
            raise ValueError("Runtime has truth or differs from scores")
        if set(frame.data_role) != ({"train_oof"} if role == "train" else {"validation"}):
            raise ValueError("Wrong saved role")
        frames[name] = frame
    folds = json.loads((run / "train_oof_rf_context/folds.json").read_text("utf-8"))
    seen = []
    for fold in folds:
        fit, held = set(fold["fit_people"]), set(fold["held_out_people"])
        if fit & held or fit | held != set(split["train"]):
            raise ValueError("OOF fit overlap or incomplete TRAIN complement")
        all_oof = frames["train_oof_rf_context"]
        part = all_oof.loc[all_oof.held_out_fold.eq(fold["fold"])]
        if set(part.subject_id.astype(int)) != held or len(part) != fold["held_out_windows"]:
            raise ValueError("OOF held-out membership mismatch")
        seen.extend(held)
    if sorted(seen) != sorted(split["train"]):
        raise ValueError("OOF people not held out exactly once")
    for name in names[:2]:
        metadata = json.loads((run / name / "fit_metadata.json").read_text("utf-8"))
        model = A1FogModel.load(run / name / "model.joblib")
        if list(model.fit_participants) != split["train"] or list(model.threshold_participants) != split["validation"] or metadata["test_people_used"]:
            raise ValueError("Model fit provenance mismatch")
        inferred = model.predict_frame(roles["validation"].drop(columns=["fog", "activity", "fog_severity"]))
        np.testing.assert_allclose(inferred.fog_probability, frames[name].fog_probability, atol=1e-7, rtol=0)
        frame = frames[name]
        grid = cnn_config["threshold_grid"]
        options = []
        for cutoff in np.linspace(grid["start"], grid["stop"], grid["count"]):
            p = frame.fog_probability >= cutoff
            options.append((float(f1_score(frame.truth_fog_label, p, zero_division=0)),
                            float(balanced_accuracy_score(frame.truth_fog_label, p)), float(cutoff)))
        best = max(options)
        selection = json.loads((run / name / "threshold_selection.json").read_text("utf-8"))["selected"]
        np.testing.assert_allclose([selection["fog_f1"], selection["balanced_accuracy"], model.threshold], best, atol=1e-12)
    history = pd.read_csv(run / "author_cnn/training_history.csv")
    best, epoch = float("inf"), None
    for row in history.itertuples():
        if row.validation_loss < best - cnn_config["cnn"]["min_delta"]:
            best, epoch = row.validation_loss, row.epoch
    metadata = json.loads((run / "author_cnn/fit_metadata.json").read_text("utf-8"))
    if metadata["best_epoch"] != epoch or metadata["parameters"] != 8353 or not metadata["trained_from_random_weights"]:
        raise ValueError("CNN architecture/checkpoint provenance differs")
    expected = {(name, "own_support"): frame for name, frame in frames.items()}
    for pool, subset in [("gyro_pair_common", names[:2]), ("all_four_common", names[:4])]:
        expected.update({(name, pool): frame for name, frame in matched_frames({n: frames[n] for n in subset}).items()})
    comparison = pd.read_csv(run / "report/comparison.csv")
    for row in comparison.itertuples():
        frame = expected[row.model, row.pool]
        for field, value in metrics(frame).items():
            if value is not None and not np.isclose(getattr(row, field), value, atol=1e-10, rtol=0):
                raise ValueError(f"Saved comparison differs: {row.model}/{row.pool}/{field}")
        if row.pool == "own_support":
            role = "train" if row.model == "train_oof_rf_context" else "validation"
            support = EventSupport.build(roles[role], frame, allowed=split[role])
            for field, value in support.metrics(frame.is_fog).items():
                if not np.isclose(getattr(row, field), value, atol=1e-6, rtol=0):
                    raise ValueError(f"Event summary differs: {field}")
    result = {"status": "PASS", "verified_window_rows": checked, "data_roles": ["train_oof", "validation"],
              "test_windows_constructed": 0, "independent_retraining": False,
              "fit_role_and_oof_membership": "PASS", "raw_inventory_and_labels": "PASS",
              "validation_threshold_and_checkpoint": "PASS", "saved_model_rescoring": "PASS",
              "runtime_truth_independence": "PASS", "metrics_and_time_support": "PASS",
              "model_files_checked": len(lock["files"]), "report_files_checked": len(report_lock["files"]),
              "model_lock_sha256": sha(run / "MODEL_LOCK.json")}
    write_json(run / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
