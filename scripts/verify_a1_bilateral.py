"""Independent left-anchored inventory, pure failure-pair and frozen RF audit."""

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_author import matched_frames, metrics
from fogguard.evaluation.a1_bilateral import support_summary
from fogguard.evaluation.a1_event import EventSupport
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.modeling.a1_bilateral import (
    LEFT,
    RIGHT,
    anchored_gyro_windows,
    left_view,
    matched_features,
)
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix

KEYS = ["subject_id", "session_id", "task_id", "window_start_s", "window_end_s"]


def inventory(raw):
    rows = []
    for key, group in raw.groupby(["subjectID", "sessionID", "taskID"], sort=False):
        t, left, right = group.timestamp.to_numpy(float), group[list(LEFT)].to_numpy(float), group[list(RIGHT)].to_numpy(float)
        truth, activity = group.fog.to_numpy(int), group.activity.to_numpy(int)
        start = None
        for i in range(len(t)):
            if not np.isfinite(left[i]).all():
                start = None
                continue
            if start is None or (i and t[i] - t[i - 1] > 1.5 / 60):
                start = i
            count = i - start + 1
            if count < 120 or (count - 120) % 30:
                continue
            a = i - 119
            fraction = float(truth[a:i + 1].mean())
            counts = np.bincount(activity[a:i + 1], minlength=8)
            top = np.flatnonzero(counts == counts.max())
            label = str(int(top[0])) if len(top) == 1 and counts.max() >= 60 else "mixed"
            rows.append((*map(int, key), t[a], t[i], int(fraction >= 0.5), fraction, label, bool(np.isfinite(right[a:i + 1]).all())))
    return pd.DataFrame(rows, columns=[*KEYS, "truth_fog_label", "truth_fog_fraction", "truth_activity", "right_complete"])


def check_frame(frame, expected):
    if len(frame) != len(expected):
        raise ValueError("Eligible window count differs")
    for column in [*KEYS, "truth_fog_label", "truth_fog_fraction"]:
        np.testing.assert_allclose(frame[column].astype(float), expected[column].astype(float), atol=1e-6, rtol=0)
    if not np.array_equal(frame.truth_activity.astype(str), expected.truth_activity.astype(str)):
        raise ValueError("Raw activity grouping differs")
    if not np.allclose(frame.available_at_s, frame.window_end_s, atol=1e-9, rtol=0):
        raise ValueError("Backdated prediction")
    if not np.array_equal(frame.is_fog, frame.fog_probability >= frame.fog_threshold):
        raise ValueError("Score/threshold decision mismatch")
    return len(frame)


def check_selected_pairs(frame, values, selected):
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1
    z = (values - values.mean(axis=0)) / scale
    for pair in selected:
        best = None
        for i, row in frame.iterrows():
            if row.truth_fog_fraction > 1e-12 or not row.is_fog or str(row.truth_activity) != str(pair["activity"]):
                continue
            mask = (frame.subject_id.eq(row.subject_id) & frame.truth_activity.astype(str).eq(str(pair["activity"]))
                    & frame.truth_fog_fraction.ge(1 - 1e-12))
            mask &= (~frame.session_id.eq(row.session_id) | ~frame.task_id.eq(row.task_id)
                     | (frame.window_start_s - row.window_start_s).abs().ge(6 - 1e-6))
            for j in np.flatnonzero(mask):
                distance = float(np.sqrt(np.mean((z[j, :33] - z[i, :33])**2)))
                candidate = distance, int(i), int(j)
                if best is None or candidate < best:
                    best = candidate
        if best is None or pair["negative_index"] != best[1] or pair["positive_index"] != best[2]:
            raise ValueError("Failure pair is not closest eligible same-person/activity pair")
        np.testing.assert_allclose(pair["left_distance"], best[0], atol=1e-8, rtol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    config, split, lock, protocol = [json.loads((run / p).read_text("utf-8")) for p in ("config.json", "split.json", "MODEL_LOCK.json", "protocol.json")]
    complete = json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8"))
    started = json.loads((run / "evaluation/VALIDATION_STARTED.json").read_text("utf-8"))
    check_lock(run, lock)
    check_lock(run / "evaluation", complete)
    if sha(args.raw) != protocol["raw_sha256"] or complete["model_lock_sha256"] != sha(run / "MODEL_LOCK.json"):
        raise ValueError("Raw or lock mismatch")
    if datetime.fromisoformat(lock["locked_utc"]) >= datetime.fromisoformat(started["started_utc"]):
        raise ValueError("Validation precedes model lock")
    if protocol["data_roles_constructed"] != ["train"] or protocol["validation_windows_before_lock"] or complete["test_windows_constructed"]:
        raise ValueError("Unexpected data role exposure")
    for name, digest in protocol["source_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Changed source: {name}")
    tree = ast.parse((root / "scripts/evaluate_a1_bilateral.py").read_text("utf-8"))
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"} for n in ast.walk(tree)):
        raise ValueError("Evaluation contains a fitting call")
    raw = pd.read_csv(args.raw)
    raws = {role: role_frame(raw, split, role) for role in ("train", "validation")}
    del raw
    inventories = {role: inventory(frame) for role, frame in raws.items()}
    train_expected = inventories["train"].loc[lambda f: f.right_complete].reset_index(drop=True)
    val_expected = inventories["validation"].loc[lambda f: f.right_complete].reset_index(drop=True)
    dual = anchored_gyro_windows(raws["train"], require_right=True)
    banks = matched_features(dual)
    x, names = a1_matrix(left_view(dual), "baseline_2s_v1")
    np.testing.assert_allclose(x, banks["left_rf"][0], atol=1e-8, rtol=1e-12)
    if names != banks["left_rf"][1]:
        raise ValueError("Left prefix is not the standalone left feature set")
    full_reference = pd.read_csv(run / "input_audit/left_reference_predictions.csv", float_precision="round_trip")
    joint_reference = pd.read_csv(run / "input_audit/joint_reference_predictions.csv", float_precision="round_trip")
    check_frame(full_reference, inventories["train"])
    check_frame(joint_reference, train_expected)
    np.testing.assert_array_equal(np.load(run / "input_audit/matched_left_indices.npy"), np.flatnonzero(inventories["train"].right_complete))
    supports = json.loads((run / "input_audit/support.json").read_text("utf-8"))
    for name, frame in [("left", full_reference), ("dual", joint_reference)]:
        for field, value in support_summary(raws["train"], frame).items():
            if not np.isclose(supports[name][field], value, atol=1e-6, rtol=0):
                raise ValueError("Input support audit differs")
    selected = json.loads((run / "input_audit/selected_pairs.json").read_text("utf-8"))
    check_selected_pairs(joint_reference, banks["bilateral_rf"][0], selected)
    plans = json.loads((run / "fold_plan.json").read_text("utf-8"))
    groups, labels = train_expected.subject_id.to_numpy(int), train_expected.truth_fog_label.to_numpy(int)
    actual = tuple(sorted(map(int, set(groups))))
    frames, all_frames, checked = {}, {}, 0
    for model_id, frame_name in [("left_rf", "left_matched"), ("bilateral_rf", "bilateral_rf")]:
        oof = pd.read_csv(run / model_id / "oof_predictions.csv", float_precision="round_trip")
        val = pd.read_csv(run / "evaluation" / frame_name / "predictions.csv", float_precision="round_trip")
        checked += check_frame(oof, train_expected) + check_frame(val, val_expected)
        values = banks[model_id][0]
        seen = np.zeros(len(groups), int)
        for plan in plans:
            fit = np.flatnonzero(np.isin(groups, plan["fit_people"]))
            held = np.flatnonzero(np.isin(groups, plan["held_out_people"]))
            if set(plan["fit_people"]) & set(plan["held_out_people"]) or set(plan["fit_people"]) | set(plan["held_out_people"]) != set(split["train"]):
                raise ValueError("Outer participant leakage")
            if set(labels[fit]) != {0, 1}:
                raise ValueError("Outer fit missing a class")
            classifier = joblib.load(run / model_id / f"fold_{plan['fold']}.joblib")
            if len(held):
                np.testing.assert_allclose(classifier.predict_proba(values[held])[:, 1], oof.iloc[held].fog_probability, atol=1e-7, rtol=0)
                seen[held] += 1
        if not np.all(seen == 1):
            raise ValueError("Incomplete TRAIN OOF")
        model = A1FogModel.load(run / model_id / "model.joblib")
        if model.fit_participants != actual or model.threshold_participants != actual:
            raise ValueError("Actual fit/calibration participants differ")
        choice = choose_threshold(labels, oof.fog_probability.to_numpy(), config)["selected"]
        np.testing.assert_allclose(model.threshold, choice["threshold"], atol=1e-12, rtol=0)
        val_dual = anchored_gyro_windows(raws["validation"], require_right=True)
        val_windows = left_view(val_dual) if model_id == "left_rf" else val_dual
        np.testing.assert_allclose(model.predict_windows(val_windows).fog_probability, val.fog_probability, atol=1e-7, rtol=0)
        sensors = raws["validation"].iloc[:180].drop(columns=["fog", "activity", "fog_severity"])
        pd.testing.assert_frame_equal(model.predict_frame(sensors), model.predict_frame(sensors.assign(fog=1, activity=3)))
        runtime = pd.read_csv(run / "evaluation" / frame_name / "runtime.csv")
        if any(c.startswith("truth_") for c in runtime):
            raise ValueError("Truth exposed in runtime")
        frames[frame_name], all_frames[frame_name, "validation_own"], all_frames[model_id, "train_oof_calibration"] = val, val, oof
    for name in ("left_natural", "previous_author", "reference_rf_context", "reference_rf_ankle"):
        frame = pd.read_csv(run / "evaluation" / name / "predictions.csv", float_precision="round_trip")
        if name == "left_natural":
            checked += check_frame(frame, inventories["validation"])
        frames[name], all_frames[name, "validation_own"] = frame, frame
    all_frames.update({(name, "all_common"): frame for name, frame in matched_frames(frames).items()})
    table = pd.read_csv(run / "evaluation/comparison.csv")
    for row in table.itertuples():
        frame = all_frames[row.model, row.pool]
        for field, value in metrics(frame).items():
            if value is not None and not np.isclose(getattr(row, field), value, atol=1e-10, rtol=0):
                raise ValueError(f"Metric mismatch: {row.model}/{row.pool}/{field}")
        if row.pool != "all_common":
            role = "train" if row.pool == "train_oof_calibration" else "validation"
            support = EventSupport.build(raws[role], frame, allowed=split[role])
            for field, value in support.metrics(frame.is_fog).items():
                if not np.isclose(getattr(row, field), value, atol=1e-6, rtol=0):
                    raise ValueError(f"Event/time mismatch: {field}")
    own = table.loc[table.pool.eq("validation_own")].set_index("model")
    a, b, natural = own.loc["left_matched"], own.loc["bilateral_rf"], own.loc["left_natural"]
    conditions = {"matched_f1_higher": bool(b.fog_f1 > a.fog_f1),
        "false_alarm_seconds_not_higher": bool(b.false_alarm_s <= a.false_alarm_s + 1e-6),
        "fog_time_recall_not_lower": bool(b.time_recall + 1e-9 >= a.time_recall),
        "natural_left_support_not_lost": bool(b.decision_support_fraction + 1e-9 >= natural.decision_support_fraction)}
    gate = json.loads((run / "evaluation/development_gate.json").read_text("utf-8"))
    if gate["conditions"] != conditions or gate["passed"] != all(conditions.values()):
        raise ValueError("Gate differs")
    result = {"status": "PASS", "verified_new_model_window_rows": checked, "fold_models_rescored": 6,
        "independent_retraining": False, "left_anchor_right_finiteness_raw_truth": "PASS",
        "feature_prefix_and_failure_pairs": "PASS", "train_only_oof_threshold_actual_people": "PASS",
        "saved_models_metrics_and_time_support": "PASS", "model_lock_before_validation": "PASS",
        "test_windows_constructed": 0, "model_files": len(lock["files"]), "evaluation_files": len(complete["files"]),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json")}
    write_json(run / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
