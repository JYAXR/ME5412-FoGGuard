"""Independently audit objective isolation, raw targets, nested fits and scores."""

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json
from verify_a1_author_cnn import verify_frame
from verify_a1_temporal_cnn import independent_links

from fogguard.evaluation.a1_author import matched_frames, metrics
from fogguard.evaluation.a1_event import EventSupport
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.modeling.a1_author_cnn import GYRO_CHANNELS, gyro_windows
from fogguard.modeling.a1_fog import A1FogModel
from fogguard.modeling.a1_temporal_cnn import sequence_batch


def raw_activity_targets(raw, frame):
    records = {tuple(map(int, key)): (g.timestamp.to_numpy(float), g.activity.to_numpy(int))
               for key, g in raw.groupby(["subjectID", "sessionID", "taskID"], sort=False)}
    result = []
    for row in frame.itertuples():
        times, activity = records[int(row.subject_id), int(row.session_id), int(row.task_id)]
        i = int(np.argmin(np.abs(times - row.window_start_s)))
        counts = np.bincount(activity[i:i + 120], minlength=8)
        top = np.flatnonzero(counts == counts.max())
        result.append(int(top[0] - 1) if len(top) == 1 and 1 <= top[0] <= 7 and counts.max() >= 60 else -100)
    return np.asarray(result, int)


def check_weights(classifier, groups, targets, balanced):
    groups = np.asarray(groups, int)
    people = sorted(set(groups))
    expected = np.array([len(groups) / (len(people) * int((groups == p).sum())) if balanced else 1.0 for p in groups])
    np.testing.assert_allclose(classifier.sample_weights_, expected, atol=1e-6, rtol=1e-7)
    if classifier.fit_people_ != tuple(map(int, people)):
        raise ValueError("Weight fit uses wrong people")
    counts = {str(v): int((targets == v).sum()) for v in (-100, *range(7))}
    if counts != classifier.activity_target_counts_:
        raise ValueError("Fit activity targets include wrong rows")


def checkpoint_loss(classifier, x, lengths, labels):
    classifier.network_.eval()
    total, batch = 0.0, classifier.config["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(labels), batch):
            stop = start + batch
            logits = classifier.network_(torch.from_numpy(x[start:stop]), torch.as_tensor(lengths[start:stop]))
            total += float(torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(labels[start:stop], dtype=torch.float32))) * len(labels[start:stop])
        return total / len(labels) + classifier.config["kernel_l2"] * float(classifier.network_.kernel_penalty())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    root, run = Path(__file__).resolve().parents[1], args.run_dir
    config, split, protocol, lock = [json.loads((run / p).read_text("utf-8")) for p in
                                    ("config.json", "split.json", "protocol.json", "MODEL_LOCK.json")]
    complete = json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8"))
    started = json.loads((run / "evaluation/VALIDATION_STARTED.json").read_text("utf-8"))
    check_lock(run, lock)
    check_lock(run / "evaluation", complete)
    if sha(args.raw) != protocol["raw_sha256"] or complete["model_lock_sha256"] != sha(run / "MODEL_LOCK.json"):
        raise ValueError("Input or model lock changed")
    if datetime.fromisoformat(lock["locked_utc"]) >= datetime.fromisoformat(started["started_utc"]):
        raise ValueError("Validation before model lock")
    if protocol["data_roles_constructed"] != ["train"] or protocol["validation_windows_before_lock"] or complete["test_windows_constructed"]:
        raise ValueError("Unexpected role exposure")
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Changed source: {path}")
    tree = ast.parse((root / "scripts/evaluate_a1_context_learning.py").read_text("utf-8"))
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"} for n in ast.walk(tree)):
        raise ValueError("Fitting call inside evaluation")
    torch.set_num_threads(config["optimization"]["cpu_threads"])
    raw = pd.read_csv(args.raw)
    raws = {role: role_frame(raw, split, role) for role in ("train", "validation")}
    del raw
    train, validation = gyro_windows(raws["train"]), gyro_windows(raws["validation"])
    x, lengths, links = sequence_batch(train)
    vx, vl, vlinks = sequence_batch(validation)
    groups, y = np.array([int(w.subject_id) for w in train]), np.array([w.fog_label for w in train])
    plans = json.loads((run / "fold_plan.json").read_text("utf-8"))
    metadata = np.load(run / "training_metadata.npz")
    np.testing.assert_array_equal(links, metadata["links"])
    all_frames, val_frames, checked = {}, {}, 0
    for name in config["models"]:
        oof = pd.read_csv(run / name / "train_oof/predictions.csv")
        val = pd.read_csv(run / "evaluation" / name / "predictions.csv")
        checked += verify_frame(raws["train"], oof, list(GYRO_CHANNELS), split["train"])
        checked += verify_frame(raws["validation"], val, list(GYRO_CHANNELS), split["validation"])
        np.testing.assert_array_equal(links, independent_links(oof))
        np.testing.assert_array_equal(vlinks, independent_links(val))
        targets = raw_activity_targets(raws["train"], oof)
        np.testing.assert_array_equal(targets, metadata["activities"])
        epochs, held_seen = [], []
        for plan in plans:
            inner, check, fit, held = [set(plan[p]) for p in ("inner_fit_people", "checkpoint_people", "fit_people", "held_out_people")]
            if inner & check or inner | check != fit or fit & held or fit | held != set(split["train"]):
                raise ValueError("Nested role leakage")
            index = {k: np.flatnonzero(np.isin(groups, list(v))) for k, v in [("inner", inner), ("check", check), ("fit", fit), ("held", held)]}
            held_seen.extend(held)
            dest = run / name / f"fold_{plan['fold']}"
            audit = json.loads((dest / "fit_audit.json").read_text("utf-8"))
            if audit["objective"] != config["arms"][name] or audit["checkpoint_activity_targets_used"] or audit["checkpoint_weights_used_for_oof"]:
                raise ValueError("Objective/checkpoint audit differs")
            history = pd.read_csv(dest / "checkpoint_history.csv")
            best_loss, epoch = float("inf"), None
            for row in history.itertuples():
                if row.checkpoint_loss < best_loss - config["optimization"]["min_delta"]:
                    best_loss, epoch = row.checkpoint_loss, row.epoch
            tuned = joblib.load(dest / "checkpoint_model.joblib")
            check_weights(tuned, groups[index["inner"]], targets[index["inner"]], config["arms"][name]["person_balance"])
            np.testing.assert_allclose(checkpoint_loss(tuned, x[index["check"]], lengths[index["check"]], y[index["check"]]), best_loss, atol=1e-6, rtol=0)
            fitted = joblib.load(dest / "oof_model.joblib")
            if audit["best_epoch"] != epoch or len(fitted.history_) != epoch:
                raise ValueError("OOF epoch does not match TRAIN checkpoint")
            check_weights(fitted, groups[index["fit"]], targets[index["fit"]], config["arms"][name]["person_balance"])
            np.testing.assert_allclose(fitted.predict_proba(x[index["held"]], lengths[index["held"]])[:, 1],
                                       oof.iloc[index["held"]].fog_probability, atol=1e-7, rtol=0)
            if not np.all(oof.iloc[index["held"]].held_out_fold.eq(plan["fold"])):
                raise ValueError("Saved held-out fold differs")
            epochs.append(epoch)
        if sorted(held_seen) != sorted(split["train"]):
            raise ValueError("TRAIN people not held out once")
        detector = A1FogModel.load(run / name / "model.joblib")
        model = detector.classifier
        if detector.fit_participants != tuple(split["train"]) or detector.threshold_participants != tuple(split["train"]):
            raise ValueError("Final model/calibration role differs")
        if model.objective != config["arms"][name] or model.best_epoch_ != int(np.median(epochs)) or model.parameter_count_ != 14536:
            raise ValueError("Final objective, architecture or epoch differs")
        check_weights(model, groups, targets, config["arms"][name]["person_balance"])
        selected = choose_threshold(y, oof.fog_probability.to_numpy(), config)["selected"]
        if detector.threshold != selected["threshold"]:
            raise ValueError("Threshold not TRAIN-OOF selected")
        np.testing.assert_allclose(model.predict_proba(vx, vl)[:, 1], val.fog_probability, atol=1e-7, rtol=0)
        stream = detector.new_stream()
        streamed = pd.concat([stream.push(w) for w in validation[:20]], ignore_index=True)
        np.testing.assert_allclose(streamed.fog_probability, detector.predict_windows(validation[:20]).fog_probability, atol=1e-6, rtol=0)
        sensors = raws["validation"].iloc[:180].drop(columns=["fog", "activity", "fog_severity"])
        pd.testing.assert_frame_equal(detector.predict_frame(sensors), detector.predict_frame(sensors.assign(fog=1, activity=3)))
        for dest in (run / name / "train_oof", run / "evaluation" / name):
            if any(c.startswith("truth_") or c in {"activity", "fog"} for c in pd.read_csv(dest / "runtime.csv")):
                raise ValueError("Truth present in runtime")
        val_frames[name], all_frames[name, "validation_own"], all_frames[name, "train_oof_calibration"] = val, val, oof
    for name in ("previous_gru", "previous_author", "reference_rf_context", "reference_rf_ankle"):
        frame = pd.read_csv(run / "evaluation" / name / "predictions.csv")
        val_frames[name], all_frames[name, "validation_own"] = frame, frame
    all_frames.update({(name, "all_common"): frame for name, frame in matched_frames(val_frames).items()})
    comparison = pd.read_csv(run / "evaluation/comparison.csv")
    for row in comparison.itertuples():
        frame = all_frames[row.model, row.pool]
        for field, value in metrics(frame).items():
            if value is not None and not np.isclose(getattr(row, field), value, atol=1e-10, rtol=0):
                raise ValueError(f"Metric differs: {row.model}/{row.pool}/{field}")
        if row.pool != "all_common":
            role = "train" if row.pool == "train_oof_calibration" else "validation"
            support = EventSupport.build(raws[role], frame, allowed=split[role])
            for field, value in support.metrics(frame.is_fog).items():
                if not np.isclose(getattr(row, field), value, atol=1e-6, rtol=0):
                    raise ValueError(f"Event/support differs: {field}")
    gates = json.loads((run / "evaluation/development_gates.json").read_text("utf-8"))
    own = comparison.loc[comparison.pool.eq("validation_own")].set_index("model")
    base = own.loc["control"]
    for name in ("person_balanced", "activity_aux"):
        row = own.loc[name]
        conditions = {"f1_higher": bool(row.fog_f1 > base.fog_f1), "false_alarm_seconds_not_higher": bool(row.false_alarm_s <= base.false_alarm_s + 1e-6),
                      "fog_time_recall_not_lower": bool(row.time_recall + 1e-9 >= base.time_recall)}
        if gates["candidates"][name] != {"conditions": conditions, "passed": all(conditions.values())}:
            raise ValueError("Development gate differs")
    result = {"status": "PASS", "verified_new_model_window_rows": checked, "oof_models_rescored": 9,
        "checkpoint_models_rescored": 9, "independent_retraining": False, "raw_activity_targets_and_fit_only_weights": "PASS",
        "train_only_nested_epoch_and_threshold": "PASS", "raw_window_history_and_scores": "PASS",
        "runtime_truth_independence_and_stream": "PASS", "model_lock_before_validation": "PASS",
        "test_windows_constructed": 0, "model_files": len(lock["files"]), "evaluation_files": len(complete["files"]),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json")}
    write_json(run / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
