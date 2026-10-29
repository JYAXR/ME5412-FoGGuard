"""Audit causal history, nested roles/checkpoints, locked models and saved scores."""

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
from verify_a1_author_cnn import verify_frame

from fogguard.evaluation.a1_author import matched_frames, metrics
from fogguard.evaluation.a1_event import EventSupport
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.modeling.a1_author_cnn import GYRO_CHANNELS, gyro_windows
from fogguard.modeling.a1_fog import A1FogModel
from fogguard.modeling.a1_temporal_cnn import sequence_batch


def independent_links(frame):
    expected, history, previous = [], [], None
    for i, row in enumerate(frame.itertuples()):
        key = (row.subject_id, row.session_id, row.task_id)
        if previous is None or key != previous[0] or abs(row.window_end_s - previous[1] - 0.5) > 1e-6:
            history = []
        history.append(i)
        history = history[-9:]
        chosen = [history[-n] for n in (9, 5, 1) if len(history) >= n]
        expected.append(chosen + [-1] * (3 - len(chosen)))
        previous = key, row.window_end_s
    return np.asarray(expected, int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    lock = json.loads((run / "MODEL_LOCK.json").read_text("utf-8"))
    protocol = json.loads((run / "protocol.json").read_text("utf-8"))
    complete = json.loads((run / "evaluation/EVALUATION_COMPLETED.json").read_text("utf-8"))
    started = json.loads((run / "evaluation/VALIDATION_STARTED.json").read_text("utf-8"))
    check_lock(run, lock)
    check_lock(run / "evaluation", complete)
    if sha(args.raw) != protocol["raw_sha256"] or complete["model_lock_sha256"] != sha(run / "MODEL_LOCK.json"):
        raise ValueError("Raw/model lock mismatch")
    if datetime.fromisoformat(lock["locked_utc"]) >= datetime.fromisoformat(started["started_utc"]):
        raise ValueError("Validation started before model lock")
    if protocol["data_roles_constructed"] != ["train"] or protocol["validation_windows_before_lock"] or complete["test_windows_constructed"]:
        raise ValueError("Unexpected data role before locking or TEST usage")
    for path, digest in protocol["source_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Evaluated source differs: {path}")
    tree = ast.parse((root / "scripts/evaluate_a1_temporal_cnn.py").read_text("utf-8"))
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in
           {"fit", "fit_transform", "partial_fit"} for n in ast.walk(tree)):
        raise ValueError("Evaluation contains a fitting call")
    split = json.loads((run / "split.json").read_text("utf-8"))
    config = json.loads((run / "config.json").read_text("utf-8"))
    plans = json.loads((run / "fold_plan.json").read_text("utf-8"))
    raw = pd.read_csv(args.raw)
    role_raw = {role: role_frame(raw, split, role) for role in ("train", "validation")}
    del raw
    train_windows, val_windows = gyro_windows(role_raw["train"]), gyro_windows(role_raw["validation"])
    x, lengths, links = sequence_batch(train_windows)
    val_x, val_lengths, val_links = sequence_batch(val_windows)
    groups = np.array([int(w.subject_id) for w in train_windows])
    labels = np.array([w.fog_label for w in train_windows])
    frozen = np.load(run / "training_history_links.npz")
    np.testing.assert_array_equal(links, frozen["links"])
    checked, validation_frames, all_frames = 0, {}, {}
    for name in config["models"]:
        oof = pd.read_csv(run / name / "train_oof/predictions.csv")
        validation = pd.read_csv(run / "evaluation" / name / "predictions.csv")
        checked += verify_frame(role_raw["train"], oof, list(GYRO_CHANNELS), split["train"])
        checked += verify_frame(role_raw["validation"], validation, list(GYRO_CHANNELS), split["validation"])
        np.testing.assert_array_equal(independent_links(oof), links)
        np.testing.assert_array_equal(independent_links(validation), val_links)
        np.testing.assert_array_equal(validation.context_windows, val_lengths if name == "cnn_gru" else np.ones(len(validation)))
        held_seen, best_epochs = [], []
        for plan in plans:
            fold = plan["fold"]
            inner, checkpoint = set(plan["inner_fit_people"]), set(plan["checkpoint_people"])
            fit, held = set(plan["fit_people"]), set(plan["held_out_people"])
            if inner & checkpoint or inner | checkpoint != fit or fit & held or fit | held != set(split["train"]):
                raise ValueError("Nested participant leakage")
            held_seen.extend(held)
            index = np.flatnonzero(np.isin(groups, list(held)))
            if not np.all(oof.iloc[index].held_out_fold == fold):
                raise ValueError("OOF fold mismatch")
            directory = run / name / f"fold_{fold}"
            audit = json.loads((directory / "fit_audit.json").read_text("utf-8"))
            for key in plan:
                if audit[key] != plan[key]:
                    raise ValueError("Fold audit changed")
            history = pd.read_csv(directory / "checkpoint_history.csv")
            best_loss, best_epoch = float("inf"), None
            for row in history.itertuples():
                if row.checkpoint_loss < best_loss - config["optimization"]["min_delta"]:
                    best_loss, best_epoch = row.checkpoint_loss, row.epoch
            if best_epoch != audit["best_epoch"] or audit["checkpoint_weights_used_for_oof"]:
                raise ValueError("Wrong TRAIN checkpoint or reused weights")
            classifier = joblib.load(directory / "oof_model.joblib")
            if len(classifier.history_) != best_epoch or classifier.best_epoch_ != best_epoch:
                raise ValueError("OOF refit epochs differ from inner selection")
            np.testing.assert_allclose(classifier.predict_proba(x[index], lengths[index])[:, 1], oof.iloc[index].fog_probability, atol=1e-7, rtol=0)
            best_epochs.append(best_epoch)
        if sorted(held_seen) != sorted(split["train"]):
            raise ValueError("TRAIN not held out exactly once")
        selection = choose_threshold(labels, oof.fog_probability.to_numpy(), config)["selected"]
        model = A1FogModel.load(run / name / "model.joblib")
        metadata = json.loads((run / name / "fit_metadata.json").read_text("utf-8"))
        if model.fit_participants != tuple(split["train"]) or model.threshold_participants != tuple(split["train"]):
            raise ValueError("Deployment/calibration uses wrong people")
        if metadata["epochs"] != int(np.median(best_epochs)) or metadata["validation_people_used"] or metadata["test_people_used"]:
            raise ValueError("Deployment epoch or role mismatch")
        if model.threshold != selection["threshold"]:
            raise ValueError("Threshold not derived from TRAIN OOF")
        np.testing.assert_allclose(model.classifier.predict_proba(val_x, val_lengths)[:, 1], validation.fog_probability, atol=1e-7, rtol=0)
        sample = val_windows[:20]
        batch = model.predict_windows(sample)
        stream = model.new_stream()
        streamed = pd.concat([stream.push(w) for w in sample], ignore_index=True)
        np.testing.assert_allclose(streamed.fog_probability, batch.fog_probability, atol=1e-6, rtol=0)
        sensors = role_raw["validation"].iloc[:180].drop(columns=["fog", "activity", "fog_severity"])
        pd.testing.assert_frame_equal(model.predict_frame(sensors), model.predict_frame(sensors.assign(fog=1, activity=7)))
        for directory in (run / name / "train_oof", run / "evaluation" / name):
            runtime = pd.read_csv(directory / "runtime.csv")
            if any(c.startswith("truth_") for c in runtime):
                raise ValueError("Truth exposed in runtime")
        validation_frames[name] = validation
        all_frames[name, "validation_own"] = validation
        all_frames[name, "train_oof_calibration"] = oof
    for name in ("reference_author_cnn", "reference_rf_context", "reference_rf_ankle"):
        frame = pd.read_csv(run / "evaluation" / name / "predictions.csv")
        validation_frames[name] = frame
        all_frames[name, "validation_own"] = frame
    all_frames.update({(name, "all_common"): frame for name, frame in matched_frames(validation_frames).items()})
    comparison = pd.read_csv(run / "evaluation/comparison.csv")
    for row in comparison.itertuples():
        frame = all_frames[row.model, row.pool]
        for field, value in metrics(frame).items():
            if value is not None and not np.isclose(getattr(row, field), value, atol=1e-10, rtol=0):
                raise ValueError(f"Metric mismatch: {row.model}/{row.pool}/{field}")
        if row.pool != "all_common":
            role = "train" if row.pool == "train_oof_calibration" else "validation"
            support = EventSupport.build(role_raw[role], frame, allowed=split[role])
            for field, value in support.metrics(frame.is_fog).items():
                if not np.isclose(getattr(row, field), value, atol=1e-6, rtol=0):
                    raise ValueError(f"Event/support mismatch: {field}")
    own = comparison.loc[comparison.pool.eq("validation_own")].set_index("model")
    a, b = own.loc["cnn_control"], own.loc["cnn_gru"]
    conditions = {"f1_higher": bool(b.fog_f1 > a.fog_f1),
                  "false_alarm_seconds_not_higher": bool(b.false_alarm_s <= a.false_alarm_s + 1e-6),
                  "fog_time_recall_not_lower": bool(b.time_recall + 1e-9 >= a.time_recall)}
    gate = json.loads((run / "evaluation/development_gate.json").read_text("utf-8"))
    if conditions != gate["conditions"] or all(conditions.values()) != gate["passed"]:
        raise ValueError("Development gate differs")
    result = {"status": "PASS", "verified_new_model_window_rows": checked, "history_rows": len(links) + len(val_links),
        "nested_oof_models_rescored": 6, "independent_retraining": False,
        "train_only_checkpoint_epoch_threshold": "PASS", "raw_window_and_causal_history": "PASS",
        "saved_scores_metrics_and_support": "PASS", "batch_stream_and_truth_independence": "PASS",
        "model_lock_before_validation": "PASS", "test_windows_constructed": 0,
        "model_files": len(lock["files"]), "evaluation_files": len(complete["files"]),
        "model_lock_sha256": sha(run / "MODEL_LOCK.json")}
    write_json(run / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
