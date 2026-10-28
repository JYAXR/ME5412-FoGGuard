"""TRAIN-only grouped RF diagnostics and an author-architecture CNN reproduction.

This development run never constructs or scores TEST windows.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json
from sklearn.model_selection import GroupKFold

from fogguard.data.fogstar import iter_fogstar_windows
from fogguard.evaluation.a1_author import annotate
from fogguard.evaluation.a1_fixed_split import (
    check_split,
    choose_threshold,
    classifier_for,
    role_frame,
)
from fogguard.modeling.a1_author_cnn import (
    AuthorCNNClassifier,
    GyroFoG,
    centered_gyro,
    gyro_windows,
)
from fogguard.modeling.a1_context import context_matrix
from fogguard.modeling.a1_fixed_split import prediction_frame
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix


def save_scores(directory, windows, score, cutoff, name, role):
    frame = annotate(windows, prediction_frame(windows, score, cutoff, name), role)
    frame["prediction_source"] = "participant_held_out_train_oof" if role == "train_oof" else "train14_validation"
    frame.to_csv(directory / "predictions.csv", index=False)
    frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(directory / "runtime.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_author_cnn_v1.json"))
    parser.add_argument("--reference", type=Path, default=Path("reports/a1_fixed_split_v1_20261029"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    output, root = args.run_dir, Path(__file__).resolve().parents[1]
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh run directory")
    config = json.loads(args.config.read_text("utf-8"))
    split_config = json.loads(Path(config["split_config"]).read_text("utf-8"))
    reference_lock = json.loads((args.reference / "TRAINING_LOCK.json").read_text("utf-8"))
    check_lock(args.reference, reference_lock)
    reference_protocol = json.loads((args.reference / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != reference_protocol["raw_sha256"]:
        raise ValueError("Raw data differs from fixed split")
    if split_config["split"] != json.loads((args.reference / "frozen_config.json").read_text("utf-8"))["split"]:
        raise ValueError("Reference participant split differs")
    output.mkdir(parents=True)
    raw = pd.read_csv(args.raw)
    check_split(split_config["split"], raw.subjectID.unique())
    frames = {role: role_frame(raw, split_config["split"], role) for role in ("train", "validation")}
    del raw
    write_json(output / "config.json", config)
    write_json(output / "split_config.json", split_config)
    write_json(output / "protocol.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "raw_sha256": sha(args.raw),
        "config_sha256": sha(args.config), "split_sha256": sha(config["split_config"]),
        "reference_training_lock_sha256": sha(args.reference / "TRAINING_LOCK.json"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
            root / "scripts/run_a1_author_cnn.py", root / "scripts/report_a1_author_cnn.py",
            root / "scripts/verify_a1_author_cnn.py", *sorted((root / "src/fogguard").rglob("*.py"))]},
        "data_roles_constructed": ["train", "validation"], "test_windows_constructed": 0,
        "reference_sources": {name: sha(args.reference / name / "validation_predictions.csv")
                              for name in ("rf_context", "rf_ankle")},
        "records": {role: frame.groupby(["subjectID", "sessionID", "taskID"]).ngroups
                    for role, frame in frames.items()}})
    train, validation = gyro_windows(frames["train"]), gyro_windows(frames["validation"])
    x, vx = centered_gyro(train), centered_gyro(validation)
    y, vy = np.array([w.fog_label for w in train]), np.array([w.fog_label for w in validation])
    print(f"TRAIN gyro windows={len(train)}; VALIDATION={len(validation)}; no TEST windows", flush=True)
    for name in config["models"]:
        directory = output / name
        directory.mkdir()
        print(f"Fresh TRAIN-only fit: {name}", flush=True)
        if name == "author_cnn":
            classifier = AuthorCNNClassifier(config["cnn"], config["random_state"]).fit(x, y, (vx, vy))
            names = ()
            score = classifier.predict_proba(vx)[:, 1]
            pd.DataFrame(classifier.history_).to_csv(directory / "training_history.csv", index=False)
        else:
            values, names = a1_matrix(train, "baseline_2s_v1")
            val_values, val_names = a1_matrix(validation, "baseline_2s_v1")
            if names != val_names:
                raise ValueError("Gyro RF feature schema differs")
            classifier = classifier_for("rf_2s", split_config).fit(values, y)
            score = classifier.predict_proba(val_values)[:, 1]
        choice = choose_threshold(vy, score, config)
        model = GyroFoG(classifier, name, train[0].channel_names, names, choice["selected"]["threshold"],
                        tuple(split_config["split"]["train"]), tuple(split_config["split"]["validation"]))
        model.save(directory / "model.joblib")
        restored = A1FogModel.load(directory / "model.joblib")
        pd.testing.assert_frame_equal(model.predict_windows(validation), restored.predict_windows(validation))
        runtime_frame = frames["validation"].drop(columns=["fog", "activity", "fog_severity"])
        pd.testing.assert_frame_equal(restored.predict_frame(frames["validation"]), restored.predict_frame(runtime_frame))
        save_scores(directory, validation, score, model.threshold, name, "validation")
        write_json(directory / "threshold_selection.json", choice)
        write_json(directory / "fit_metadata.json", {
            "fit_people": list(model.fit_participants), "checkpoint_and_threshold_people": list(model.threshold_participants),
            "fit_windows": len(train), "validation_windows": len(validation), "test_people_used": [],
            "parameters": getattr(classifier, "parameter_count_", None),
            "best_epoch": getattr(classifier, "best_epoch_", None), "trained_from_random_weights": name == "author_cnn",
            "normalization": "per-window gyro demean; no population scaler" if name == "author_cnn" else "existing filtered 33-feature extractor",
            "refit_after_validation": False, "serialization_and_truth_independence_checked": True})
    for name in ("rf_context", "rf_ankle"):
        directory = output / f"reference_{name}"
        directory.mkdir()
        frame = pd.read_csv(args.reference / name / "validation_predictions.csv")
        positions = ("ankleL",) if name == "rf_ankle" else ("ankleL", "back")
        windows = list(iter_fogstar_windows(frames["validation"], sensor_positions=positions, max_interpolation_gap_s=0))
        expected = prediction_frame(windows, frame.fog_probability.to_numpy(), float(frame.fog_threshold.iloc[0]), name)
        for column in ("subject_id", "session_id", "task_id", "window_start_s", "window_end_s"):
            if not np.allclose(expected[column].astype(float), frame[column].astype(float), atol=1e-6, rtol=0):
                raise ValueError("Saved reference validation windows differ")
        save_scores(directory, windows, frame.fog_probability.to_numpy(), float(frame.fog_threshold.iloc[0]), name, "validation")
    print("Building TRAIN-only three-fold participant-held-out RF_context diagnostics", flush=True)
    windows = list(iter_fogstar_windows(frames["train"], max_interpolation_gap_s=0))
    values = context_matrix(windows)[0]
    labels = np.array([w.fog_label for w in windows])
    groups = np.array([int(w.subject_id) for w in windows])
    scores, seen, audit = np.full(len(windows), np.nan), np.zeros(len(windows), int), []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=config["diagnostic_folds"]).split(values, labels, groups)):
        print(f"Diagnostic RF fold {fold + 1}/{config['diagnostic_folds']}", flush=True)
        if set(groups[fit]) & set(groups[held]) or set(labels[fit]) != {0, 1}:
            raise ValueError("OOF leakage or one-class fit")
        classifier = classifier_for("rf_context", split_config).fit(values[fit], labels[fit])
        scores[held] = classifier.predict_proba(values[held])[:, 1]
        seen[held] += 1
        audit.append({"fold": fold, "fit_people": sorted(map(int, set(groups[fit]))),
                      "held_out_people": sorted(map(int, set(groups[held]))),
                      "fit_windows": len(fit), "held_out_windows": len(held)})
    if not np.all(seen == 1) or not np.isfinite(scores).all():
        raise ValueError("Incomplete TRAIN OOF")
    directory = output / "train_oof_rf_context"
    directory.mkdir()
    frame = save_scores(directory, windows, scores, config["diagnostic_threshold"], "rf_context", "train_oof")
    membership = {p: fold["fold"] for fold in audit for p in fold["held_out_people"]}
    frame["held_out_fold"] = frame.subject_id.astype(int).map(membership)
    frame.to_csv(directory / "predictions.csv", index=False)
    write_json(directory / "folds.json", audit)
    (output / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    write_json(output / "MODEL_LOCK.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "test_windows_constructed": 0, "test_driven_choices": 0, "default_replaced": False,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print("Models and TRAIN OOF locked. VALIDATION is development, not independent TEST.", flush=True)


if __name__ == "__main__":
    main()
