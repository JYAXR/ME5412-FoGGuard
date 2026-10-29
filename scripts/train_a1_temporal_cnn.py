"""Nested TRAIN-only CNN/GRU fits; lock epochs/models/thresholds before VALIDATION."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evaluate_a1_fixed_split import check_lock
from run_a1_experiment import sha, write_json

from fogguard.evaluation.a1_author import annotate
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.evaluation.a1_temporal import assert_history_isolation, nested_plan, person_indices
from fogguard.modeling.a1_author_cnn import GYRO_CHANNELS, gyro_windows
from fogguard.modeling.a1_fixed_split import prediction_frame
from fogguard.modeling.a1_fog import A1FogModel
from fogguard.modeling.a1_temporal_cnn import TemporalClassifier, TemporalFoG, sequence_batch


def save_predictions(path, windows, scores, cutoff, name, lengths, role):
    frame = annotate(windows, prediction_frame(windows, scores, cutoff, name), role)
    frame["model_version"] = f"a1_temporal_cnn_v1/{name}"
    frame["prediction_source"] = "participant_held_out_train_oof" if role == "train_oof" else "train14_frozen_validation"
    frame["context_windows"] = lengths if name == "cnn_gru" else 1
    frame.to_csv(path / "predictions.csv", index=False)
    frame.drop(columns=[c for c in frame if c.startswith("truth_")]).to_csv(path / "runtime.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/a1_temporal_cnn_v1.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path(__file__).resolve().parents[1], args.run_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh training directory")
    config = json.loads(args.config.read_text("utf-8"))
    if config["models"] != ["cnn_control", "cnn_gru"] or config["history_lag_steps"] != [8, 4, 0]:
        raise ValueError("Unexpected declared comparison")
    split = json.loads(Path(config["split_config"]).read_text("utf-8"))["split"]
    reference = Path(config["reference_run"])
    old_lock = json.loads((reference / "MODEL_LOCK.json").read_text("utf-8"))
    check_lock(reference, old_lock)
    old_protocol = json.loads((reference / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != old_protocol["raw_sha256"]:
        raise ValueError("Raw data differs from previous fixed cohort")
    if split != json.loads((reference / "split_config.json").read_text("utf-8"))["split"]:
        raise ValueError("Participant roles changed")
    raw = pd.read_csv(args.raw)
    train_raw = role_frame(raw, split, "train")
    del raw
    windows = gyro_windows(train_raw)
    del train_raw
    x, lengths, links = sequence_batch(windows)
    labels, groups = np.array([w.fog_label for w in windows]), np.array([int(w.subject_id) for w in windows])
    if set(groups) != set(split["train"]):
        raise ValueError("Some TRAIN people lack gyro windows")
    folds = json.loads((reference / "train_oof_rf_context/folds.json").read_text("utf-8"))
    plans = nested_plan(groups, labels, folds, checkpoint_people=config["checkpoint_people_per_fold"], seed=config["checkpoint_seed"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "split.json", split)
    write_json(output / "fold_plan.json", plans)
    np.savez_compressed(output / "training_history_links.npz", links=links, lengths=lengths, people=groups)
    paths = [root / f"scripts/{name}.py" for name in ("train_a1_temporal_cnn", "evaluate_a1_temporal_cnn",
             "verify_a1_temporal_cnn", "report_a1_author_cnn", "run_a1_event", "verify_a1_author_cnn")]
    paths += [root / "recognition/diagnostics.py", *sorted((root / "src/fogguard").rglob("*.py"))]
    write_json(output / "protocol.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": sha(args.raw), "config_sha256": sha(args.config), "split_sha256": sha(config["split_config"]),
        "reference_model_lock_sha256": sha(reference / "MODEL_LOCK.json"),
        "reference_folds_sha256": sha(reference / "train_oof_rf_context/folds.json"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in paths},
        "window_people": sorted(map(int, set(groups))), "fit_windows": len(windows),
        "data_roles_constructed": ["train"], "validation_windows_before_lock": 0, "test_windows_constructed": 0})
    print(f"TRAIN only: {len(windows)} gyro windows / {len(set(groups))} people. Two architectures, nested 3 folds.", flush=True)
    for name in config["models"]:
        directory = output / name
        directory.mkdir()
        scores, seen, audits = np.full(len(windows), np.nan), np.zeros(len(windows), int), []
        for plan in plans:
            fold = plan["fold"]
            destination = directory / f"fold_{fold}"
            destination.mkdir()
            indices = {key: person_indices(groups, plan[key]) for key in
                       ("inner_fit_people", "checkpoint_people", "fit_people", "held_out_people")}
            for key, chosen in indices.items():
                assert_history_isolation(links, groups, chosen, plan[key])
            inner, check = indices["inner_fit_people"], indices["checkpoint_people"]
            seed = config["random_state"] + fold
            print(f"{name} fold {fold + 1}/3: INNER fit {plan['inner_fit_people']} / checkpoint {plan['checkpoint_people']}", flush=True)
            tuned = TemporalClassifier(name, config["optimization"], seed).fit(x[inner], lengths[inner], labels[inner],
                validation=(x[check], lengths[check], labels[check]))
            epoch = tuned.best_epoch_
            pd.DataFrame(tuned.history_).to_csv(destination / "checkpoint_history.csv", index=False)
            joblib.dump(tuned, destination / "checkpoint_model.joblib")
            fit, held = indices["fit_people"], indices["held_out_people"]
            print(f"{name} fold {fold + 1}/3: fresh REFIT all outer TRAIN {len(fit)} windows for {epoch} epochs", flush=True)
            classifier = TemporalClassifier(name, config["optimization"], seed).fit(
                x[fit], lengths[fit], labels[fit], fixed_epochs=epoch)
            pd.DataFrame(classifier.history_).to_csv(destination / "refit_history.csv", index=False)
            joblib.dump(classifier, destination / "oof_model.joblib")
            scores[held] = classifier.predict_proba(x[held], lengths[held])[:, 1]
            seen[held] += 1
            audit = {**plan, "seed": seed, "best_epoch": epoch, "checkpoint_weights_used_for_oof": False,
                     **{key.replace("people", "windows"): len(chosen) for key, chosen in indices.items()}}
            write_json(destination / "fit_audit.json", audit)
            audits.append(audit)
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError("Incomplete participant-held-out TRAIN predictions")
        choice = choose_threshold(labels, scores, config)
        cutoff = choice["selected"]["threshold"]
        epoch = int(np.median([a["best_epoch"] for a in audits]))
        oof = directory / "train_oof"
        oof.mkdir()
        frame = save_predictions(oof, windows, scores, cutoff, name, lengths, "train_oof")
        membership = {p: a["fold"] for a in audits for p in a["held_out_people"]}
        frame["held_out_fold"] = frame.subject_id.astype(int).map(membership)
        frame.to_csv(oof / "predictions.csv", index=False)
        write_json(directory / "threshold_selection.json", choice)
        print(f"{name}: final TRAIN14 fresh fit for median {epoch} epochs; OOF threshold={cutoff:.2f}", flush=True)
        classifier = TemporalClassifier(name, config["optimization"], config["random_state"]).fit(x, lengths, labels, fixed_epochs=epoch)
        pd.DataFrame(classifier.history_).to_csv(directory / "final_training_history.csv", index=False)
        detector = TemporalFoG(classifier, name, GYRO_CHANNELS, (), cutoff, tuple(split["train"]), tuple(split["train"]))
        detector.save(directory / "model.joblib")
        restored = A1FogModel.load(directory / "model.joblib")
        pd.testing.assert_frame_equal(restored.predict_windows(windows[:20]), detector.predict_windows(windows[:20]))
        write_json(directory / "fit_metadata.json", {"fit_people": split["train"], "threshold_people": split["train"],
            "threshold_source": "TRAIN participant-held-out OOF, not validation", "epochs": epoch,
            "inner_best_epochs": [a["best_epoch"] for a in audits], "fit_windows": len(windows),
            "parameters": classifier.parameter_count_, "validation_people_used": [], "test_people_used": [],
            "final_initialization": "fresh random; no transferred fold/author weights", "refit_after_validation": False})
    (output / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    write_json(output / "MODEL_LOCK.json", {"locked_utc": datetime.now(timezone.utc).isoformat(),
        "fit_people": split["train"], "epoch_and_threshold_people": split["train"],
        "validation_windows_before_lock": 0, "test_windows_constructed": 0, "default_replaced": False,
        "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    print("MODEL LOCK complete. Independent VALIDATION process may now score; no TEST evaluation.", flush=True)


if __name__ == "__main__":
    main()
