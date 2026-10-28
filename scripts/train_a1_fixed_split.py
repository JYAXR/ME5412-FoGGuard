"""TRAIN-only fresh fits, validation-only choices, immutable lock before testing."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from run_a1_experiment import sha, write_json

from fogguard.data.fogstar import iter_fogstar_windows, sensor_columns
from fogguard.evaluation.a1_fixed_split import (
    check_split,
    choose_threshold,
    classifier_for,
    role_frame,
    training_noise_statistics,
)
from fogguard.modeling.a1_fixed_split import (
    FixedSplitFoG,
    feature_bank,
    feature_kind,
    prediction_frame,
    scores_for,
)
from fogguard.modeling.a1_fog import A1FogModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_fixed_split_v1.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    output, root = args.run_dir, Path(__file__).resolve().parents[1]
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh training directory")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/matplotlib-a1-fixed-cache"))
    config = json.loads(args.config.read_text("utf-8"))
    raw = pd.read_csv(args.raw)
    check_split(config["split"], raw.subjectID.unique())
    config_text = json.dumps(config, indent=2, ensure_ascii=False)
    (output / "frozen_config.json").write_text(config_text, encoding="utf-8")
    roles = {int(p): role for role in ("train", "validation", "test") for p in config["split"][role]}
    assignments = raw.groupby(["subjectID", "sessionID", "taskID"]).agg(samples=("fog", "size"), fog_samples=("fog", "sum")).reset_index()
    assignments["role"] = assignments.subjectID.astype(int).map(roles)
    assignments.to_csv(output / "record_assignments.csv", index=False)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha(args.config),
        "raw_sha256": sha(args.raw), "split": config["split"],
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in [root / "scripts/train_a1_fixed_split.py",
            *sorted((root / "src/fogguard").rglob("*.py"))]},
        "test_usage_before_lock": "Subject assignment/count metadata only; no test windows, fitting, thresholds or performance"}
    write_json(output / "protocol.json", protocol)
    train_raw, validation_raw = role_frame(raw, config["split"], "train"), role_frame(raw, config["split"], "validation")
    del raw
    noise = training_noise_statistics(train_raw, sensor_columns(("ankleL", "back")), config["split"]["train"])
    write_json(output / "training_noise_stats.json", noise)
    banks, windows_by_role = {}, {}
    for role, frame in [("train", train_raw), ("validation", validation_raw)]:
        print(f"Building {role.upper()} sensor windows only ...", flush=True)
        for kind, positions in [("primary", ("ankleL", "back")), ("ankle", ("ankleL",))]:
            windows = list(iter_fogstar_windows(frame, sensor_positions=positions, max_interpolation_gap_s=0))
            if not windows or {int(w.subject_id) for w in windows} - set(config["split"][role]):
                raise ValueError("Wrong role windows or no valid inputs")
            windows_by_role[role, kind] = windows
            banks[role, kind] = feature_bank(windows, ankle=kind == "ankle")
    choices = {}
    for model_id in config["models"]:
        print(f"Fresh TRAIN fit: {model_id}; threshold selection on VALIDATION only", flush=True)
        kind = "ankle" if model_id == "rf_ankle" else "primary"
        train_windows, val_windows = windows_by_role["train", kind], windows_by_role["validation", kind]
        x, names = banks["train", kind][feature_kind(model_id)]
        val_x, val_names = banks["validation", kind][feature_kind(model_id)]
        if names != val_names:
            raise ValueError("Validation feature schema differs")
        y, val_y = np.array([w.fog_label for w in train_windows]), np.array([w.fog_label for w in val_windows])
        classifier = classifier_for(model_id, config).fit(x, y)
        score = scores_for(classifier, model_id, val_x)
        audit = choose_threshold(val_y, score, config)
        cutoff = audit["selected"]["threshold"]
        directory = output / model_id
        directory.mkdir()
        detector = FixedSplitFoG(classifier, model_id, train_windows[0].channel_names, names, cutoff,
                                 fit_participants=tuple(config["split"]["train"]),
                                 threshold_participants=tuple(config["split"]["validation"]))
        detector.save(directory / "model.joblib")
        loaded = A1FogModel.load(directory / "model.joblib")
        if loaded.fit_participants != detector.fit_participants or loaded.threshold != cutoff:
            raise ValueError("Saved model fit provenance differs")
        # Infer with saved object on a short sequence, not on any test input.
        expected = detector.predict_windows(val_windows[:10])
        pd.testing.assert_frame_equal(expected, loaded.predict_windows(val_windows[:10]))
        val = prediction_frame(val_windows, score, cutoff, model_id)
        val["truth_fog_label"], val["truth_fog_fraction"] = val_y, [w.fog_fraction for w in val_windows]
        val["data_role"] = "validation"
        val.to_csv(directory / "validation_predictions.csv", index=False)
        write_json(directory / "validation_selection.json", audit)
        write_json(directory / "fit_metadata.json", {"fit_people": list(detector.fit_participants),
            "threshold_people": list(detector.threshold_participants), "fit_windows": len(x),
            "validation_windows": len(val_x), "feature_count": len(names) if names else "12x120 waveform",
            "training_loss": getattr(classifier, "loss_history_", None),
            "test_people_used": [], "retrained_after_validation": False})
        choices[model_id] = audit
        write_json(output / "validation_choices.json", choices)
    eligible = [m for m in config["models"] if m != "rf_ankle"]
    winner = max(enumerate(eligible), key=lambda pair: (
        choices[pair[1]]["selected"]["fog_f1"], choices[pair[1]]["average_precision"], -pair[0]))[1]
    environment = subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "environment.txt").write_text(environment, encoding="utf-8")
    selected = {"primary_model": winner, "fallback_model": "rf_ankle", "chosen_on": "validation only",
                "test_people": config["split"]["test"], "do_not_reselect_on_test": True}
    write_json(output / "validation_winner.json", selected)
    paths = [p for p in output.rglob("*") if p.is_file()]
    lock = {"locked_utc": datetime.now(timezone.utc).isoformat(), "selection": selected,
            "fit_people": config["split"]["train"], "threshold_people": config["split"]["validation"],
            "test_people": config["split"]["test"], "files": {str(p.relative_to(output)): sha(p) for p in paths}}
    write_json(output / "TRAINING_LOCK.json", lock)
    print(f"TRAINING LOCKED. Validation-selected primary: {winner}. No TEST scores evaluated.", flush=True)
    print(f"Lock SHA256: {hashlib.sha256((output / 'TRAINING_LOCK.json').read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
