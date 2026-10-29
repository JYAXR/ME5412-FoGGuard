"""TRAIN-only matched per-axis66 / invariant10 RF ablation, then immutable lock."""

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
from sklearn.ensemble import RandomForestClassifier

from fogguard.evaluation.a1_author import annotate
from fogguard.evaluation.a1_fixed_split import choose_threshold, role_frame
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_fixed_split import prediction_frame
from fogguard.modeling.a1_invariant import InvariantFoG, feature_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_invariant_v1.json"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    if run.exists():
        parser.error("Use a new run directory")
    config = json.loads(args.config.read_text("utf-8"))
    reference = Path(config["reference_run"])
    check_lock(reference, json.loads((reference / "MODEL_LOCK.json").read_text("utf-8")))
    old_protocol = json.loads((reference / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != old_protocol["raw_sha256"]:
        raise ValueError("Source raw data changed")
    split = json.loads(Path(config["split_config"]).read_text("utf-8"))["split"]
    if split != json.loads((reference / "split.json").read_text("utf-8")) or config["models"] != ["axis_control", "invariant_rf"]:
        raise ValueError("Unexpected split or variants")
    raw = pd.read_csv(args.raw)
    train = role_frame(raw, split, "train")
    del raw
    windows = anchored_gyro_windows(train, require_right=True)
    old = pd.read_csv(reference / "bilateral_rf/oof_predictions.csv", float_precision="round_trip")
    for w, p in zip(windows, old.itertuples(), strict=True):
        assert (int(w.subject_id), int(w.session_id), int(w.task_id)) == (p.subject_id, p.session_id, p.task_id)
        assert abs(w.end_s - p.window_end_s) < 1e-6 and w.fog_label == p.truth_fog_label
    print(f"TRAIN only: {len(windows)} matched windows; no source VAL/TEST windows", flush=True)
    plans = json.loads((reference / "fold_plan.json").read_text("utf-8"))
    people = np.array([int(w.subject_id) for w in windows])
    y = np.array([w.fog_label for w in windows])
    actual = tuple(sorted(map(int, set(people))))
    run.mkdir(parents=True)
    for name, value in (("config.json", config), ("split.json", split), ("fold_plan.json", plans)):
        write_json(run / name, value)
    sources = [*sorted((root / "src/fogguard").rglob("*.py")), root / "recognition/diagnostics.py"]
    sources += [root / f"scripts/{name}.py" for name in ("train_a1_invariant", "evaluate_a1_invariant", "run_a1_event", "report_a1_author_cnn")]
    write_json(run / "protocol.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "raw_sha256": sha(args.raw),
        "config_sha256": sha(args.config), "reference_lock_sha256": sha(reference / "MODEL_LOCK.json"),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in sources},
        "data_roles_constructed": ["train"], "validation_windows_before_lock": 0, "test_windows_constructed": 0})
    for name in config["models"]:
        values, names = feature_matrix(windows, name)
        folder = run / name
        folder.mkdir()
        np.save(folder / "train_features.npy", values)
        scores, seen, metadata = np.full(len(y), np.nan), np.zeros(len(y), int), []
        for plan in plans:
            fit, held = [np.flatnonzero(np.isin(people, plan[k])) for k in ("fit_people", "held_out_people")]
            if set(plan["fit_people"]) & set(plan["held_out_people"]) or set(y[fit]) != {0, 1}:
                raise ValueError("Invalid TRAIN participant fold")
            print(f"{name} fold{plan['fold']}: fit {len(fit)}, held {len(held)}", flush=True)
            classifier = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(values[fit], y[fit])
            joblib.dump(classifier, folder / f"fold_{plan['fold']}.joblib")
            scores[held], seen[held] = classifier.predict_proba(values[held])[:, 1], seen[held] + 1
            metadata.append({"fold": plan["fold"], "actual_fit_people": sorted(map(int, set(people[fit]))), "held_people": plan["held_out_people"]})
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError("Incomplete TRAIN OOF")
        if name == "axis_control":
            np.testing.assert_allclose(scores, old.fog_probability, atol=1e-12, rtol=0)
        choice = choose_threshold(y, scores, config)
        threshold = choice["selected"]["threshold"]
        frame = annotate(windows, prediction_frame(windows, scores, threshold, name), "train_oof")
        frame["prediction_source"] = "participant_held_out_train_oof_calibration"
        frame.to_csv(folder / "oof_predictions.csv", index=False)
        write_json(folder / "threshold_selection.json", choice)
        classifier = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(values, y)
        model = InvariantFoG(classifier, name, DUAL, names, threshold, actual, actual)
        model.save(folder / "model.joblib")
        pd.testing.assert_frame_equal(model.predict_windows(windows[:3]), InvariantFoG.load(folder / "model.joblib").predict_windows(windows[:3]))
        write_json(folder / "fit_metadata.json", {"fit_people": list(actual), "threshold_people": list(actual),
            "fit_windows": len(y), "features": len(names), "folds": metadata, "validation_people_used": [], "test_people_used": []})
        print(f"{name}: TRAIN OOF selected {choice['selected']}", flush=True)
    (run / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), "utf-8")
    write_json(run / "MODEL_LOCK.json", {"locked_utc": datetime.now(timezone.utc).isoformat(), "fit_people": list(actual),
        "validation_windows_before_lock": 0, "test_windows_constructed": 0,
        "files": {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}})
    print("Both models/thresholds locked; old 66D TRAIN OOF reproduced", flush=True)


if __name__ == "__main__":
    main()
