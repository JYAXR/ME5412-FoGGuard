"""Fit a fixed bilateral feature block ablation using TRAIN people only."""

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
from fogguard.modeling.a1_coordination import EXTRA_NAMES, CoordinationFoG, coordination_matrix
from fogguard.modeling.a1_fixed_split import prediction_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/a1_coordination_v1.json"))
    args = parser.parse_args()
    run, root = args.run_dir, Path(__file__).resolve().parents[1]
    if run.exists():
        parser.error("Use a new run directory")
    config = json.loads(args.config.read_text("utf-8"))
    reference = Path(config["reference_run"])
    check_lock(reference, json.loads((reference / "MODEL_LOCK.json").read_text("utf-8")))
    protocol = json.loads((reference / "protocol.json").read_text("utf-8"))
    if sha(args.raw) != protocol["raw_sha256"]:
        raise ValueError("Raw data changed")
    split = json.loads(Path(config["split_config"]).read_text("utf-8"))["split"]
    if split != json.loads((reference / "split.json").read_text("utf-8")):
        raise ValueError("People split changed")
    if tuple("coord__" + n for n in config["additional_features"]) != EXTRA_NAMES or config["epsilon"] != 1e-12 or config["max_lag_samples"] != 30:
        raise ValueError("Feature protocol differs")
    raw = pd.read_csv(args.raw)
    train = role_frame(raw, split, "train")
    del raw
    windows = anchored_gyro_windows(train, require_right=True)
    plans = json.loads((reference / "fold_plan.json").read_text("utf-8"))
    old = pd.read_csv(reference / "bilateral_rf/oof_predictions.csv", float_precision="round_trip")
    if len(windows) != len(old) or any(abs(w.end_s - r.window_end_s) > 1e-6 or int(w.subject_id) != r.subject_id for w, r in zip(windows, old.itertuples(), strict=True)):
        raise ValueError("TRAIN input schedule changed")
    print(f"TRAIN only: {len(windows)} matched windows; extract 66+12 fixed features", flush=True)
    banks = coordination_matrix(windows)
    y, people = np.array([w.fog_label for w in windows]), np.array([int(w.subject_id) for w in windows])
    run.mkdir(parents=True)
    for name, value in [("config.json", config), ("split.json", split), ("fold_plan.json", plans)]:
        write_json(run / name, value)
    sources = [*sorted((root / "src/fogguard").rglob("*.py")), root / "recognition/diagnostics.py"]
    sources += [root / f"scripts/{name}.py" for name in ("train_a1_coordination", "evaluate_a1_coordination", "verify_a1_coordination", "report_a1_coordination", "verify_a1_bilateral", "report_a1_author_cnn", "run_a1_event")]
    write_json(run / "protocol.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "raw_sha256": sha(args.raw), "config_sha256": sha(args.config), "split_sha256": sha(config["split_config"]),
        "reference_lock_sha256": sha(reference / "MODEL_LOCK.json"),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in sources},
        "data_roles_constructed": ["train"], "validation_windows_before_lock": 0, "test_windows_constructed": 0})
    pd.DataFrame(banks["coordination_rf"][0][:, 66:], columns=EXTRA_NAMES).to_csv(run / "train_coordination_features.csv", index=False)
    actual = tuple(sorted(map(int, set(people))))
    for name in config["models"]:
        values, names = banks[name]
        directory = run / name
        directory.mkdir()
        scores, seen, fold_metadata = np.full(len(y), np.nan), np.zeros(len(y), int), []
        for plan in plans:
            fit, held = [np.flatnonzero(np.isin(people, plan[p])) for p in ("fit_people", "held_out_people")]
            if set(plan["fit_people"]) & set(plan["held_out_people"]) or set(y[fit]) != {0, 1}:
                raise ValueError("Invalid participant fold")
            print(f"{name} fold{plan['fold']}: fit={len(fit)}, held={len(held)}", flush=True)
            classifier = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(values[fit], y[fit])
            joblib.dump(classifier, directory / f"fold_{plan['fold']}.joblib")
            scores[held], seen[held] = classifier.predict_proba(values[held])[:, 1], seen[held] + 1
            fold_metadata.append({"fold": plan["fold"], "actual_fit_people": sorted(map(int, set(people[fit]))), "held_people": plan["held_out_people"]})
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError("Incomplete TRAIN OOF")
        if name == "bilateral_control":
            np.testing.assert_allclose(scores, old.fog_probability, atol=1e-12, rtol=0)
        choice = choose_threshold(y, scores, config)
        threshold = choice["selected"]["threshold"]
        frame = annotate(windows, prediction_frame(windows, scores, threshold, name), "train_oof")
        frame["model_version"], frame["prediction_source"] = f"a1_coordination_v1/{name}", "participant_held_out_train_oof"
        frame.to_csv(directory / "oof_predictions.csv", index=False)
        write_json(directory / "threshold_selection.json", choice)
        classifier = RandomForestClassifier(**config["rf"], random_state=config["random_state"], n_jobs=-1).fit(values, y)
        model = CoordinationFoG(classifier, name, DUAL, names, threshold, actual, actual)
        model.save(directory / "model.joblib")
        pd.testing.assert_frame_equal(model.predict_windows(windows[:5]), CoordinationFoG.load(directory / "model.joblib").predict_windows(windows[:5]))
        write_json(directory / "fit_metadata.json", {"fit_people": list(actual), "threshold_people": list(actual),
            "fit_windows": len(y), "features": len(names), "folds": fold_metadata,
            "validation_people_used": [], "test_people_used": []})
    (run / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    write_json(run / "MODEL_LOCK.json", {"locked_utc": datetime.now(timezone.utc).isoformat(), "fit_people": list(actual),
        "validation_windows_before_lock": 0, "test_windows_constructed": 0,
        "files": {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}})
    print("Models locked; old 66D TRAIN OOF reproduced. VALIDATION may now be evaluated separately.", flush=True)


if __name__ == "__main__":
    main()
