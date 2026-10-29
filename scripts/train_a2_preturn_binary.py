"""Select binary preturn models using TRAIN subject OOF; evaluate fixed validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fogguard.data.fogstar import sensor_columns
from fogguard.turning.learned import KEYS, RAW_KEYS
from fogguard.turning.preturn_binary import BinaryPreturnModel, dataset


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def create_classifier(family, config):
    if family == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000,
            random_state=config["random_state"]))
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["n_estimators"], min_samples_leaf=config["min_samples_leaf"],
            class_weight="balanced_subsample", random_state=config["random_state"], n_jobs=4)
    raise ValueError("Unknown classifier")


def binary_score(model, features):
    return model.predict_proba(features)[:, list(model.classes_).index(1)]


def alarm_episodes(meta, mask):
    alarms = meta.loc[mask]
    count = 0
    for _, group in alarms.groupby(KEYS):
        count += int(1 + (group.window_end_s.diff().dropna() > 0.75).sum())
    return count


def evaluate(meta, score, threshold, direction=None):
    valid = (meta.input_valid & meta.target.notna()).to_numpy() & np.isfinite(score)
    walking = valid & meta.target.isin(["no_turn", "left", "right"]).to_numpy()
    positive = meta.target.isin(["left", "right"]).to_numpy()
    prediction = score >= threshold
    y, p = positive[walking], prediction[walking]
    true_positive = int((y & p).sum())
    primary = {
        "windows": int(walking.sum()), "positive_windows": int(y.sum()),
        "accuracy": float(accuracy_score(y, p)) if len(y) else None,
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=[False, True], zero_division=0)),
        "balanced_accuracy": (float((p[y].mean() + (~p[~y]).mean()) / 2)
                              if y.any() and (~y).any() else None),
        "AUROC": (float(roc_auc_score(y, score[walking]))
                  if y.any() and (~y).any() else None),
        "average_precision": (float(average_precision_score(y, score[walking]))
                              if y.any() else None),
        "positive_prevalence": float(y.mean()) if len(y) else None,
        "always_no_turn_accuracy": float((~y).mean()) if len(y) else None,
        "confusion_matrix_TN_FP_FN_TP": confusion_matrix(y, p, labels=[False, True]).ravel().tolist(),
        "true_positive_windows": true_positive,
    }
    negative = valid & ~positive
    fp_mask = negative & prediction
    negative_minutes = int(negative.sum()) * 0.5 / 60
    episodes = alarm_episodes(meta, fp_mask)
    event_rows = []
    positive_meta = meta.loc[valid & positive]
    for key, group in positive_meta.groupby(KEYS + ["target_onset_s"]):
        index = group.index.to_numpy()
        detected = prediction[index]
        correct_direction = (detected & (direction[index] == group.target.to_numpy())
                             if direction is not None else np.zeros(len(index), dtype=bool))
        event_rows.append({"detected": bool(detected.any()),
                           "correct_direction": bool(correct_direction.any()),
                           "lead_s": (float(key[-1] - group.loc[detected, "window_end_s"].min())
                                      if detected.any() else None)})
    events = pd.DataFrame(event_rows)
    full = {
        "valid_scored_windows": int(valid.sum()), "false_alarm_episodes": episodes,
        "negative_observed_minutes": negative_minutes,
        "false_alarms_per_minute": episodes / negative_minutes if negative_minutes else None,
        "negative_alarm_fraction": float(fp_mask.sum() / negative.sum()) if negative.any() else None,
        "false_alarm_windows": int(fp_mask.sum()),
        "event_count": len(events), "detected_events": int(events.detected.sum()) if len(events) else 0,
        "event_recall": float(events.detected.mean()) if len(events) else None,
        "correct_direction_events": int(events.correct_direction.sum()) if len(events) else 0,
        "joint_event_recall": float(events.correct_direction.mean()) if len(events) else None,
        "median_detected_lead_s": (float(events.lead_s.dropna().median())
                                   if len(events) and events.lead_s.notna().any() else None),
        "alarm_merge_gap_s": 0.75,
    }
    known_turns = valid & positive
    if direction is not None and known_turns.any():
        full["direction_accuracy_given_true_upcoming_turn"] = float(
            np.mean(direction[known_turns] == meta.loc[known_turns, "target"].to_numpy()))
    return {"walking_binary": primary, "full_stream": full,
            "input_valid_fraction": float(meta.input_valid.mean())}


def choose_threshold(meta, score, config):
    search = []
    for threshold in config["thresholds"]:
        metrics = evaluate(meta, score, threshold)
        feasible = (metrics["full_stream"]["false_alarms_per_minute"]
                    <= config["maximum_oof_false_alarms_per_minute"]
                    and metrics["full_stream"]["negative_alarm_fraction"]
                    <= config["maximum_oof_negative_alarm_fraction"])
        search.append({"threshold": threshold, "feasible": feasible, "metrics": metrics})
    permitted = [r for r in search if r["feasible"]]
    if not permitted:
        return None, search
    selected = max(permitted, key=lambda r: (r["metrics"]["walking_binary"]["f1"],
                                             r["metrics"]["walking_binary"]["precision"],
                                             r["threshold"]))
    return selected, search


def prepare_meta(meta, horizon):
    result = meta.copy()
    result["target"] = result[f"target_{horizon}"]
    result["target_onset_s"] = result[f"onset_{horizon}"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/a2_preturn_binary.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    config = json.loads(args.config.read_text("utf-8"))
    fixed = json.loads(Path(config["split_file"]).read_text("utf-8"))
    if (config["history_samples"], config["short_samples"], config["target_band_s"]) != (
            120, [15, 30, 60, 120], 0.5):
        raise ValueError("Unsupported history/feature schema")
    if sha(args.raw) != fixed["raw_sha256"]:
        raise ValueError("Unexpected raw data hash")
    split = fixed["split"]
    people = set(split["train"]) | set(split["validation"])
    if set(split["train"]) & set(split["validation"]) or people & set(split["test"]):
        raise ValueError("Participant split overlaps")
    args.output.mkdir(parents=True)
    columns = ["timestamp", *RAW_KEYS, "activity", *sensor_columns(("ankleL", "ankleR", "back"))]
    frame = pd.concat([c.loc[c.subjectID.isin(people)] for c in pd.read_csv(
        args.raw, usecols=columns, chunksize=50000)], ignore_index=True)
    datasets = {}
    for feature_set in config["feature_sets"]:
        print(f"Extracting {feature_set}", flush=True)
        datasets[feature_set] = dataset(frame, feature_set, config["horizons_s"])
        print(f"{feature_set}: {datasets[feature_set][0].shape}; "
              f"valid {datasets[feature_set][1].input_valid.sum()}", flush=True)
    reference_meta = next(iter(datasets.values()))[1]
    identity = KEYS + ["window_start_s", "window_end_s"]
    for _, meta in datasets.values():
        pd.testing.assert_frame_equal(reference_meta[identity], meta[identity])
    common_valid = np.logical_and.reduce([m.input_valid.to_numpy() for _, m in datasets.values()])
    train_mask = reference_meta.subject_id.isin(split["train"]).to_numpy()
    val_mask = reference_meta.subject_id.isin(split["validation"]).to_numpy()
    folds = fixed["bilateral_folds"]
    seen = set()
    for fold in folds:
        fit, hold = set(fold["fit_people"]), set(fold["held_out_people"])
        if fit & hold or seen & hold or fit | hold != set(split["train"]):
            raise ValueError("Invalid TRAIN subject fold")
        seen |= hold
    if seen != set(split["train"]):
        raise ValueError("Incomplete TRAIN OOF coverage")
    results = {}
    for horizon in config["horizons_s"]:
        print(f"Selecting horizon {horizon} using TRAIN OOF only", flush=True)
        all_candidates, candidate_arrays = [], {}
        for feature_set, (x, meta) in datasets.items():
            train_meta = prepare_meta(meta.loc[train_mask].reset_index(drop=True), horizon)
            train_meta["input_valid"] = common_valid[train_mask]
            x_train = x[train_mask]
            valid = (train_meta.input_valid & train_meta.target.notna()).to_numpy()
            y = train_meta.target.isin(["left", "right"]).astype(int).to_numpy()
            for family in config["families"]:
                score = np.full(len(train_meta), np.nan)
                direction = np.full(len(train_meta), "unknown", dtype=object)
                for fold in folds:
                    fit = valid & train_meta.subject_id.isin(fold["fit_people"]).to_numpy()
                    hold = valid & train_meta.subject_id.isin(fold["held_out_people"]).to_numpy()
                    trigger = create_classifier(family, config).fit(x_train[fit], y[fit])
                    score[hold] = binary_score(trigger, x_train[hold])
                    positive = fit & (y == 1)
                    head = create_classifier(family, config).fit(
                        x_train[positive], train_meta.loc[positive, "target"])
                    direction[hold] = head.predict(x_train[hold])
                if not np.isfinite(score[valid]).all():
                    raise AssertionError("Missing OOF scores")
                chosen, thresholds = choose_threshold(train_meta, score, config)
                name = f"{feature_set}:{family}"
                record = {"name": name, "feature_set": feature_set, "family": family,
                          "features": x.shape[1], "threshold_search": thresholds,
                          "selected_operating_point": chosen}
                if chosen is not None:
                    record["oof_metrics"] = evaluate(train_meta, score, chosen["threshold"], direction)
                all_candidates.append(record)
                candidate_arrays[name] = (score, direction, train_meta)
                print(name, "OOF", json.dumps(chosen["metrics"]["walking_binary"]
                      if chosen else {"status": "no_feasible_threshold"}), flush=True)
        eligible = [c for c in all_candidates if c["selected_operating_point"] is not None]
        if not eligible:
            raise ValueError("No feasible model at the predefined false alarm budget")
        chosen = max(eligible, key=lambda c: (c["oof_metrics"]["walking_binary"]["f1"],
                                             c["oof_metrics"]["walking_binary"]["average_precision"]))
        feature_set, family = chosen["feature_set"], chosen["family"]
        x, meta = datasets[feature_set]
        full_meta = prepare_meta(meta, horizon)
        fit = (train_mask & common_valid & full_meta.target.notna().to_numpy())
        y = full_meta.target.isin(["left", "right"]).astype(int).to_numpy()
        trigger = create_classifier(family, config).fit(x[fit], y[fit])
        positives = fit & (y == 1)
        head = create_classifier(family, config).fit(x[positives], full_meta.loc[positives, "target"])
        model = BinaryPreturnModel(trigger, head, feature_set, horizon,
                                   chosen["selected_operating_point"]["threshold"])
        val_meta = full_meta.loc[val_mask].reset_index(drop=True)
        val_features = x[val_mask]
        score, direction, _ = model.predict_features(val_features)
        validation = evaluate(val_meta, score, model.threshold, direction)
        common_meta = val_meta.copy()
        common_meta["input_valid"] = common_valid[val_mask]
        common_metrics = evaluate(common_meta, score, model.threshold, direction)
        per_person = {}
        for person in split["validation"]:
            mask = val_meta.subject_id.eq(person).to_numpy()
            per_person[str(person)] = evaluate(val_meta.loc[mask].reset_index(drop=True),
                                               score[mask], model.threshold, direction[mask])
        report = {
            "horizon_s": horizon, "target_band_s": 0.5,
            "selection_scope": "TRAIN subject OOF; all candidates use identical complete-sensor rows",
            "validation_scope": "Previously inspected development-validation people; not blind TEST",
            "selected": {k: chosen[k] for k in ["name", "feature_set", "family", "features"]},
            "threshold": model.threshold, "oof": chosen["oof_metrics"],
            "validation": validation, "validation_common_sensor_rows": common_metrics,
            "per_person_validation": per_person, "candidates": all_candidates,
            "train_positive_windows": int(positives.sum()), "train_fit_windows": int(fit.sum()),
            "direction_status": "Exploratory second head; true-turn and joint recall both reported",
        }
        operational = validation["full_stream"]
        report["binary_operating_target_met"] = (
            operational["event_recall"] >= 0.5
            and operational["false_alarms_per_minute"] <= config["maximum_oof_false_alarms_per_minute"]
            and operational["negative_alarm_fraction"] <= config["maximum_oof_negative_alarm_fraction"])
        directory = args.output / f"preturn_{horizon:.1f}s"
        directory.mkdir()
        model.save(directory / "model.joblib")
        runtime_frame = frame.loc[frame.subjectID.isin(split["validation"])].drop(columns="activity")
        runtime = model.predict_frame(runtime_frame)
        finite = val_meta.input_valid.to_numpy()
        np.testing.assert_allclose(runtime.loc[finite, "turn_score"], score[finite])
        runtime.to_csv(directory / "runtime.csv", index=False)
        val_meta[identity + ["target", "target_onset_s"]].to_csv(
            directory / "evaluation_truth.csv", index=False)
        oof_score, oof_direction, oof_meta = candidate_arrays[chosen["name"]]
        oof_meta.assign(turn_score=oof_score, predicted_direction=oof_direction).to_csv(
            directory / "oof_evaluation.csv", index=False)
        (directory / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), "utf-8")
        val_known = finite & val_meta.target.isin(["left", "right", "no_turn"]).to_numpy()
        labels = val_meta.loc[val_known, "target"].isin(["left", "right"])
        precision, recall, _ = precision_recall_curve(labels, score[val_known])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision, label="Binary preturn")
        ax.axhline(labels.mean(), color="grey", linestyle="--", label="Positive prevalence")
        ax.scatter([validation["walking_binary"]["recall"]],
                   [validation["walking_binary"]["precision"]], color="red", label="TRAIN-selected threshold")
        ax.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.05), xlim=(0, 1),
               title=f"Development validation: minimum lead {horizon:.1f}s")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "precision_recall.png", dpi=180)
        plt.close(fig)
        results[str(horizon)] = report
        print("VALIDATION", horizon, json.dumps(validation), flush=True)
    sources = [Path(__file__), Path("src/fogguard/turning/preturn_binary.py"),
               Path("src/fogguard/turning/learned.py")]
    manifest = {
        "schema": "fogguard.a2.binary.results.v1", "config": config, "split": split,
        "raw_sha256": sha(args.raw), "config_sha256": sha(args.config),
        "source_sha256": {p.resolve().relative_to(Path.cwd()).as_posix(): sha(p) for p in sources},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "test_scored": False, "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), "utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
