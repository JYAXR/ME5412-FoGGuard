"""Train and evaluate A2 using fixed TRAIN/validation people; never score TEST."""

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from fogguard.data.fogstar import sensor_columns
from fogguard.turning.learned import (
    CURRENT_CLASSES,
    KEYS,
    PRETURN_CLASSES,
    DirectionModel,
    build_dataset,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scores(truth, predicted, classes):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    p, r, f, n = precision_recall_fscore_support(
        truth, predicted, labels=classes, zero_division=0)
    return {
        "windows": len(truth),
        "accuracy": float(accuracy_score(truth, predicted)) if len(truth) else None,
        "macro_f1": float(f.mean()),
        "classes": classes,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
        "per_class": {c: {"precision": float(a), "recall": float(b), "f1": float(v),
                          "support": int(s)} for c, a, b, v, s in zip(classes, p, r, f, n)},
        "prediction_coverage": float(np.mean(predicted != "unknown")) if len(truth) else None,
    }


def evaluate(meta, pred, classes):
    valid = meta.input_valid.to_numpy() & meta.target.notna().to_numpy()
    known = valid & meta.target.ne("unknown").to_numpy()
    all_report = scores(meta.loc[valid, "target"], pred[valid], classes)
    primary = scores(meta.loc[known, "target"], pred[known], classes)
    primary["macro_f1"] = float(np.mean([primary["per_class"][c]["f1"]
                                        for c in classes if c != "unknown"]))
    accepted = known & (pred != "unknown")
    primary["accepted_accuracy"] = (float(accuracy_score(meta.loc[accepted, "target"],
                                                        pred[accepted]))
                                    if accepted.any() else None)
    report = {"primary_known_targets": primary, "full_stream_including_unknown": all_report,
              "total_grid_windows": len(meta), "unscored_windows": int((~valid).sum()),
              "missing_sensor_windows": int((~meta.input_valid).sum()),
              "censored_future_windows": int(meta.target.isna().sum()),
              "per_subject": {}}
    for person in sorted(meta.subject_id.unique()):
        mask = known & meta.subject_id.eq(person).to_numpy()
        person_report = scores(meta.loc[mask, "target"], pred[mask], classes)
        person_report["macro_f1"] = float(np.mean([
            person_report["per_class"][c]["f1"] for c in classes if c != "unknown"]))
        report["per_subject"][str(person)] = person_report
    if "no_turn" in classes:
        negative = valid & ~meta.target.isin(["left", "right"]).to_numpy()
        alarms = meta.loc[negative & np.isin(pred, ["left", "right"]),
                          KEYS + ["window_end_s"]].copy()
        count = 0
        for _, group in alarms.groupby(KEYS):
            count += int(1 + (group.window_end_s.diff().dropna() > 0.75).sum())
        minutes = int(negative.sum()) * 0.5 / 60
        events = meta.loc[valid & meta.target.isin(["left", "right"])]
        covered, total = 0, 0
        for _, group in events.groupby(KEYS + ["target_onset_s"]):
            total += 1
            covered += int(np.any(pred[group.index] == group.target.to_numpy()))
        report["preturn_events"] = {
            "evaluable_events": total, "correct_direction_events": covered,
            "missed_or_wrong_events": total - covered,
            "recall": covered / total if total else None,
            "false_alarm_episodes": count, "negative_observed_minutes": minutes,
            "false_alarms_per_minute": count / minutes if minutes else None,
            "alarm_episode_merge_gap_s": 0.75,
            "scope": "Events with a complete walking-history grid window in the target band",
        }
        report["majority_no_turn_accuracy"] = float(meta.loc[known, "target"].eq("no_turn").mean())
    else:
        report["majority_straight_accuracy"] = float(meta.loc[known, "target"].eq("straight").mean())
    return report


def yaw_baseline(train_meta, validation_meta):
    mask = train_meta.input_valid & train_meta.target.isin(CURRENT_CLASSES[:3])
    best = (-1, None)
    for sign in (-1, 1):
        for threshold in (5, 10, 15, 20, 30, 45):
            rate = train_meta.loc[mask, "back_vertical_rate"].to_numpy() * sign
            pred = np.where(rate > threshold, "left",
                            np.where(rate < -threshold, "right", "straight"))
            score = f1_score(train_meta.loc[mask, "target"], pred, labels=CURRENT_CLASSES[:3],
                             average="macro", zero_division=0)
            if score > best[0]:
                best = (score, (sign, threshold))
    sign, threshold = best[1]
    rate = validation_meta.back_vertical_rate.to_numpy() * sign
    pred = np.where(rate > threshold, "left",
                    np.where(rate < -threshold, "right", "straight")).astype(object)
    pred[~validation_meta.input_valid.to_numpy()] = "unknown"
    return {"sign": sign, "threshold_dps": threshold,
            "selection": "TRAIN macro-F1 only", **evaluate(validation_meta, pred, CURRENT_CLASSES)}


def preturn_support(frame, meta, pred, horizon):
    """Count all annotated walk->turn onsets, including unavailable histories."""
    events = []
    for key, group in frame.groupby(["subjectID", "sessionID", "taskID"], sort=False):
        times = group.timestamp.to_numpy(float)
        activity = group.activity.to_numpy(int)
        onsets = np.flatnonzero(np.isin(activity[1:], [6, 7]) & (activity[:-1] == 1)) + 1
        for index in onsets:
            onset = times[index]
            mask = np.ones(len(meta), dtype=bool)
            for column, value in zip(KEYS, key, strict=True):
                mask &= meta[column].eq(value).to_numpy()
            mask &= (meta.target_onset_s.sub(onset).abs() < 1e-7).to_numpy()
            mask &= meta.input_valid.to_numpy()
            target = "left" if activity[index] == 7 else "right"
            correct = mask & (pred == target)
            events.append({**dict(zip(KEYS, map(int, key), strict=True)),
                           "onset_s": float(onset), "target": target,
                           "evaluable": bool(mask.any()), "correct": bool(correct.any()),
                           "actual_lead_s": (float(onset - meta.loc[correct, "window_end_s"].min())
                                             if correct.any() else None)})
    table = pd.DataFrame(events)
    return table, {"annotated_walk_to_turn_onsets": len(table),
                   "evaluable_events": int(table.evaluable.sum()) if len(table) else 0,
                   "unavailable_events": int((~table.evaluable).sum()) if len(table) else 0,
                   "correct_events": int(table.correct.sum()) if len(table) else 0,
                   "all_annotated_onset_recall": float(table.correct.mean()) if len(table) else None,
                   "minimum_lead_s": horizon, "target_band_s": 0.5}


def plot_confusion(report, path, title):
    matrix = np.array(report["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > matrix.max() / 2 else "black")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=report["classes"],
           yticklabels=report["classes"], xlabel="Prediction", ylabel="Truth", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fogstar/sensor_data.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/a2_turning.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["current", "preturn", "all"], default="all")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    config = json.loads(args.config.read_text("utf-8"))
    if (config["window_samples"], config["step_samples"], config["sample_rate_hz"],
            config["preturn_target_band_s"]) != (120, 30, 60, 0.5):
        raise ValueError("This feature schema requires 120/30 samples at 60 Hz and a 0.5 s band")
    split_path = Path(config["split_file"])
    split_config = json.loads(split_path.read_text("utf-8"))
    split = split_config["split"]
    if (set(split["train"]) & set(split["validation"]) or
            (set(split["train"]) | set(split["validation"])) & set(split["test"])):
        raise ValueError("Participant leakage in split")
    raw_hash = digest(args.raw)
    if raw_hash != split_config["raw_sha256"]:
        raise ValueError("Raw data hash does not match the fixed A1 split source")
    args.output.mkdir(parents=True)
    channels = sensor_columns(config["positions"])
    usecols = ["timestamp", "subjectID", "sessionID", "taskID", "activity", *channels]
    selected_people = set(split["train"]) | set(split["validation"])
    # Filter chunks before retaining data. TEST is not featurized or evaluated.
    frame = pd.concat([c.loc[c.subjectID.isin(selected_people)]
                       for c in pd.read_csv(args.raw, usecols=usecols, chunksize=50000)],
                      ignore_index=True)
    print(f"Loaded {len(frame)} TRAIN/validation samples; TEST excluded", flush=True)
    horizons = ([0.0] if args.task == "current" else config["preturn_horizons_s"]
                if args.task == "preturn" else [0.0, *config["preturn_horizons_s"]])
    results = {}
    for horizon in horizons:
        name = "current" if not horizon else f"preturn_{horizon:.1f}s"
        print(f"Building {name} dataset", flush=True)
        features, meta = build_dataset(frame, channels, horizon, config["current_label_purity"])
        fit_mask = (meta.subject_id.isin(split["train"]) & meta.input_valid
                    & meta.target.notna()).to_numpy()
        val_mask = meta.subject_id.isin(split["validation"]).to_numpy()
        train_meta = meta.loc[fit_mask].reset_index(drop=True)
        val_meta = meta.loc[val_mask].reset_index(drop=True)
        val_features = features[val_mask]
        print(f"{name} TRAIN: {train_meta.target.value_counts().to_dict()}; "
              f"validation: {val_meta.target.value_counts().to_dict()}", flush=True)
        classes = PRETURN_CLASSES if horizon else CURRENT_CLASSES
        eligible = (val_meta.input_valid & val_meta.target.notna()).to_numpy()
        candidate_results, candidates = [], []
        # Small, explicit feature comparison on development validation, never TEST.
        for feature_set in config["feature_sets"]:
            if feature_set not in ("full", "gravity_rotation"):
                raise ValueError("Unsupported feature set")
            indices = (None if feature_set == "full" else
                       tuple(range(features.shape[1] - 12, features.shape[1])))
            fit_features = features[fit_mask]
            if indices is not None:
                fit_features = fit_features[:, indices]
            clf = RandomForestClassifier(
                n_estimators=config["n_estimators"], min_samples_leaf=config["min_samples_leaf"],
                class_weight="balanced_subsample", random_state=config["random_state"], n_jobs=4)
            clf.fit(fit_features, meta.loc[fit_mask, "target"])
            candidate = DirectionModel(clf, channels, horizon_s=horizon, feature_indices=indices)
            threshold_scores = []
            for threshold in config["rejection_thresholds"]:
                candidate.rejection_threshold = threshold
                pred, _ = candidate.predict_features(val_features)
                score = f1_score(val_meta.loc[eligible, "target"], pred[eligible], labels=classes,
                                 average="macro", zero_division=0)
                threshold_scores.append({"threshold": threshold, "macro_f1": float(score)})
            chosen = max(threshold_scores, key=lambda r: r["macro_f1"])
            candidate.rejection_threshold = chosen["threshold"]
            candidate_pred, _ = candidate.predict_features(val_features)
            candidate_pred[~val_meta.input_valid.to_numpy()] = "unknown"
            candidate_results.append({"feature_set": feature_set, **chosen,
                                      "threshold_search": threshold_scores,
                                      "metrics": evaluate(val_meta, candidate_pred, classes)})
            candidates.append(candidate)
        best = max(range(len(candidates)), key=lambda i: candidate_results[i]["macro_f1"])
        model = candidates[best]
        selected = candidate_results[best]
        pred, conf = model.predict_features(val_features)
        pred[~val_meta.input_valid.to_numpy()] = "unknown"
        conf[~val_meta.input_valid.to_numpy()] = 0.0
        report = evaluate(val_meta, pred, classes)
        report.update(horizon_s=horizon, rejection_threshold=model.rejection_threshold,
                      feature_set=selected["feature_set"], candidate_results=candidate_results,
                      train_class_counts=train_meta.target.value_counts().to_dict(),
                      feature_count=(features.shape[1] if model.feature_indices is None else
                                     len(model.feature_indices)),
                      evaluation_scope="Participant-held-out development validation; "
                                       "feature set and rejection threshold selected on this set")
        if not horizon:
            report["yaw_baseline"] = yaw_baseline(train_meta, val_meta)
        else:
            recalls = [report["primary_known_targets"]["per_class"][c]["recall"]
                       for c in ["left", "right"]]
            report["deployment_status"] = ("research_only" if min(recalls) < 0.5 else
                                           "candidate_requires_independent_evaluation")
            event_table, support = preturn_support(
                frame.loc[frame.subjectID.isin(split["validation"])], val_meta, pred, horizon)
            report["onset_support_audit"] = support
        directory = args.output / name
        directory.mkdir()
        if horizon:
            event_table.to_csv(directory / "event_support.csv", index=False)
        model.save(directory / "model.joblib")
        runtime = val_meta[KEYS + ["window_start_s", "window_end_s", "available_at_s",
                                  "input_valid", "input_reason", "direction_task", "horizon_s"]]
        runtime = runtime.assign(task_prediction=pred, task_confidence=conf,
                                 direction=np.where(pred == "no_turn", "unknown", pred),
                                 direction_confidence=np.where(
                                     np.isin(pred, ["no_turn", "unknown"]), 0.0, conf),
                                 target_band_s=0.5 if horizon else 0.0,
                                 model_version="a2-rf-v1", prediction_source="real_imu")
        runtime.to_csv(directory / "direction_runtime.csv", index=False)
        val_meta[KEYS + ["window_start_s", "window_end_s", "target", "target_onset_s"]].to_csv(
            directory / "evaluation_truth.csv", index=False)
        (directory / "metrics.json").write_text(json.dumps(report, indent=2), "utf-8")
        failures = val_meta.loc[eligible & (pred != val_meta.target.to_numpy())].copy()
        failures["predicted"] = pred[failures.index]
        failures.head(100).to_csv(directory / "failure_examples.csv", index=False)
        plot_confusion(report["full_stream_including_unknown"], directory / "confusion.png", name)
        # Verify the public runtime route cannot depend on annotation columns.
        runtime_frame = frame.loc[frame.subjectID.isin(split["validation"])].drop(columns="activity")
        check = model.predict_frame(runtime_frame)
        if not np.array_equal(check.task_prediction.to_numpy(), pred):
            raise AssertionError("Runtime and evaluated predictions disagree")
        results[name] = report
        print(f"{name}: " + json.dumps(report["primary_known_targets"]), flush=True)
    manifest = {
        "schema": "fogguard.a2.results.v1", "split": split, "raw_sha256": raw_hash,
        "config": config, "config_sha256": digest(args.config),
        "source_sha256": {p.resolve().relative_to(Path.cwd()).as_posix(): digest(p)
                          for p in [Path(__file__), Path("src/fogguard/turning/learned.py")]},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "test_scored": False, "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
