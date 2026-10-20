"""Train and evaluate a subject-independent FoG-STAR multi-task baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from fogguard.data import load_fogstar_windows
from fogguard.modeling import (
    FogStarMultiTaskModel,
    build_feature_matrix,
    create_baseline_classifiers,
)


def _jsonable_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[object]) -> dict:
    details = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "balanced_accuracy": float(np.mean([details[str(label)]["recall"] for label in labels])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "outside_label_predictions": int((~np.isin(y_pred, labels)).sum()),
        "classification_report": details,
    }


def _select_fog_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    return float(
        max(
            candidates,
            key=lambda threshold: (
                balanced_accuracy_score(y_true, probabilities >= threshold),
                f1_score(y_true, probabilities >= threshold, average="macro"),
                threshold,
            ),
        )
    )


def train_and_evaluate(
    dataset_path: Path,
    *,
    sensor_positions: tuple[str, ...],
    n_splits: int,
    n_estimators: int,
    random_state: int,
) -> tuple[FogStarMultiTaskModel, dict]:
    windows = load_fogstar_windows(dataset_path, sensor_positions=sensor_positions)
    if any(window.fog_label is None or window.direction_label is None for window in windows):
        raise ValueError("Training requires FoG and activity labels on every window")

    x, feature_names = build_feature_matrix(windows)
    fog_feature_indices = tuple(
        index
        for index, name in enumerate(feature_names)
        if not name.endswith(("__raw_mean", "__raw_median"))
    )
    fog_x = x[:, fog_feature_indices]
    y_fog = np.asarray([window.fog_label for window in windows], dtype=int)
    y_direction = np.asarray([window.direction_label for window in windows], dtype=object)
    y_context = np.where(y_direction == "unknown", "other", y_direction)
    groups = np.asarray([window.subject_id for window in windows], dtype=object)
    direction_trainable = y_direction != "unknown"
    unique_subjects = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique_subjects):
        raise ValueError("n_splits must be between 2 and the number of subjects")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fog_prediction = np.full(len(windows), -1, dtype=int)
    fog_probability = np.full(len(windows), np.nan, dtype=float)
    direction_prediction = np.full(len(windows), "unknown", dtype=object)
    fold_subjects: list[dict[str, list[str]]] = []
    fold_fog_thresholds: list[float] = []

    for train_index, test_index in splitter.split(fog_x, y_fog, groups):
        fog_model, direction_model = create_baseline_classifiers(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        fog_model.fit(fog_x[train_index], y_fog[train_index])
        fog_classes = list(fog_model.classes_)
        fog_index = fog_classes.index(1)
        threshold = _select_fog_threshold(
            y_fog[train_index],
            fog_model.oob_decision_function_[:, fog_index],
        )
        fold_fog_thresholds.append(threshold)
        direction_model.fit(x[train_index], y_context[train_index])
        fog_probability[test_index] = fog_model.predict_proba(fog_x[test_index])[:, fog_index]
        fog_prediction[test_index] = fog_probability[test_index] >= threshold
        direction_prediction[test_index] = direction_model.predict(x[test_index])
        fold_subjects.append(
            {
                "train": sorted(set(groups[train_index])),
                "test": sorted(set(groups[test_index])),
            }
        )

    direction_labels = ["straight", "left", "right"]
    valid_state = (y_fog == 1) | direction_trainable
    true_state = np.where(y_fog == 1, "fog", y_direction)
    predicted_state = np.where(fog_prediction == 1, "fog", direction_prediction)
    report = {
        "dataset": {
            "path": str(dataset_path),
            "md5": hashlib.md5(dataset_path.read_bytes()).hexdigest(),
            "subjects": len(unique_subjects),
            "windows": len(windows),
            "sensor_positions": list(sensor_positions),
            "fog_windows": dict(Counter(map(str, y_fog))),
            "direction_windows": dict(Counter(map(str, y_direction))),
        },
        "validation": {
            "method": f"{n_splits}-fold stratified subject-group cross-validation",
            "random_state": random_state,
            "fold_subjects": fold_subjects,
            "fold_fog_thresholds": fold_fog_thresholds,
            "deployment_fog_threshold": float(np.median(fold_fog_thresholds)),
        },
        "fog": {
            **_jsonable_report(y_fog, fog_prediction, [0, 1]),
            "roc_auc": float(roc_auc_score(y_fog, fog_probability)),
        },
        "direction": _jsonable_report(
            y_direction[direction_trainable],
            direction_prediction[direction_trainable],
            direction_labels,
        ),
        "context": _jsonable_report(
            y_context,
            direction_prediction,
            ["straight", "left", "right", "other"],
        ),
        "four_state": _jsonable_report(
            true_state[valid_state],
            predicted_state[valid_state],
            ["straight", "left", "right", "fog"],
        ),
    }

    fog_model, direction_model = create_baseline_classifiers(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    fog_model.fit(fog_x, y_fog)
    direction_model.fit(x, y_context)
    model = FogStarMultiTaskModel(
        fog_classifier=fog_model,
        direction_classifier=direction_model,
        channel_names=windows[0].channel_names,
        feature_names=feature_names,
        fog_feature_indices=fog_feature_indices,
        fog_threshold=float(np.median(fold_fog_thresholds)),
    )
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/fogstar/sensor_data.csv"),
    )
    parser.add_argument("--sensor", action="append", dest="sensors")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=5412)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/fogstar_multitask_baseline.joblib"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/fogstar_multitask_baseline.json"),
    )
    args = parser.parse_args()
    sensors = tuple(args.sensors or ("ankleL", "back"))

    model, report = train_and_evaluate(
        args.data,
        sensor_positions=sensors,
        n_splits=args.folds,
        n_estimators=args.trees,
        random_state=args.random_state,
    )
    model.save(args.model_output)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value["macro_f1"] for key, value in report.items() if key in {"fog", "direction", "four_state"}}, indent=2))
    print(f"Wrote model to {args.model_output}")
    print(f"Wrote report to {args.report_output}")


if __name__ == "__main__":
    main()
