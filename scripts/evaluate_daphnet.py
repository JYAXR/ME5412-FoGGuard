"""Participant-level cross-validation on the public Daphnet FoG dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from fogguard.data import DAPHNET_SAMPLE_RATE_HZ, WindowReplay, load_daphnet_dataset
from fogguard.detection import FreezeIndexDetector
from fogguard.evaluation import classification_metrics, evaluate_fog_events
from fogguard.types import SensorWindow


def _scores(windows: Iterable[SensorWindow]) -> tuple[list[int], list[float]]:
    detector = FreezeIndexDetector()
    labels: list[int] = []
    probabilities: list[float] = []
    for window in windows:
        labels.append(int(window.metadata["fog_label"]))
        probabilities.append(detector.predict(window).probability)
    return labels, probabilities


def _threshold_report(
    labels: list[int], probabilities: list[float], threshold: float
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=bool)
    predictions = np.asarray(probabilities, dtype=float) >= threshold
    true_positives = int(np.count_nonzero(truth & predictions))
    true_negatives = int(np.count_nonzero(~truth & ~predictions))
    false_positives = int(np.count_nonzero(~truth & predictions))
    false_negatives = int(np.count_nonzero(truth & ~predictions))
    precision = true_positives / (true_positives + false_positives or 1)
    recall = true_positives / (true_positives + false_negatives or 1)
    specificity = true_negatives / (true_negatives + false_positives or 1)
    f1 = 2 * precision * recall / (precision + recall or 1)
    return {
        "threshold": threshold,
        "accuracy": (true_positives + true_negatives) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "n_windows": len(labels),
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _select_threshold(
    labels: list[int], probabilities: list[float], thresholds: Iterable[float]
) -> dict[str, float | int]:
    reports = [_threshold_report(labels, probabilities, float(value)) for value in thresholds]
    return max(
        reports,
        key=lambda report: (
            float(report["balanced_accuracy"]),
            float(report["f1"]),
            -int(report["false_positives"]),
            float(report["threshold"]),
        ),
    )


def evaluate_daphnet(
    root: Path,
    *,
    sensor_position: str = "ankle",
) -> dict[str, object]:
    by_subject = load_daphnet_dataset(root, sensor_position=sensor_position)
    subject_frames = dict(sorted(by_subject.items()))
    subject_scores: dict[str, tuple[list[int], list[float]]] = {}
    subject_replays: dict[str, list[WindowReplay]] = {}
    for subject, frames in subject_frames.items():
        replays = [
            WindowReplay(frame, sample_rate_hz=DAPHNET_SAMPLE_RATE_HZ)
            for frame in frames
        ]
        subject_replays[subject] = replays
        subject_scores[subject] = _scores(
            window for replay in replays for window in replay.windows()
        )

    thresholds = np.linspace(0.01, 0.99, 99)
    folds: list[dict[str, object]] = []
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_delays: list[float] = []
    total_events = 0
    detected_events = 0
    outside_positive_windows = 0
    total_duration_s = 0.0

    for held_out in subject_frames:
        train_labels = [
            label
            for subject, (labels, _) in subject_scores.items()
            if subject != held_out
            for label in labels
        ]
        train_probabilities = [
            probability
            for subject, (_, probabilities) in subject_scores.items()
            if subject != held_out
            for probability in probabilities
        ]
        selected = _select_threshold(train_labels, train_probabilities, thresholds)
        threshold = float(selected["threshold"])
        test_labels, test_probabilities = subject_scores[held_out]
        test_report = _threshold_report(test_labels, test_probabilities, threshold)
        test_predictions = [int(value >= threshold) for value in test_probabilities]
        all_labels.extend(test_labels)
        all_predictions.extend(test_predictions)

        subject_event_count = 0
        subject_detected_count = 0
        subject_outside_count = 0
        subject_delays: list[float] = []
        subject_duration_s = 0.0
        detector = FreezeIndexDetector(threshold=threshold)
        for replay in subject_replays[held_out]:
            event_report = evaluate_fog_events(replay, detector)
            subject_event_count += int(event_report["n_events"])
            subject_detected_count += int(event_report["n_detected_events"])
            subject_outside_count += int(
                event_report["n_positive_windows_outside_events"]
            )
            subject_delays.extend(
                float(event["delay_s"])
                for event in event_report["events"]
                if event["delay_s"] is not None
            )
            subject_duration_s += float(replay.frame["timestamp_s"].iloc[-1]) - float(
                replay.frame["timestamp_s"].iloc[0]
            )

        total_events += subject_event_count
        detected_events += subject_detected_count
        outside_positive_windows += subject_outside_count
        total_duration_s += subject_duration_s
        all_delays.extend(subject_delays)
        folds.append(
            {
                "held_out_subject": held_out,
                "selected_threshold": threshold,
                **test_report,
                "n_events": subject_event_count,
                "n_detected_events": subject_detected_count,
                "event_recall": (
                    subject_detected_count / subject_event_count
                    if subject_event_count
                    else None
                ),
                "mean_detection_delay_s": (
                    float(np.mean(subject_delays)) if subject_delays else None
                ),
                "false_positive_windows_per_minute": (
                    subject_outside_count / (subject_duration_s / 60.0)
                    if subject_duration_s
                    else None
                ),
            }
        )

    aggregate = {
        **classification_metrics(all_labels, all_predictions),
        "n_subjects": len(subject_frames),
        "n_windows": len(all_labels),
        "n_events": total_events,
        "n_detected_events": detected_events,
        "event_recall": detected_events / total_events if total_events else 0.0,
        "mean_detection_delay_s": float(np.mean(all_delays)) if all_delays else None,
        "false_positive_windows_per_minute": (
            outside_positive_windows / (total_duration_s / 60.0)
            if total_duration_s
            else None
        ),
        "median_selected_threshold": float(
            np.median([float(fold["selected_threshold"]) for fold in folds])
        ),
    }
    return {
        "dataset": "Daphnet Freezing of Gait",
        "license": "CC BY 4.0",
        "sample_rate_hz": DAPHNET_SAMPLE_RATE_HZ,
        "sensor_position": sensor_position,
        "validation": "leave-one-subject-out",
        "aggregate": aggregate,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("data/raw/daphnet"),
    )
    parser.add_argument(
        "--sensor-position",
        choices=("ankle", "thigh", "trunk"),
        default="ankle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/daphnet_loso_evaluation.json"),
    )
    args = parser.parse_args()
    report = evaluate_daphnet(args.root, sensor_position=args.sensor_position)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
