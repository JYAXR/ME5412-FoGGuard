"""Reproducible threshold sweep helpers for FoG probability scores."""

from collections.abc import Sequence

from fogguard.evaluation.metrics import classification_metrics
from fogguard.interfaces import DatasetReplay, FoGDetector

ThresholdReport = dict[str, float | int]


def sweep_fog_thresholds(
    replays: Sequence[DatasetReplay],
    detector: FoGDetector,
    thresholds: Sequence[float],
) -> list[ThresholdReport]:
    """Evaluate candidate thresholds on the combined labelled replay windows."""
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be between 0 and 1")

    y_true: list[int] = []
    probabilities: list[float] = []
    for replay in replays:
        for window in replay.windows():
            label = window.metadata.get("fog_label")
            if label not in {"0", "1"}:
                raise ValueError(
                    "FoG calibration requires every window to have a binary "
                    "metadata['fog_label']"
                )
            probability = detector.predict(window).probability
            if not 0.0 <= probability <= 1.0:
                raise ValueError("FoG detector probabilities must be between 0 and 1")
            y_true.append(int(label))
            probabilities.append(probability)

    if not y_true:
        raise ValueError("FoG calibration requires at least one labelled window")

    reports: list[ThresholdReport] = []
    for threshold in thresholds:
        y_pred = [int(probability >= threshold) for probability in probabilities]
        true_positives = sum(
            truth == 1 and prediction == 1
            for truth, prediction in zip(y_true, y_pred, strict=True)
        )
        true_negatives = sum(
            truth == 0 and prediction == 0
            for truth, prediction in zip(y_true, y_pred, strict=True)
        )
        false_positives = sum(
            truth == 0 and prediction == 1
            for truth, prediction in zip(y_true, y_pred, strict=True)
        )
        false_negatives = sum(
            truth == 1 and prediction == 0
            for truth, prediction in zip(y_true, y_pred, strict=True)
        )
        reports.append(
            {
                "threshold": float(threshold),
                **classification_metrics(y_true, y_pred),
                "n_windows": len(y_true),
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
        )
    return reports


def select_fog_threshold(
    reports: Sequence[ThresholdReport],
    *,
    max_false_positives: int = 0,
) -> ThresholdReport:
    """Select by false-positive constraint, then F1, recall, and threshold."""
    if max_false_positives < 0:
        raise ValueError("max_false_positives must not be negative")

    eligible = [
        report
        for report in reports
        if int(report["false_positives"]) <= max_false_positives
    ]
    if not eligible:
        raise ValueError("No threshold satisfies the false-positive constraint")

    selected = max(
        eligible,
        key=lambda report: (
            float(report["f1"]),
            float(report["recall"]),
            float(report["threshold"]),
        ),
    )
    return dict(selected)
