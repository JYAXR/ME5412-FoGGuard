"""Evaluate a FoG detector against window-level labels."""

from fogguard.evaluation.metrics import classification_metrics
from fogguard.interfaces import DatasetReplay, FoGDetector


def evaluate_fog_detector(
    replay: DatasetReplay,
    detector: FoGDetector,
) -> dict[str, float | int]:
    """Return classification metrics and window counts for a labelled replay."""
    y_true: list[int] = []
    y_pred: list[int] = []

    for window in replay.windows():
        label = window.metadata.get("fog_label")
        if label not in {"0", "1"}:
            raise ValueError(
                "FoG evaluation requires every window to have a binary "
                "metadata['fog_label']"
            )
        y_true.append(int(label))
        y_pred.append(int(detector.predict(window).is_fog))

    if not y_true:
        raise ValueError("FoG evaluation requires at least one labelled window")

    return {
        **classification_metrics(y_true, y_pred),
        "n_windows": len(y_true),
        "n_true_fog": sum(y_true),
        "n_predicted_fog": sum(y_pred),
    }
