"""Small metric helpers with JSON-serializable outputs."""

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
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
    specificity_denominator = true_negatives + false_positives
    specificity = (
        true_negatives / specificity_denominator if specificity_denominator else 0.0
    )
    sensitivity_denominator = true_positives + false_negatives
    sensitivity = (
        true_positives / sensitivity_denominator if sensitivity_denominator else 0.0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
    }
