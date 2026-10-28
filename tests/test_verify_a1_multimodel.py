"""The independent checker must reject wrong decisions and aggregate metrics."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

spec = importlib.util.spec_from_file_location("verify_a1_models", Path(__file__).resolve().parents[1]
                                            / "scripts/verify_a1_multimodel.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def example():
    frame = pd.DataFrame({"subject_id": [1, 1, 2, 2], "truth_fog_label": [0, 1, 0, 1],
                          "fog_probability": [0.1, 0.8, 0.6, 0.9], "fog_threshold": 0.5,
                          "is_fog": [False, True, True, True]})
    metrics = {"fog_f1": f1_score(frame.truth_fog_label, frame.is_fog),
               "auc": roc_auc_score(frame.truth_fog_label, frame.fog_probability),
               "average_precision": average_precision_score(frame.truth_fog_label,
                                                               frame.fog_probability)}
    return frame, metrics


def test_recomputed_metrics_and_tampered_boolean():
    frame, metrics = example()
    checker.verify_predictions(frame, metrics, {1: 0.5, 2: 0.5})
    frame.loc[0, "is_fog"] = True
    with pytest.raises(ValueError, match="decision mismatch"):
        checker.verify_predictions(frame, metrics, {1: 0.5, 2: 0.5})


def test_rejects_changed_metric_and_wrong_fold_threshold():
    frame, metrics = example()
    with pytest.raises(ValueError, match="Aggregate|aggregate"):
        checker.verify_predictions(frame, {**metrics, "fog_f1": 1.0}, {1: 0.5, 2: 0.5})
    with pytest.raises(ValueError, match="wrong participant"):
        checker.verify_predictions(frame, metrics, {1: 0.4, 2: 0.5})
