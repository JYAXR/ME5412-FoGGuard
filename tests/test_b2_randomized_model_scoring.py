"""Validate missing-output denominators, rejection and causal onset scoring."""

import numpy as np
import pandas as pd
import pytest

from fogguard.evaluation.randomized_model_scoring import (
    attach_to_truth,
    direction_metrics,
    event_metrics,
    fog_metrics,
)


def truth():
    return pd.DataFrame({"window_start_s": [0., 0.5, 1., 1.5],
                         "window_end_s": [2., 2.5, 3., 3.5],
                         "truth_fog_label": [1, 1, 0, 0],
                         "truth_direction": ["left", "right", "unknown", "straight"]})


def test_missing_predictions_are_unknown_not_false_negatives():
    pred = pd.DataFrame({"window_start_s": [0., 1., 1.5], "window_end_s": [2., 3., 3.5],
                         "is_fog": [True, True, False]})
    aligned, outside = attach_to_truth(truth(), pred, kind="fog")
    values = fog_metrics(aligned)
    assert outside == 0
    assert values["planned_windows"] == 4 and values["supported_windows"] == 3
    assert values["unsupported_positive_windows"] == 1
    assert (values["TP"], values["FP"], values["FN"], values["TN"]) == (1, 1, 0, 1)
    assert values["recall"] == 1 and values["precision"] == 0.5


def test_rejected_known_direction_counts_as_incorrect():
    pred = truth()[["window_start_s", "window_end_s"]].copy()
    pred["direction"] = ["left", "unknown", "unknown", "right"]
    pred["input_valid"] = [True, True, False, True]
    aligned, _ = attach_to_truth(truth(), pred, kind="direction")
    scores = direction_metrics(aligned)
    assert scores["supported_windows"] == 3 and scores["known_target_windows"] == 3
    assert scores["known_accuracy_including_rejection"] == pytest.approx(1/3)
    assert scores["known_accepted_coverage"] == pytest.approx(2/3)
    assert scores["accepted_only_accuracy"] == pytest.approx(0.5)


def test_unavailable_or_duplicated_outputs_never_become_normal():
    aligned, _ = attach_to_truth(truth(), None, kind="fog")
    assert fog_metrics(aligned)["recall"] is None
    assert fog_metrics(aligned)["unsupported_positive_windows"] == 2
    duplicates = pd.DataFrame({"window_start_s": [0., 0.], "window_end_s": [2., 2.],
                               "is_fog": [True, False]})
    with pytest.raises(ValueError, match="Duplicate prediction"):
        attach_to_truth(truth(), duplicates, kind="fog")
    with pytest.raises(ValueError, match="truth"):
        attach_to_truth(truth(), duplicates.assign(truth_fog_label=0), kind="fog")


def test_event_detection_does_not_backdate_a_two_second_window():
    samples = pd.DataFrame({"timestamp": np.arange(240) / 60})
    samples["fog"] = samples.timestamp.ge(1) & samples.timestamp.lt(2)
    predictions = pd.DataFrame({"window_end_s": [0.5, 1.5, 2.0],
                                "supported": [True, True, True],
                                "predicted": [False, True, False]})
    events = event_metrics(samples, predictions, 0, 4)
    assert len(events) == 1 and events[0]["covered"]
    assert events[0]["delay_s"] == pytest.approx(0.5)
    assert not events[0]["preexisting_alarm"]
    # No prediction is backfilled into the preceding input window.
    later = predictions.assign(predicted=[False, False, True])
    assert not event_metrics(samples, later, 0, 4)[0]["covered"]


def test_duplicate_timestamp_outside_window_does_not_make_all_windows_invalid():
    pytest.importorskip("torch")
    from scripts.evaluate_b2_randomized_models import valid_source_windows

    times = np.arange(300) / 60
    frame = pd.DataFrame({"timestamp": times, "ankle": 1.0})
    frame.loc[250, "timestamp"] = frame.loc[249, "timestamp"]
    output = pd.DataFrame({"window_start_s": [0.0], "window_end_s": [119/60],
                           "available_at_s": [119/60]})
    assert valid_source_windows(frame, output, ["ankle"]) == 0
    frame.loc[80, "timestamp"] = frame.loc[79, "timestamp"]
    assert valid_source_windows(frame, output, ["ankle"]) == 1
