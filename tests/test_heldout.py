"""Scoring conventions fixed on synthetic data before opening TEST results."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fogguard.evaluation.heldout import (
    align_direction_truth,
    assert_split,
    at_source_arrivals,
    attach_fog_truth,
    binary_report,
    direction_report,
    validate_fog_membership,
    verify_lock,
)
from fogguard.perception import PerceptionConfig


def grid():
    return pd.DataFrame({"subject_id": [1, 1, 1], "session_id": 1, "task_id": 1,
                         "window_start_s": [0, 0.5, 1], "window_end_s": [2, 2.5, 3],
                         "available_at_s": [2, 2.5, 3]})


def test_binary_report_handles_errors_and_reports_negative_baseline():
    frame = pd.DataFrame({"truth_fog_label": [0, 0, 1, 1],
                          "is_fog": [False, True, False, True],
                          "fog_probability": [0.1, 0.9, 0.2, 0.8]})
    result = binary_report(frame)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == result["precision"] == result["recall"] == 0.5
    assert result["always_non_fog_accuracy"] == 0.5


def test_known_direction_rejection_counts_as_error_and_coverage_separate():
    frame = grid().assign(input_valid=True, target=["straight", "left", "right"],
                          direction=["straight", "unknown", "right"])
    result = direction_report(frame)["primary_known_targets"]
    assert result["accuracy"] == result["prediction_coverage"] == 2 / 3
    assert result["accepted_accuracy"] == 1
    assert result["per_class"]["left"]["recall"] == 0
    assert result["confusion_matrix"][1][3] == 1


def test_invalid_windows_and_unknown_targets_have_explicit_denominators():
    frame = grid().assign(input_valid=[True, True, False], target=["left", "unknown", "right"],
                          direction=["left", "unknown", "unknown"])
    result = direction_report(frame)
    assert result["primary_known_targets"]["windows"] == 1
    assert result["full_stream_including_unknown"]["windows"] == 2
    assert result["missing_sensor_windows"] == result["unknown_target_windows"] == 1


def test_empty_known_direction_targets_are_not_zero_accuracy():
    frame = grid().assign(input_valid=True, target="unknown", direction="unknown")
    assert direction_report(frame)["primary_known_targets"]["accuracy"] is None


def test_truth_alignment_requires_exact_record_and_window_identity():
    runtime = grid()
    meta = runtime.assign(input_valid=True, target=["left", "right", "straight"])
    runtime["input_valid"] = True
    result = align_direction_truth(runtime, meta.iloc[::-1])
    assert result.target.tolist() == ["left", "right", "straight"]
    meta.loc[0, "task_id"] = 2
    with pytest.raises(ValueError, match="grids differ"):
        align_direction_truth(runtime, meta)


def test_fog_truth_attachment_never_uses_row_order():
    runtime = grid()
    windows = [SimpleNamespace(subject_id="1", session_id="1", task_id="1",
                               start_s=s, end_s=e, fog_label=y, fog_fraction=float(y))
               for s, e, y in [(1, 3, 1), (0.5, 2.5, 0), (0, 2, 1)]]
    assert attach_fog_truth(runtime, windows).truth_fog_label.tolist() == [1, 0, 1]


def test_scoring_never_counts_extra_timer_or_expiry_ticks_as_windows():
    source = grid()
    perception = pd.concat([source, source.iloc[:1].assign(available_at_s=2.2)])
    sampled = at_source_arrivals(perception, source)
    assert sampled.available_at_s.tolist() == [2, 2.5, 3]
    with pytest.raises(ValueError, match="Missing fused decision"):
        at_source_arrivals(perception.iloc[:1], source)


def test_test_membership_cannot_enter_fitting_or_threshold_selection():
    split = {"train": [1, 2], "validation": [3], "test": [4]}
    validate_fog_membership(SimpleNamespace(fit_participants=[1, 2], threshold_participants=[3]), split)
    for fit, threshold in [([1, 2, 4], [3]), ([1, 2], [4])]:
        with pytest.raises(ValueError):
            validate_fog_membership(SimpleNamespace(fit_participants=fit,
                                                    threshold_participants=threshold), split)
    with pytest.raises(ValueError):
        assert_split({"train": [1], "validation": [2], "test": [1]})


def test_changed_model_source_or_config_fails_locked_evaluation():
    lock = {"schema": "fogguard.a_group.test_lock.v1", "files_sha256": {"model": "original"},
            "perception_config": asdict(PerceptionConfig())}
    verify_lock(lock, {"model": "original"})
    with pytest.raises(ValueError, match="changed"):
        verify_lock(lock, {"model": "changed"})
    lock["perception_config"]["source_max_hold_s"] = 0.6
    with pytest.raises(ValueError, match="configuration differs"):
        verify_lock(lock, {"model": "original"})


def test_source_arrival_comparison_does_not_fill_missing_predictions():
    source = grid()
    perception = source.copy()
    perception["turn"] = ["unknown", "left", "unknown"]
    selected = at_source_arrivals(perception, source)
    np.testing.assert_array_equal(selected.turn, ["unknown", "left", "unknown"])
