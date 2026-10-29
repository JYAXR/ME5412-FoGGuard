"""Regression checks for leakage, missing records, and preturn causality."""

import numpy as np
import pandas as pd
import pytest

from fogguard.data.fogstar import sensor_columns
from fogguard.evaluation.readiness import (
    audit_records,
    frozen_folds,
    preturn_candidates,
    split_runtime_truth,
)


def raw_record():
    data = pd.DataFrame({name: np.zeros(300) for name in sensor_columns(
        ("ankleL", "ankleR", "back"))})
    data["timestamp"] = np.arange(300) / 60
    data["subjectID"], data["sessionID"], data["taskID"] = 1, 1, 1
    data["fog"] = 0
    data["activity"] = np.where(data.timestamp < 4, 1, 7)
    return data


def test_preturn_history_is_before_decision_and_independent_of_future_signals():
    raw = raw_record()
    result = preturn_candidates(raw)
    assert result.eligible.all()
    assert (result.input_end_s <= result.decision_time_s + 1e-8).all()
    assert np.allclose(result.onset_s - result.decision_time_s, result.horizon_s)
    changed = raw.copy()
    changed.loc[changed.timestamp > 3.5, "back_acc_x"] = np.nan
    pd.testing.assert_frame_equal(result, preturn_candidates(changed))


def test_preturn_rejects_missing_history_and_signal_gap():
    raw = raw_record()
    raw.loc[raw.timestamp < 3.6, "back_acc_x"] = np.nan
    assert set(preturn_candidates(raw).reason) == {"missing_sensor"}
    raw = raw_record().drop(index=120)
    assert "signal_gap" in set(preturn_candidates(raw).reason)


def test_unpredicted_record_and_missing_bilateral_data_stay_visible():
    raw = raw_record()
    raw["ankleR_acc_x"] = np.nan
    empty = pd.DataFrame(columns=["subject_id", "session_id", "task_id"])
    result = audit_records(raw, empty).iloc[0]
    assert result.oof_windows == 0
    assert result.fog_baseline_has_2s_raw_run
    assert not result.bilateral_ankles_has_2s_raw_run


def test_frozen_folds_reject_leakage_and_duplicate_test_people():
    predictions = pd.DataFrame({"subject_id": [1, 2]})
    report = {"validation": {"fold_subjects": [
        {"train": [2], "test": [1]}, {"train": [1], "test": [2]}],
        "random_state": 5412, "method": "grouped"}}
    assert len(frozen_folds(report, predictions)["folds"]) == 2
    report["validation"]["fold_subjects"][1] = {"train": [2], "test": [1]}
    with pytest.raises(ValueError, match="leakage"):
        frozen_folds(report, predictions)


def test_runtime_never_receives_truth_columns():
    frame = pd.DataFrame({"subject_id": [1], "session_id": [1], "task_id": [1],
                          "window_start_s": [0.0], "window_end_s": [2.0],
                          "fog_probability": [0.8], "truth_fog_label": [1]})
    runtime, truth = split_runtime_truth(frame)
    assert "truth_fog_label" not in runtime
    assert "fog_probability" not in truth
    assert truth.truth_fog_label.iloc[0] == 1
