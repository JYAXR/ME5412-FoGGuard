"""Binary preturn feature causality and threshold/evaluation safeguards."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import sensor_columns
from fogguard.turning.learned import build_dataset
from fogguard.turning.preturn_binary import BinaryPreturnModel, dataset, short_features
from scripts.train_a2_preturn_binary import choose_threshold, evaluate


def record(n=600):
    frame = pd.DataFrame({"timestamp": np.arange(n) / 60, "subjectID": 1,
                          "sessionID": 1, "taskID": 1, "activity": 1})
    for c in sensor_columns(("ankleL", "ankleR", "back")):
        frame[c] = 0.0
    for position in ("ankleL", "ankleR", "back"):
        frame[f"{position}_acc_x"] = -1.0
    frame.loc[300:359, "activity"] = 7
    return frame


def test_short_features_retain_recent_changes_and_bilateral_information():
    channels = sensor_columns(("ankleL", "ankleR", "back"))
    samples = np.zeros((120, 18))
    for position in ("ankleL", "ankleR", "back"):
        samples[:, channels.index(f"{position}_acc_x")] = -1
    samples[-15:, channels.index("back_gyro_x")] = 20
    a = short_features(samples, channels)
    samples[-15:, channels.index("back_gyro_x")] = 0
    samples[:15, channels.index("back_gyro_x")] = 20
    b = short_features(samples, channels)
    assert a.shape == b.shape == (420,)
    assert not np.allclose(a, b)
    samples[:, channels.index("ankleR_gyro_x")] = 30
    assert not np.allclose(b, short_features(samples, channels))
    samples[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        short_features(samples, channels)


def test_targets_match_previous_task_and_features_ignore_all_labels():
    frame = record()
    features, meta = dataset(frame, "left_back_short")
    _, old_meta = build_dataset(frame, sensor_columns(("ankleL", "back")), 0.5)
    assert meta["target_0.5"].equals(old_meta.target)
    assert (meta.loc[meta["target_0.5"] == "left", "window_end_s"] <= 4.5).all()
    frame.activity = 3
    changed, changed_meta = dataset(frame, "left_back_short")
    np.testing.assert_array_equal(features, changed)
    assert not meta["target_0.5"].equals(changed_meta["target_0.5"])


def test_binary_runtime_ignores_truth_future_and_missing_channels(tmp_path):
    frame = record()
    x, _ = dataset(frame, "bilateral_back_short")
    # Tiny models are only fixtures for schema and source isolation.
    trigger = RandomForestClassifier(n_estimators=5, random_state=1).fit(
        [x[0], x[0] + 1], [0, 1])
    direction = RandomForestClassifier(n_estimators=5, random_state=1).fit(
        [x[0], x[0] + 1], ["left", "right"])
    model = BinaryPreturnModel(trigger, direction, "bilateral_back_short", 0.5, 0.0)
    path = tmp_path / "binary.joblib"
    model.save(path)
    model = BinaryPreturnModel.load(path)
    before = model.predict_frame(frame.drop(columns="activity"))
    frame.activity = 2
    pd.testing.assert_frame_equal(before, model.predict_frame(frame))
    frame.loc[120:, "back_gyro_x"] = 999
    pd.testing.assert_series_equal(before.iloc[0], model.predict_frame(frame).iloc[0])
    frame.loc[0, "ankleR_gyro_x"] = np.nan
    output = model.predict_frame(frame)
    assert output.iloc[0].task_prediction == "unknown"
    assert output.iloc[0].direction == "unknown"
    assert not output.iloc[0].turn_expected
    assert np.isnan(output.iloc[0].turn_score)
    assert output.iloc[0].available_at_s == pytest.approx(119 / 60)
    assert not any(c in output for c in ["activity", "target", "fog"])


def test_threshold_selection_rejects_overactive_alarms():
    meta = pd.DataFrame({"subject_id": 1, "session_id": 1, "task_id": 1,
                         "window_end_s": np.arange(10) * 0.5, "input_valid": True,
                         "target": ["no_turn"] * 8 + ["left", "right"],
                         "target_onset_s": [np.nan] * 8 + [5.0, 5.5]})
    scores = np.array([0.2] * 8 + [0.9, 0.9])
    config = {"thresholds": [0.1, 0.8], "maximum_oof_false_alarms_per_minute": 1,
              "maximum_oof_negative_alarm_fraction": 0.1}
    chosen, search = choose_threshold(meta, scores, config)
    assert not search[0]["feasible"]
    assert chosen["threshold"] == 0.8
    assert chosen["metrics"]["walking_binary"]["f1"] == 1.0
    assert chosen["metrics"]["full_stream"]["detected_events"] == 2
    # Nonwalking/ambiguous contexts also count in full-stream false alarms.
    meta.loc[0, "target"] = "unknown"
    report = evaluate(meta, np.ones(10), 0.5)
    assert report["full_stream"]["false_alarm_windows"] == 8
    assert report["full_stream"]["negative_alarm_fraction"] == 1


def test_no_positive_subject_is_reported_without_inventing_auc():
    meta = pd.DataFrame({"subject_id": 1, "session_id": 1, "task_id": 1,
                         "window_end_s": np.arange(10) * 0.5, "input_valid": True,
                         "target": "no_turn", "target_onset_s": np.nan})
    result = evaluate(meta, np.zeros(10), 0.5)
    assert result["walking_binary"]["AUROC"] is None
    assert result["full_stream"]["event_count"] == 0
    assert result["full_stream"]["event_recall"] is None
