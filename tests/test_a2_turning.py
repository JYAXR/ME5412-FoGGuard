"""A2 timing, source isolation and refusal behavior on controlled records."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.turning.learned import (
    DirectionModel,
    build_dataset,
    current_target,
    default_channels,
    extract_features,
    iter_raw_windows,
)


def record(n=600):
    frame = pd.DataFrame({"timestamp": np.arange(n) / 60, "subjectID": 1,
                          "sessionID": 1, "taskID": 1, "activity": 1})
    for channel in default_channels():
        frame[channel] = 0.0
    frame["back_acc_x"] = -1.0
    frame["ankleL_acc_x"] = -1.0
    return frame


def model():
    samples = np.zeros((120, 12))
    a = extract_features(samples, default_channels())
    b = extract_features(samples + 1, default_channels())
    classifier = RandomForestClassifier(n_estimators=5, random_state=5412).fit(
        [a, b], ["straight", "left"])
    return DirectionModel(classifier, default_channels())


def test_current_target_rejects_mixed_other_and_old_direction():
    assert current_target(np.full(120, 1)) == "straight"
    assert current_target(np.full(120, 6)) == "right"
    assert current_target(np.full(120, 7)) == "left"
    assert current_target(np.full(120, 3)) == "unknown"
    assert current_target(np.r_[np.ones(60), np.full(60, 7)]) == "unknown"
    assert current_target(np.r_[np.ones(119), 6]) == "unknown"


@pytest.mark.parametrize("horizon", [0.5, 1.0])
def test_preturn_inputs_end_before_onset_and_include_negatives(horizon):
    frame = record()
    frame.loc[300:359, "activity"] = 7
    _, meta = build_dataset(frame, default_channels(), horizon)
    positives = meta.loc[meta.target == "left"]
    assert not positives.empty
    lead = positives.target_onset_s - positives.window_end_s
    assert (lead >= horizon - 1e-8).all()
    assert (lead < horizon + 0.5).all()
    for row in positives.itertuples():
        history = frame.loc[frame.timestamp.between(row.window_start_s, row.window_end_s)]
        assert history.activity.eq(1).all()
    assert (meta.target == "no_turn").any()
    assert meta.target.isna().any()  # Right-censored future is never a negative.


def test_nonturn_transition_is_negative_and_future_gap_is_unscorable():
    frame = record()
    frame.loc[300:, "activity"] = 3
    _, meta = build_dataset(frame, default_channels(), 0.5)
    assert not meta.target.isin(["left", "right"]).any()
    row = meta.iloc[5]
    frame.loc[frame.timestamp > row.window_end_s, "timestamp"] += 1
    _, changed = build_dataset(frame, default_channels(), 0.5)
    assert pd.isna(changed.iloc[5].target)


def test_features_preserve_signed_rotation_and_reject_missing():
    samples = np.zeros((120, 12))
    samples[:, default_channels().index("back_acc_x")] = -1
    samples[:, default_channels().index("back_gyro_x")] = 20
    left = extract_features(samples, default_channels())
    samples[:, default_channels().index("back_gyro_x")] = -20
    assert not np.allclose(left, extract_features(samples, default_channels()))
    samples[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_features(samples, default_channels())


def test_runtime_has_no_annotation_dependency_and_no_future_dependency():
    predictor = model()
    frame = record()
    unlabelled = frame.drop(columns="activity")
    first = predictor.predict_frame(unlabelled)
    frame.activity = np.arange(len(frame)) % 8
    frame["fog"] = 1
    pd.testing.assert_frame_equal(first, predictor.predict_frame(frame))
    frame.loc[120:, list(default_channels())] = 999
    after = predictor.predict_frame(frame)
    pd.testing.assert_series_equal(first.iloc[0], after.iloc[0])
    assert first.iloc[0].available_at_s == pytest.approx(119 / 60)
    assert not any("truth" in c or c in ["activity", "fog", "target"] for c in first)


def test_missing_gaps_record_switch_and_time_reversal():
    predictor = model()
    frame = record(240)
    frame.loc[0, "back_gyro_x"] = np.nan
    out = predictor.predict_frame(frame)
    assert out.iloc[0].direction == "unknown"
    assert out.iloc[0].direction_confidence == 0
    frame = record(240)
    frame.loc[60:, "timestamp"] += 1
    assert not predictor.predict_frame(frame).iloc[0].input_valid
    frame = record(238)
    frame.loc[119:, "taskID"] = 2
    assert list(iter_raw_windows(frame, default_channels())) == []
    frame = record(240)
    frame.loc[40, "timestamp"] = 0
    with pytest.raises(ValueError, match="strictly increase"):
        predictor.predict_frame(frame)


def test_model_roundtrip_and_rejection(tmp_path):
    predictor = model()
    predictor.rejection_threshold = 1.1
    path = tmp_path / "model.joblib"
    predictor.save(path)
    out = DirectionModel.load(path).predict_frame(record())
    assert out.direction.eq("unknown").all()


def test_forecast_no_turn_is_not_current_straight():
    predictor = model()
    features = extract_features(np.zeros((120, 12)), default_channels())
    predictor.classifier.fit([features], ["no_turn"])
    predictor.horizon_s = 0.5
    out = predictor.predict_frame(record())
    assert out.task_prediction.eq("no_turn").all()
    assert out.direction.eq("unknown").all()
    assert out.direction_confidence.eq(0.0).all()
    assert out.direction_task.eq("preturn").all()


def test_training_features_are_independent_of_all_activity_labels():
    frame = record()
    before, meta = build_dataset(frame, default_channels(), 0.5)
    frame.activity = 7
    after, changed = build_dataset(frame, default_channels(), 0.5)
    np.testing.assert_array_equal(before, after)
    assert not meta.target.equals(changed.target)


def test_selected_feature_schema_is_used_in_runtime(tmp_path):
    predictor = model()
    features = extract_features(np.zeros((120, 12)), default_channels())
    predictor.feature_indices = tuple(range(len(features) - 12, len(features)))
    predictor.classifier.fit([features[list(predictor.feature_indices)]], ["left"])
    path = tmp_path / "rotation.joblib"
    predictor.save(path)
    assert DirectionModel.load(path).predict_frame(record()).direction.eq("left").all()
