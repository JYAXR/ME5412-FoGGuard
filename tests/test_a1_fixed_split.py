"""Disjoint roles, train-only perturbation scales and frozen model inference."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import FogStarWindow, sensor_columns
from fogguard.evaluation.a1_fixed_split import (
    check_split,
    choose_threshold,
    perturb_sensors,
    role_frame,
    training_noise_statistics,
)
from fogguard.modeling.a1_fixed_split import FixedSplitFoG, feature_bank
from fogguard.modeling.a1_fog import A1FogModel


def frame():
    channels = sensor_columns(("ankleL", "back"))
    result = pd.DataFrame(np.random.default_rng(1).normal(size=(720, 12)), columns=channels)
    result["subjectID"], result["sessionID"], result["taskID"] = 1, 1, 1
    result["timestamp"] = np.arange(len(result)) / 60
    result["fog"], result["activity"] = 1, 7
    return result


def test_participant_split_never_overlaps_or_silently_omits_people():
    split = {"train": [1], "validation": [2], "test": [3]}
    check_split(split, [1, 2, 3])
    bad = dict(split, test=[1])
    with pytest.raises(ValueError, match="overlap"):
        check_split(bad, [1, 2, 3])
    with pytest.raises(ValueError, match="incomplete"):
        check_split(split, [1, 2, 3, 4])
    raw = pd.concat([frame().assign(subjectID=p) for p in [1, 2, 3]], ignore_index=True)
    assert set(role_frame(raw, split, "validation").subjectID) == {2}


def test_noise_statistics_reject_validation_or_test_people():
    raw = frame()
    channels = sensor_columns(("ankleL", "back"))
    statistics = training_noise_statistics(raw, channels, [1])
    assert statistics[channels[0]] == pytest.approx(raw[channels[0]].std(ddof=0))
    with pytest.raises(ValueError, match="TRAIN"):
        training_noise_statistics(raw.assign(subjectID=2), channels, [1])


@pytest.mark.parametrize("scenario", ["clean", "noise_005", "noise_015", "rotation_15deg", "gyro_bias_3dps", "back_dropout_1s", "ankle_dropout_1s"])
def test_stress_preserves_times_and_truth_and_does_not_read_labels(scenario):
    raw = frame()
    columns = sensor_columns(("ankleL", "back"))
    scale = dict.fromkeys(columns, 1.0)
    result = perturb_sensors(raw, scenario, scale, 1)
    pd.testing.assert_frame_equal(result[["timestamp", "fog", "activity"]], raw[["timestamp", "fog", "activity"]])
    changed = perturb_sensors(raw.assign(fog=0, activity=1), scenario, scale, 1)
    np.testing.assert_array_equal(result[list(columns)], changed[list(columns)])
    if scenario == "rotation_15deg":
        for position in ("ankleL", "back"):
            for sensor in ("acc", "gyro"):
                cols = [f"{position}_{sensor}_{a}" for a in "xyz"]
                np.testing.assert_allclose(np.linalg.norm(result[cols], axis=1), np.linalg.norm(raw[cols], axis=1))
    if "dropout" in scenario:
        assert result.isna().any(axis=1).sum() == 60


def test_threshold_objective_uses_supplied_validation_scores():
    cfg = {"threshold_grid": {"start": 0.1, "stop": 0.9, "count": 9}}
    result = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.75, 0.9], cfg)
    assert result["selected"]["fog_f1"] == 1
    assert result["selected"]["threshold"] == pytest.approx(0.7)


def test_fixed_model_roundtrip_retains_fit_people_and_is_label_independent(tmp_path):
    raw = frame()
    names = sensor_columns(("ankleL", "back"))
    windows = [FogStarWindow(raw.loc[i * 30:i * 30 + 119, names].to_numpy(), names, 60,
        i * 0.5, i * 0.5 + 119 / 60, "1", "1", "1") for i in range(10)]
    values, schema = feature_bank(windows)["tabular"]
    forest = RandomForestClassifier(n_estimators=4, random_state=1).fit(values, np.arange(10) % 2)
    model = FixedSplitFoG(forest, "rf_2s", names, schema, 0.3, (1,), (2,))
    path = tmp_path / "fixed.joblib"
    model.save(path)
    loaded = A1FogModel.load(path)
    assert loaded.fit_participants == (1,) and loaded.threshold_participants == (2,)
    before = loaded.predict_windows(windows)
    changed = [replace(w, fog_label=1, activity_label=7) for w in windows]
    pd.testing.assert_frame_equal(before, loaded.predict_windows(changed))
    raw = raw.drop(columns=["subjectID", "sessionID", "taskID"])
    assert loaded.predict_frame(raw).subject_id.eq("new").all()
