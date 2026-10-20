from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fogguard.data import FOGSTAR_SAMPLE_RATE_HZ, iter_fogstar_windows, load_fogstar_frame
from fogguard.modeling import (
    FogStarMultiTaskModel,
    build_feature_matrix,
    create_baseline_classifiers,
)


def _frame(seconds: float = 3.0) -> pd.DataFrame:
    n = round(seconds * FOGSTAR_SAMPLE_RATE_HZ)
    timestamp = np.arange(n) / FOGSTAR_SAMPLE_RATE_HZ
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "activity": np.where(timestamp < 1.1, 1, 6),
            "fog": np.where(timestamp >= 2.0, 1, 0),
            "fog_severity": 0,
            "subjectID": 1,
            "sessionID": 1,
            "taskID": 1,
        }
    )
    for position in ("ankleL", "back"):
        for sensor in ("acc", "gyro"):
            for axis_index, axis in enumerate("xyz", 1):
                frame[f"{position}_{sensor}_{axis}"] = np.sin(
                    2 * np.pi * axis_index * timestamp
                )
    return frame


def test_fogstar_windows_keep_independent_fog_and_direction_targets() -> None:
    windows = list(iter_fogstar_windows(_frame()))

    assert len(windows) == 3
    assert windows[0].direction_label == "straight"
    assert windows[-1].direction_label == "right"
    assert windows[-1].fog_label == 1
    assert windows[-1].samples.shape == (120, 12)


def test_fogstar_short_gaps_are_interpolated_but_long_gaps_remove_windows() -> None:
    short = _frame()
    short.loc[30:32, "ankleL_acc_x"] = np.nan
    assert len(list(iter_fogstar_windows(short))) == 3

    long = _frame()
    long.loc[50:70, "ankleL_acc_x"] = np.nan
    assert not list(iter_fogstar_windows(long))


def test_fogstar_loader_validates_labels_and_features(tmp_path: Path) -> None:
    path = tmp_path / "fogstar.csv"
    frame = _frame()
    frame.to_csv(path, index=False)
    loaded = load_fogstar_frame(path)
    windows = list(iter_fogstar_windows(loaded))
    features, names = build_feature_matrix(windows)

    assert features.shape == (3, 156)
    assert len(names) == 156
    assert np.isfinite(features).all()

    frame.loc[0, "fog"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="fog labels"):
        load_fogstar_frame(path)


def test_fogstar_model_round_trip_and_prediction(tmp_path: Path) -> None:
    windows = list(iter_fogstar_windows(_frame()))
    features, names = build_feature_matrix(windows)
    fog_indices = tuple(
        index
        for index, name in enumerate(names)
        if not name.endswith(("__raw_mean", "__raw_median"))
    )
    fog_model, direction_model = create_baseline_classifiers(n_estimators=10)
    fog_model.fit(features[:, fog_indices], np.array([0, 0, 1]))
    direction_model.fit(features, np.array(["straight", "right", "other"]))
    model = FogStarMultiTaskModel(
        fog_classifier=fog_model,
        direction_classifier=direction_model,
        channel_names=windows[0].channel_names,
        feature_names=names,
        fog_feature_indices=fog_indices,
        fog_threshold=0.3,
    )
    path = tmp_path / "model.joblib"
    model.save(path)

    prediction = FogStarMultiTaskModel.load(path).predict_window(windows[-1])

    assert 0.0 <= prediction["fog_probability"] <= 1.0
    assert prediction["direction"] in {"straight", "right", "other"}
    assert prediction["state"] in {"fog", "straight", "right", "other"}
