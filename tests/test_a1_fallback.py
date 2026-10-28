"""No zero-filled sensors, no score-based switching, no future-primary leakage."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import FogStarWindow
from fogguard.modeling.a1_fallback import (
    AnkleFallbackFoG,
    ankle_matrix,
    route_available_predictions,
)
from fogguard.modeling.a1_fog import A1FogModel


def predictions(ends):
    return pd.DataFrame({"subject_id": 1, "session_id": 1, "task_id": 1,
                         "window_start_s": np.asarray(ends) - 119 / 60, "window_end_s": ends,
                         "fog_probability": 0.7, "fog_threshold": 0.3, "is_fog": True})


def test_primary_rows_unchanged_and_ignored_ankle_does_not_extend_freshness():
    primary, ankle = predictions([2, 2.5, 4]), predictions([2, 2.25, 2.75, 3, 3.5, 4])
    result = route_available_predictions(primary, ankle)
    assert result.window_end_s.tolist() == [2, 2.5, 3, 3.5, 4]
    kept = result.loc[result.sensor_route.eq("primary_ankle_back"), primary.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, primary)
    primary["fog_probability"] = 0.01
    primary["is_fog"] = False
    assert route_available_predictions(primary, ankle).window_end_s.tolist() == [2, 2.5, 3, 3.5, 4]


def test_future_primary_cannot_remove_past_fallback_and_truth_is_ignored():
    primary, ankle = predictions([4]), predictions([2, 2.5, 3, 3.5, 4])
    before = route_available_predictions(primary.iloc[:0], ankle.iloc[:3])
    after = route_available_predictions(primary, ankle).iloc[:3]
    pd.testing.assert_frame_equal(before, after)
    primary["truth_fog_label"], ankle["truth_fog_label"] = 0, 1
    assert route_available_predictions(primary, ankle).sensor_route.tolist() == ["fallback_ankle"] * 4 + ["primary_ankle_back"]
    with pytest.raises(ValueError, match="Duplicate"):
        route_available_predictions(pd.concat([primary, primary]), ankle)


def test_csv_timestamp_roundtrip_noise_is_same_sample_not_an_extra_fallback():
    primary, ankle = predictions([2 + 1e-10]), predictions([2])
    result = route_available_predictions(primary, ankle)
    assert len(result) == 1 and result.sensor_route.iloc[0] == "primary_ankle_back"
    assert result.window_end_s.iloc[0] == primary.window_end_s.iloc[0]


def test_ankle_model_has_140_features_preserves_history_and_roundtrips(tmp_path):
    names = tuple(f"ankleL_{s}_{a}" for s in ("acc", "gyro") for a in "xyz")
    rng = np.random.default_rng(1)
    ws = [FogStarWindow(rng.normal(size=(120, 6)), names, 60, i * 0.5, i * 0.5 + 119 / 60,
                         "1", "1", "1") for i in range(10)]
    values, schema, links = ankle_matrix(ws)
    assert values.shape == (10, 140) and links[8].tolist() == [4, 0]
    changed = [replace(w, fog_label=1, fog_fraction=1, activity_label=7) for w in ws]
    np.testing.assert_array_equal(values, ankle_matrix(changed)[0])
    forest = RandomForestClassifier(n_estimators=4, random_state=1).fit(values, np.arange(10) % 2)
    model = AnkleFallbackFoG(forest, "ankle_context_v1", names, schema, 0.3)
    path = tmp_path / "ankle.joblib"
    model.save(path)
    loaded = A1FogModel.load(path)
    stream = loaded.new_stream()
    pd.testing.assert_frame_equal(loaded.predict_windows(ws),
                                   pd.concat([stream.push(w) for w in ws], ignore_index=True))
