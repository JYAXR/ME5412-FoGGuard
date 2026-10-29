"""Right-gap alignment, single-factor feature parity and label-free gyro runtime."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.evaluation.a1_bilateral import failure_pairs, match_left_indices, support_summary
from fogguard.modeling.a1_bilateral import (
    DUAL,
    LEFT,
    BilateralFoG,
    anchored_gyro_windows,
    left_view,
    matched_features,
)
from fogguard.modeling.a1_fog import a1_matrix


def signal(n=480):
    frame = pd.DataFrame(np.random.default_rng(4).normal(size=(n, 6)), columns=DUAL)
    frame["timestamp"] = np.arange(n) / 60
    frame["subjectID"], frame["sessionID"], frame["taskID"] = 1, 1, 1
    frame["fog"], frame["activity"] = 0, 3
    return frame


def test_right_missing_skips_original_left_grid_without_reanchoring():
    frame = signal()
    frame.loc[150, DUAL[-1]] = np.nan
    left = anchored_gyro_windows(frame, require_right=False)
    dual = anchored_gyro_windows(frame, require_right=True)
    starts = [round(w.start_s * 60) for w in dual]
    assert starts == [0, 30, 180, 210, 240, 270, 300, 330, 360]
    assert 151 not in starts and len(left) == 13
    np.testing.assert_array_equal(match_left_indices(left, dual), [0, 1, 6, 7, 8, 9, 10, 11, 12])
    assert all(np.isfinite(w.samples).all() for w in dual)


def test_left_missing_and_record_boundaries_reset_and_irregular_rate_rejects():
    frame = signal()
    frame.loc[150, LEFT[0]] = np.nan
    windows = anchored_gyro_windows(frame, require_right=True)
    assert any(round(w.start_s * 60) == 151 for w in windows)
    frame = signal(240)
    frame.loc[120:, "taskID"] = 2
    assert len(anchored_gyro_windows(frame, require_right=True)) == 2
    frame = signal()
    frame["timestamp"] *= 0.6
    with pytest.raises(ValueError, match="60Hz"):
        anchored_gyro_windows(frame, require_right=True)


def test_feature_prefix_uses_exactly_the_same_left_signal_and_no_labels():
    windows = anchored_gyro_windows(signal(180), require_right=True)
    bank = matched_features(windows)
    left, names = a1_matrix(left_view(windows), "baseline_2s_v1")
    np.testing.assert_allclose(left, bank["left_rf"][0], atol=1e-10, rtol=1e-12)
    assert names == bank["left_rf"][1] and left.shape == (3, 33)
    assert bank["bilateral_rf"][0].shape == (3, 66)
    changed = [replace(w, fog_label=1, activity_label=7) for w in windows]
    np.testing.assert_array_equal(bank["bilateral_rf"][0], matched_features(changed)["bilateral_rf"][0])


def test_failure_pair_requires_same_person_activity_pure_truth_and_disjoint_time():
    frame = pd.DataFrame({"subject_id": [1, 1, 1, 2], "session_id": [1] * 4, "task_id": [1] * 4,
                          "window_start_s": [0, 8, 1, 8], "truth_activity": [3] * 4,
                          "truth_fog_fraction": [0, 1, 1, 1], "is_fog": [True] * 4})
    values = np.zeros((4, 66))
    values[1, 0] = 1
    candidates, selected, summary = failure_pairs(frame, values)
    assert len(candidates) == len(selected) == 1
    assert selected[0]["negative_index"] == 0 and selected[0]["positive_index"] == 1
    assert summary[0]["pair_found"] and not summary[1]["pair_found"]
    empty = pd.DataFrame(columns=["subject_id", "window_end_s"])
    raw = pd.DataFrame({"subjectID": [1, 1], "sessionID": [1, 1], "taskID": [1, 1], "timestamp": [0, 1 / 60], "fog": [1, 1]})
    assert support_summary(raw, empty)["supported_events"] == 0
    assert support_summary(raw, empty)["truth_events"] == 1


@pytest.mark.parametrize("kind", ["left_rf", "bilateral_rf"])
def test_roundtrip_truth_future_independence_and_missing_right_is_unknown(kind, tmp_path):
    frame = signal(240)
    dual = anchored_gyro_windows(frame, require_right=True)
    windows = left_view(dual) if kind == "left_rf" else dual
    values, names = a1_matrix(windows, "baseline_2s_v1")
    forest = RandomForestClassifier(n_estimators=4, random_state=4).fit(values, [0, 1, 0, 1, 0])
    model = BilateralFoG(forest, kind, windows[0].channel_names, names, 0.4, (1,), (1,))
    model.save(tmp_path / "model.joblib")
    restored = BilateralFoG.load(tmp_path / "model.joblib")
    original = restored.predict_frame(frame)
    pd.testing.assert_frame_equal(original, restored.predict_frame(frame.assign(fog=1, activity=7)))
    frame.loc[120:, DUAL[-1]] = np.nan
    pd.testing.assert_series_equal(original.iloc[0], restored.predict_frame(frame).iloc[0])
    frame.loc[:, list(DUAL[3:])] = np.nan
    if kind == "bilateral_rf":
        with pytest.raises(ValueError, match="No complete"):
            restored.predict_frame(frame)
    else:
        assert len(restored.predict_frame(frame)) == 5
    assert not any(c.startswith("truth_") for c in original)


def test_evaluation_has_no_fitting_calls():
    source = (Path(__file__).parents[1] / "scripts/evaluate_a1_bilateral.py").read_text("utf-8")
    assert not [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"}]
