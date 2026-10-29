"""Meaningful invariants of bilateral proxies and causally available runtime."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_coordination import (
    EXTRA_NAMES,
    CoordinationFoG,
    coordination_features,
    coordination_matrix,
    lag_correlation,
)
from fogguard.modeling.a1_fog import a1_matrix


def signal(n=240):
    frame = pd.DataFrame(np.random.default_rng(17).normal(size=(n, 6)), columns=DUAL)
    frame["timestamp"] = np.arange(n) / 60
    frame["subjectID"], frame["sessionID"], frame["taskID"] = 1, 1, 1
    return frame


def test_same_ankles_zero_energy_and_known_delay():
    window = anchored_gyro_windows(signal(120), require_right=True)[0]
    same = replace(window, samples=np.tile(window.samples[:, :3], (1, 2)))
    values = coordination_features(same)
    np.testing.assert_allclose(values[[0, 1, 2, 5, 8, 9, 11]], 0, atol=1e-12)
    np.testing.assert_allclose(values[[3, 4, 10]], 1, atol=1e-12)
    np.testing.assert_array_equal(coordination_features(replace(window, samples=np.zeros((120, 6)))), np.zeros(12))
    a = np.random.default_rng(19).normal(size=120)
    b = np.r_[np.zeros(12), a[:-12]]
    scores = [lag_correlation(a, b, lag) for lag in range(-30, 31)]
    assert np.argmax(scores) - 30 == 12 and scores[42] == pytest.approx(1)


def test_extra_proxies_rotate_independently_and_swap_as_declared():
    window = anchored_gyro_windows(signal(120), require_right=True)[0]
    q1 = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))[0]
    q2 = np.linalg.qr(np.random.default_rng(12).normal(size=(3, 3)))[0]
    original = coordination_features(window)
    rotated = replace(window, samples=np.c_[window.samples[:, :3] @ q1, window.samples[:, 3:] @ q2])
    np.testing.assert_allclose(coordination_features(rotated), original, atol=1e-10)
    swapped = replace(window, samples=np.c_[window.samples[:, 3:], window.samples[:, :3]])
    changed = coordination_features(swapped)
    np.testing.assert_allclose(changed[[0, 8, 9]], -original[[0, 8, 9]], atol=1e-12)
    keep = [1, 2, 3, 4, 5, 6, 7, 10, 11]
    np.testing.assert_allclose(changed[keep], original[keep], atol=1e-12)


def test_feature_prefix_is_unchanged_and_truth_is_unused():
    windows = anchored_gyro_windows(signal(), require_right=True)
    bank = coordination_matrix(windows)
    base, names = a1_matrix(windows, "baseline_2s_v1")
    np.testing.assert_array_equal(base, bank["coordination_rf"][0][:, :66])
    assert bank["coordination_rf"][1] == names + EXTRA_NAMES
    assert bank["coordination_rf"][0].shape == (5, 78)
    changed = [replace(w, fog_label=1, activity_label=7) for w in windows]
    np.testing.assert_array_equal(bank["coordination_rf"][0], coordination_matrix(changed)["coordination_rf"][0])


@pytest.mark.parametrize("kind", ["bilateral_control", "coordination_rf"])
def test_roundtrip_future_independence_truth_free_and_gap_unknown(kind, tmp_path):
    frame = signal()
    windows = anchored_gyro_windows(frame, require_right=True)
    values, names = coordination_matrix(windows)[kind]
    classifier = RandomForestClassifier(n_estimators=4, random_state=11).fit(values, [0, 1, 0, 1, 0])
    model = CoordinationFoG(classifier, kind, DUAL, names, 0.4, (1,), (1,))
    model.save(tmp_path / "model.joblib")
    restored = CoordinationFoG.load(tmp_path / "model.joblib")
    result = restored.predict_frame(frame)
    pd.testing.assert_frame_equal(result, restored.predict_frame(frame.assign(fog=1, activity=7, fog_severity=3)))
    frame.loc[120:, list(DUAL)] *= 100
    pd.testing.assert_series_equal(result.iloc[0], restored.predict_frame(frame).iloc[0])
    frame.loc[150, DUAL[-1]] = np.nan
    assert list(restored.predict_frame(frame).window_start_s) == [0, 0.5]
    frame.loc[:, list(DUAL[3:])] = np.nan
    with pytest.raises(ValueError, match="No complete"):
        restored.predict_frame(frame)
    assert not any(c.startswith("truth_") for c in result)


def test_evaluation_is_readonly():
    source = (Path(__file__).parents[1] / "scripts/evaluate_a1_coordination.py").read_text("utf-8")
    assert not [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"}]
