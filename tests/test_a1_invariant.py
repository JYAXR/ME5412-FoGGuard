"""Representation invariance, missing-input schedule and truth-free model contract."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import FogStarWindow
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_invariant import FEATURES, InvariantFoG, feature_matrix


def example_windows():
    rng = np.random.default_rng(73)
    return [FogStarWindow(rng.normal(size=(120, 6)) * (1 + i), DUAL, 60, 0, 119 / 60, "1", "1", "1") for i in range(12)]


def example_model():
    windows = example_windows()
    x, names = feature_matrix(windows, "invariant_rf")
    forest = RandomForestClassifier(n_estimators=7, random_state=41).fit(x, np.arange(len(windows)) % 2)
    return InvariantFoG(forest, "invariant_rf", DUAL, names, 0.4, (1,), (1,)), windows


def test_model_scores_are_invariant_to_independent_ankle_rotations():
    model, windows = example_model()
    rng = np.random.default_rng(8)
    a, b = [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(2)]
    moved = [replace(w, samples=np.concatenate((w.samples[:, :3] @ a, w.samples[:, 3:] @ b), axis=1)) for w in windows]
    np.testing.assert_allclose(feature_matrix(windows, "invariant_rf")[0], feature_matrix(moved, "invariant_rf")[0], atol=1e-10)
    np.testing.assert_array_equal(model.predict_windows(windows).fog_probability, model.predict_windows(moved).fog_probability)


def test_sensor_only_inference_ignores_truth_and_roundtrips(tmp_path):
    model, windows = example_model()
    frame = pd.DataFrame(windows[0].samples, columns=DUAL).assign(timestamp=np.arange(120) / 60)
    expected = model.predict_frame(frame)
    pd.testing.assert_frame_equal(expected, model.predict_frame(frame.assign(fog=1, activity=7, fog_severity=3)))
    assert expected.available_at_s.iloc[0] == pytest.approx(119 / 60)
    assert not any(c.startswith("truth_") for c in expected)
    model.save(tmp_path / "model.joblib")
    pd.testing.assert_frame_equal(expected, InvariantFoG.load(tmp_path / "model.joblib").predict_frame(frame))


def test_right_gap_skips_windows_without_reanchoring_or_fill():
    frame = pd.DataFrame(np.ones((300, 6)), columns=DUAL).assign(timestamp=np.arange(300) / 60)
    frame.loc[130, DUAL[3]] = np.nan
    windows = anchored_gyro_windows(frame, require_right=True)
    np.testing.assert_allclose([w.start_s for w in windows], [0, 2.5, 3])
    assert all(np.isfinite(w.samples).all() for w in windows)


def test_wrong_schema_and_empty_inputs_are_rejected():
    model, windows = example_model()
    assert len(FEATURES) == 10
    for bad in ([], [replace(windows[0], sample_rate_hz=128)], [replace(windows[0], channel_names=DUAL[::-1])]):
        with pytest.raises(ValueError):
            model.predict_windows(bad)
    with pytest.raises(ValueError):
        feature_matrix(windows, "unknown")


def test_evaluation_cannot_refit_or_recalibrate_and_training_only_constructs_train():
    root = Path(__file__).parents[1]
    tree = ast.parse((root / "scripts/evaluate_a1_invariant.py").read_text("utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                ((isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"}) or
                 (isinstance(n.func, ast.Name) and n.func.id == "choose_threshold"))]
    train = ast.parse((root / "scripts/train_a1_invariant.py").read_text("utf-8"))
    roles = [n.args[2].value for n in ast.walk(train) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "role_frame"]
    assert roles == ["train"]
