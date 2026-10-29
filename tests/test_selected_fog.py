"""Contracts for the retained methods, not tests for removed experiments."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import sensor_columns
from fogguard.modeling.a1_ankle_history import ankle_matrix, history_links
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.selected_fog import METHODS, load_selected, matrix_for, windows_for, wrap_classifier


def signal(n=420):
    names = (*sensor_columns(("ankleL",)), *DUAL[3:])
    f = pd.DataFrame(np.random.default_rng(82).normal(size=(n, len(names))), columns=names)
    return f.assign(timestamp=np.arange(n) / 60, subjectID=1, sessionID=1, taskID=1)


@pytest.mark.parametrize("method,features", [("ankle_history_rf", 140), ("bilateral_rf", 66)])
def test_rf_roundtrip_truth_independence_and_future_independence(tmp_path, method, features):
    frame = signal()
    windows = windows_for(frame, method)
    x, names = matrix_for(windows, method)
    assert x.shape == (11, features)
    forest = RandomForestClassifier(n_estimators=5, random_state=8).fit(x, np.arange(len(x)) % 2)
    model = wrap_classifier(forest, method, windows[0].channel_names, names, 0.3, [1], [1])
    plain = model.predict_frame(frame)
    pd.testing.assert_frame_equal(plain, model.predict_frame(frame.assign(fog=1, activity=7)))
    assert not any(c.startswith("truth_") for c in plain)
    assert np.allclose(plain.available_at_s, plain.window_end_s)
    changed = frame.copy()
    changed.loc[changed.timestamp > plain.window_end_s.iloc[2], list(DUAL)] *= 900
    pd.testing.assert_frame_equal(plain.iloc[:3], model.predict_frame(changed).iloc[:3])
    model.save(tmp_path / "model.joblib")
    pd.testing.assert_frame_equal(plain, load_selected(tmp_path / "model.joblib", method).predict_frame(frame))


def test_bilateral_missing_right_skips_without_reanchoring():
    frame = signal(300)
    frame.loc[130, DUAL[3]] = np.nan
    windows = anchored_gyro_windows(frame, require_right=True)
    np.testing.assert_allclose([w.start_s for w in windows], [0, 2.5, 3])
    assert all(np.isfinite(w.samples).all() for w in windows)


def test_history_flags_reset_at_records_and_gaps():
    windows = windows_for(signal(), "ankle_history_rf")
    x, names, links = ankle_matrix(windows)
    assert links[0].tolist() == [-1, -1] and links[8].tolist() == [4, 0]
    assert x[0, names.index("history_available__lag_4s")] == 0
    reset = replace(windows[9], task_id="2")
    assert history_links([*windows[:9], reset])[-1].tolist() == [-1, -1]
    model = wrap_classifier(RandomForestClassifier(n_estimators=3, random_state=1).fit(x, np.arange(len(x)) % 2),
        "ankle_history_rf", windows[0].channel_names, names, 0.4, [1], [1])
    stream = model.new_stream()
    incremental = pd.concat([stream.push(w) for w in windows], ignore_index=True)
    pd.testing.assert_frame_equal(model.predict_windows(windows), incremental)


def test_cnn_architecture_and_saved_input_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from fogguard.modeling.a1_author_cnn import AuthorCNNClassifier, AuthorFoGCNN, centered_gyro

    classifier = AuthorCNNClassifier({"dropout": 0.4, "batch_size": 64}, 5412)
    classifier.network_, classifier.classes_ = AuthorFoGCNN(), np.array([0, 1])
    assert sum(p.numel() for p in classifier.network_.parameters()) == 8353
    windows = windows_for(signal(180), "ankle_cnn")
    x = centered_gyro(windows)
    np.testing.assert_allclose(x.mean(axis=2), 0, atol=1e-6)
    model = wrap_classifier(classifier, "ankle_cnn", windows[0].channel_names, (), 0.5, [1], [1])
    a = model.predict_frame(signal(180))
    pd.testing.assert_frame_equal(a, model.predict_frame(signal(180).assign(fog=1, activity=7)))
    model.save(tmp_path / "cnn.joblib")
    np.testing.assert_allclose(a.fog_probability, load_selected(tmp_path / "cnn.joblib", "ankle_cnn").predict_frame(signal(180)).fog_probability, atol=1e-7)
    assert classifier.network_(torch.zeros(2, 3, 120)).shape == (2,)


def test_only_three_methods_and_readonly_evaluation():
    assert METHODS == ("ankle_history_rf", "bilateral_rf", "ankle_cnn")
    with pytest.raises(ValueError):
        windows_for(signal(), "removed")
    source = (Path(__file__).parents[1] / "scripts/evaluate_selected_fog.py").read_text("utf-8")
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)]
    assert not [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"}]
