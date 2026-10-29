"""Contracts for the retained methods, not tests for removed experiments."""

import ast
import hashlib
import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import sensor_columns
from fogguard.modeling.a1_ankle_history import ankle_matrix, history_links
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.selected_fog import (
    METHODS,
    check_lock,
    load_selected,
    matrix_for,
    windows_for,
    wrap_classifier,
)


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


@pytest.mark.parametrize("sample_rate", [50, 64, 128])
def test_cnn_rejects_wrong_sample_clock(sample_rate):
    pytest.importorskip("torch")
    frame = signal(180).assign(timestamp=np.arange(180) / sample_rate)
    with pytest.raises(ValueError, match="60Hz"):
        windows_for(frame, "ankle_cnn")


def test_cnn_rejects_internal_jitter_even_with_correct_window_span():
    pytest.importorskip("torch")
    frame = signal(180)
    frame.loc[20, "timestamp"] += 0.002
    with pytest.raises(ValueError, match="60Hz"):
        windows_for(frame, "ankle_cnn")


def test_cnn_checks_direct_window_span_and_accepts_offset_timestamps():
    pytest.importorskip("torch")
    from fogguard.modeling.a1_author_cnn import centered_gyro

    frame = signal(180)
    shifted = frame.assign(timestamp=frame.timestamp + 100.25)
    plain, offset = [windows_for(f, "ankle_cnn") for f in (frame, shifted)]
    np.testing.assert_array_equal(centered_gyro(plain), centered_gyro(offset))
    assert offset[0].end_s == pytest.approx(100.25 + 119 / 60)
    invalid = replace(plain[0], end_s=plain[0].start_s + 119 / 128)
    with pytest.raises(ValueError, match="60Hz"):
        centered_gyro([invalid])


@pytest.mark.parametrize("relative", ["model/weights.bin", "model\\weights.bin"])
def test_manifest_separators_and_tamper_detection(tmp_path, relative):
    weight = tmp_path / "model/weights.bin"
    weight.parent.mkdir()
    weight.write_bytes(b"original payload")
    lock = {"files": {relative: hashlib.sha256(weight.read_bytes()).hexdigest()}}
    check_lock(tmp_path, lock)
    weight.write_bytes(b"modified payload")
    with pytest.raises(ValueError, match="Changed artifact"):
        check_lock(tmp_path, lock)


@pytest.mark.parametrize("relative", ["../weights.bin", "/weights.bin", "C:\\weights.bin"])
def test_manifest_requires_bundle_relative_paths(tmp_path, relative):
    with pytest.raises(ValueError, match="relative to its bundle"):
        check_lock(tmp_path, {"files": {relative: "unused"}})


def test_bundle_cli_from_windows_manifest_is_portable(tmp_path):
    pytest.importorskip("torch")
    from fogguard.modeling.a1_author_cnn import AuthorCNNClassifier, AuthorFoGCNN

    root = Path(__file__).parents[1]
    models = tmp_path / "models"
    for method in METHODS:
        windows = windows_for(signal(), method)
        values, names = matrix_for(windows, method)
        if method == "ankle_cnn":
            classifier = AuthorCNNClassifier({"batch_size": 64}, 5412)
            classifier.network_, classifier.classes_ = AuthorFoGCNN(), np.array([0, 1])
        else:
            classifier = RandomForestClassifier(n_estimators=3, random_state=8).fit(
                values, np.arange(len(values)) % 2)
        wrap_classifier(classifier, method, windows[0].channel_names, names, 0.5, [1], [1]).save(
            models / method / "model.joblib")
    # A Windows-authored bundle must load on Ubuntu as well as Windows.
    files = {str(p.relative_to(models)).replace("/", "\\"): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in models.rglob("*.joblib")}
    (models / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    output = tmp_path / "output"
    process = subprocess.run([sys.executable, "-X", "utf8", "scripts/bundle_selected_fog.py",
                              "--models", str(models), "--output", str(output)], cwd=root,
                             capture_output=True, text=True, encoding="utf-8", check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    with zipfile.ZipFile(output / "FoGGuard-Selected-3-Models.zip") as archive:
        manifest = json.loads(archive.read("fog-selected/MANIFEST.json"))
        assert all("\\" not in name for name in manifest["files"])
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read("fog-selected/" + name)).hexdigest() == digest
        archive.extractall(tmp_path / "unpacked")
    check_lock(tmp_path / "unpacked/fog-selected", manifest)
