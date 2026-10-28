"""Architecture and isolation checks for the optional author-architecture arm."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fogguard.evaluation.a1_author import annotate, grouped_diagnostics, matched_frames
from fogguard.modeling.a1_author_cnn import (
    GYRO_CHANNELS,
    AuthorCNNClassifier,
    AuthorFoGCNN,
    GyroFoG,
    centered_gyro,
    gyro_windows,
)


def signal(n=180):
    frame = pd.DataFrame(np.random.default_rng(4).normal(size=(n, 3)), columns=GYRO_CHANNELS)
    frame["timestamp"] = np.arange(n) / 60
    frame["subjectID"], frame["sessionID"], frame["taskID"] = 1, 1, 1
    frame["fog"], frame["activity"] = 0, 1
    return frame


def test_gyro_does_not_require_or_fill_accel_and_includes_last_complete_window():
    frame = signal()
    frame["ankleL_acc_x"] = np.nan
    windows = gyro_windows(frame)
    assert len(windows) == 3
    assert windows[-1].end_s == 179 / 60
    assert windows[0].channel_names == GYRO_CHANNELS
    np.testing.assert_array_equal(windows[0].samples, frame[list(GYRO_CHANNELS)].iloc[:120])


@pytest.mark.parametrize("boundary", ["record", "missing", "time_gap"])
def test_windows_never_cross_record_missing_gyro_or_time_gap(boundary):
    frame = signal(300)
    if boundary == "record":
        frame.loc[150:, "taskID"] = 2
    elif boundary == "missing":
        frame.loc[150, GYRO_CHANNELS[0]] = np.nan
    else:
        frame.loc[150:, "timestamp"] += 1
    windows = gyro_windows(frame)
    assert all(not (w.start_s < 150 / 60 <= w.end_s) for w in windows)
    assert len(windows) == (3 if boundary == "missing" else 4)


def test_demeaning_is_window_local_and_ignores_labels():
    w = gyro_windows(signal())[0]
    a = centered_gyro([w])
    b = centered_gyro([replace(w, samples=w.samples + 900, fog_label=1, activity_label=7)])
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(a.mean(axis=2), 0, atol=1e-7)
    with pytest.raises(ValueError, match="120 x 3"):
        centered_gyro([replace(w, channel_names=GYRO_CHANNELS[::-1])])


def test_published_architecture_size_regularization_and_causal_conv():
    network = AuthorFoGCNN()
    assert sum(p.numel() for p in network.parameters()) == 8353
    assert network(torch.zeros(2, 3, 120)).shape == (2,)
    x = torch.randn(2, 3, 120)
    old = network.conv1(x)
    x[:, :, 50:] += 999
    torch.testing.assert_close(old[:, :, :50], network.conv1(x)[:, :, :50])
    before = network.kernel_penalty().detach().clone()
    with torch.no_grad():
        network.output.weight.add_(100)
        network.dense.bias.add_(100)
    torch.testing.assert_close(before, network.kernel_penalty())


def test_training_checkpoint_roundtrip_truth_and_future_independence(tmp_path):
    cfg = json.loads((Path(__file__).parents[1] / "config/a1_author_cnn_v1.json").read_text("utf-8"))["cnn"]
    cfg.update(max_epochs=3, patience=1, min_delta=100)
    frame = signal(240)
    windows = gyro_windows(frame)
    x = centered_gyro(windows)
    y = np.array([0, 1, 0, 1, 0])
    classifier = AuthorCNNClassifier(cfg, 5412).fit(x, y, (x, y))
    assert classifier.best_epoch_ == 1 and len(classifier.history_) == 2
    with torch.inference_mode():
        restored_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
            classifier.network_(torch.from_numpy(x)), torch.as_tensor(y, dtype=torch.float32)))
        restored_loss += cfg["kernel_l2"] * float(classifier.network_.kernel_penalty())
    assert restored_loss == pytest.approx(classifier.history_[0]["validation_loss"], abs=1e-6)
    model = GyroFoG(classifier, "author_cnn", GYRO_CHANNELS, (), 0.3, (1,), (2,))
    model.save(tmp_path / "model.joblib")
    loaded = GyroFoG.load(tmp_path / "model.joblib")
    original = model.predict_frame(frame)
    pd.testing.assert_frame_equal(original, loaded.predict_frame(frame))
    frame["fog"], frame["activity"] = 1, 7
    frame.loc[120:, GYRO_CHANNELS[0]] = 1e6
    pd.testing.assert_series_equal(original.iloc[0], loaded.predict_frame(frame).iloc[0])
    assert not any(c.startswith("truth_") for c in original)
    with pytest.raises(ValueError, match="gyro schema"):
        loaded.predict_windows([replace(windows[0], channel_names=GYRO_CHANNELS[::-1])])


def test_comparison_matches_inputs_and_preserves_no_fog_groups():
    windows = gyro_windows(signal())
    from fogguard.modeling.a1_fixed_split import prediction_frame

    a = annotate(windows, prediction_frame(windows, np.array([0.8, 0.2, 0.1]), 0.5, "a"), "validation")
    b = a.iloc[1:].copy()
    matched = matched_frames({"a": a, "b": b})
    assert len(matched["a"]) == len(matched["b"]) == 2
    groups = grouped_diagnostics(a, "subject_id")
    assert groups.iloc[0].positive_windows == 0 and groups.iloc[0].FP == 1
    b["truth_fog_label"] = 1
    with pytest.raises(ValueError, match="truth differs"):
        matched_frames({"a": a, "b": b})
