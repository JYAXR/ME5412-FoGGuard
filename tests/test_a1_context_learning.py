"""Fit-only participant weights, auxiliary isolation and FoG-only runtime."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fogguard.modeling.a1_author_cnn import GYRO_CHANNELS, gyro_windows
from fogguard.modeling.a1_context_learning import (
    ContextClassifier,
    ContextFoG,
    ContextNetwork,
    activity_targets,
    person_weights,
    training_objective,
)
from fogguard.modeling.a1_temporal_cnn import sequence_batch


def sample():
    frame = pd.DataFrame(np.random.default_rng(4).normal(size=(360, 3)), columns=GYRO_CHANNELS)
    frame["timestamp"] = np.arange(360) / 60
    windows = gyro_windows(frame)
    return frame, windows, sequence_batch(windows)[:2]


def config():
    cfg = json.loads((Path(__file__).parents[1] / "config/a1_context_learning_v1.json").read_text("utf-8"))
    cfg["optimization"].update(max_epochs=2, patience=1, min_delta=100)
    return cfg


def test_equal_person_weights_use_only_fit_people_and_are_not_class_weights():
    weights, audit = person_weights([1, 2, 2, 2], True)
    np.testing.assert_allclose(weights, [2, 2 / 3, 2 / 3, 2 / 3])
    assert weights.mean() == pytest.approx(1)
    assert audit[0]["total_weight"] == audit[1]["total_weight"] == 2
    uniform, _ = person_weights([1, 2, 2, 2], False)
    np.testing.assert_array_equal(uniform, np.ones(4))
    with pytest.raises(ValueError, match="No fit"):
        person_weights([], True)


def test_activity_targets_ignore_ambiguous_unlabeled_and_allow_frozen_standing():
    _, w, _ = sample()
    rows = [replace(w[0], activity_label=v, fog_label=1) for v in (None, 0, 1, 3, 7)]
    np.testing.assert_array_equal(activity_targets(rows), [-100, -100, 0, 2, 6])


@pytest.mark.parametrize("coefficient", [0.0, 0.2])
def test_auxiliary_gradient_only_active_when_requested_and_padding_is_ignored(coefficient):
    torch.set_num_threads(2)
    _, _, (x, lengths) = sample()
    network = ContextNetwork().eval()
    assert sum(p.numel() for p in network.parameters()) == 14536
    x, lengths = torch.from_numpy(x), torch.as_tensor(lengths)
    fog, activity = network.joint_forward(x, lengths)
    y, targets = torch.as_tensor(np.arange(len(x)) % 2, dtype=torch.float32), torch.full((len(x),), 2)
    loss, _, _ = training_objective(fog, activity, y, targets, torch.ones(len(x)), coefficient)
    loss.backward()
    gradient = network.activity.weight.grad.abs().sum().item()
    assert (gradient > 0) == (coefficient > 0)
    assert network.base.output.weight.grad.abs().sum() > 0
    before = network(x, lengths)
    for i, length in enumerate(lengths):
        x[i, length:] = 1e8
    torch.testing.assert_close(before, network(x, lengths))
    ignored, _, auxiliary = training_objective(fog, activity, y, torch.full_like(targets, -100), torch.ones(len(x)), coefficient)
    assert torch.isfinite(ignored) and auxiliary.item() == 0


def test_control_training_does_not_depend_on_activity_labels():
    _, _, (x, lengths) = sample()
    cfg = config()
    labels, groups = np.arange(len(x)) % 2, np.repeat([1, 2, 3], 3)
    models = [ContextClassifier("control", cfg["optimization"], 2, cfg["arms"]["control"]).fit(
        x, lengths, labels, groups=groups, activities=np.full(len(x), target), fixed_epochs=1) for target in (0, 6)]
    np.testing.assert_array_equal(models[0].predict_proba(x, lengths), models[1].predict_proba(x, lengths))


def test_aux_runtime_roundtrip_stream_and_future_truth_independence(tmp_path):
    frame, windows, (x, lengths) = sample()
    cfg = config()
    classifier = ContextClassifier("activity_aux", cfg["optimization"], 5, cfg["arms"]["activity_aux"]).fit(
        x, lengths, np.arange(len(x)) % 2, groups=np.repeat([1, 2, 3], 3), activities=np.full(len(x), 2), fixed_epochs=1)
    model = ContextFoG(classifier, "activity_aux", GYRO_CHANNELS, (), 0.5, (1, 2, 3), (1, 2, 3))
    model.save(tmp_path / "model.joblib")
    restored = ContextFoG.load(tmp_path / "model.joblib")
    baseline = restored.predict_frame(frame)
    pd.testing.assert_frame_equal(baseline, restored.predict_frame(frame.assign(fog=1, activity=3)))
    assert baseline.context_windows.max() == 3
    stream = restored.new_stream()
    predictions = pd.concat([stream.push(w) for w in windows], ignore_index=True)
    np.testing.assert_allclose(baseline.fog_probability, predictions.fog_probability, atol=1e-6, rtol=0)
    changed = frame.copy()
    changed.loc[120:, GYRO_CHANNELS[0]] = 1e8
    pd.testing.assert_series_equal(baseline.iloc[0], restored.predict_frame(changed).iloc[0])
    assert not any(c.startswith("truth_") or c in {"activity", "fog"} for c in baseline)


def test_checkpoint_uses_only_unweighted_fog_and_evaluation_cannot_fit():
    _, _, (x, lengths) = sample()
    cfg = config()
    labels, groups = np.arange(len(x)) % 2, np.repeat([1, 2, 3], 3)
    model = ContextClassifier("person_balanced", cfg["optimization"], 4, cfg["arms"]["person_balanced"]).fit(
        x, lengths, labels, groups=groups, activities=np.full(len(x), -100), validation=(x, lengths, labels))
    assert model.best_epoch_ == 1 and len(model.history_) == 2
    with torch.inference_mode():
        loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
            model.network_(torch.from_numpy(x), torch.as_tensor(lengths)), torch.as_tensor(labels, dtype=torch.float32)))
        loss += cfg["optimization"]["kernel_l2"] * float(model.network_.kernel_penalty())
    assert loss == pytest.approx(model.history_[0]["checkpoint_loss"], abs=1e-6)
    source = (Path(__file__).parents[1] / "scripts/evaluate_a1_context_learning.py").read_text("utf-8")
    assert not [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"}]
