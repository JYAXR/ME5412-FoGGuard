"""Causal-history correctness, nested roles, padding exclusion and stream parity."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fogguard.evaluation.a1_temporal import assert_history_isolation, nested_plan
from fogguard.modeling.a1_author_cnn import GYRO_CHANNELS, gyro_windows
from fogguard.modeling.a1_temporal_cnn import (
    TemporalClassifier,
    TemporalFoG,
    TemporalNetwork,
    history_indices,
    sequence_batch,
)


def windows(n=600):
    frame = pd.DataFrame(np.random.default_rng(8).normal(size=(n, 3)), columns=GYRO_CHANNELS)
    frame["timestamp"] = np.arange(n) / 60
    return gyro_windows(frame)


def options():
    cfg = json.loads((Path(__file__).parents[1] / "config/a1_temporal_cnn_v1.json").read_text("utf-8"))["optimization"]
    cfg.update(max_epochs=2, patience=1, min_delta=100)
    return cfg


def test_history_uses_only_nonoverlapping_past_windows_without_startup_wait():
    w = windows()
    links = history_indices(w)
    np.testing.assert_array_equal(links[0], [0, -1, -1])
    np.testing.assert_array_equal(links[4], [0, 4, -1])
    np.testing.assert_array_equal(links[8], [0, 4, 8])
    x, lengths, _ = sequence_batch(w)
    assert x.shape == (17, 3, 3, 120) and lengths[0] == 1
    np.testing.assert_array_equal(history_indices([replace(v, fog_label=1, activity_label=3) for v in w]), links)
    assert np.all(links <= np.arange(len(w))[:, None])


@pytest.mark.parametrize("reset", ["record", "gap"])
def test_history_resets_on_record_or_missing_decision(reset):
    w = windows()
    tail = [replace(v, task_id="other") for v in w[9:]] if reset == "record" else w[10:]
    changed = w[:9] + tail
    np.testing.assert_array_equal(history_indices(changed)[9], [9, -1, -1])
    with pytest.raises(ValueError, match="chronological"):
        history_indices([w[0], w[0]])


@pytest.mark.parametrize("kind,count", [("cnn_control", 8353), ("cnn_gru", 14305)])
def test_architecture_and_padding_do_not_change_real_input(kind, count):
    torch.set_num_threads(2)
    network = TemporalNetwork(kind).eval()
    assert sum(p.numel() for p in network.parameters()) == count
    all_x, all_lengths, _ = sequence_batch(windows(360))
    x, lengths = all_x[[0, 4, 8]], all_lengths[[0, 4, 8]]
    a = torch.from_numpy(x.copy())
    l = torch.as_tensor(lengths)
    before = network(a, l)
    for i, size in enumerate(lengths):
        a[i, size:] = 1e8
    torch.testing.assert_close(before, network(a, l))
    if kind == "cnn_control":
        a[2, :int(lengths[2]) - 1] += 1e8
        torch.testing.assert_close(before, network(a, l))


def test_nested_checkpoint_plan_is_disjoint_and_does_not_read_outer_held_labels():
    groups = np.repeat(np.arange(12), 2)
    labels = np.repeat(np.arange(12) % 2, 2)
    folds = [{"fold": i, "held_out_people": list(range(i * 4, i * 4 + 4)),
              "fit_people": [p for p in range(12) if p // 4 != i]} for i in range(3)]
    a = nested_plan(groups, labels, folds, checkpoint_people=3, seed=22)
    changed = labels.copy()
    changed[groups < 4] = 1 - changed[groups < 4]
    b = nested_plan(groups, changed, folds, checkpoint_people=3, seed=22)
    assert a[0] == b[0]
    for plan in a:
        assert not set(plan["inner_fit_people"]) & set(plan["checkpoint_people"])
        assert not set(plan["fit_people"]) & set(plan["held_out_people"])
    with pytest.raises(ValueError, match="forbidden"):
        assert_history_isolation(np.array([[0, 1, -1]]), [0, 1], [0], [0])


def test_fixed_fit_roundtrip_stream_future_and_truth_independence(tmp_path):
    w = windows()
    x, lengths, _ = sequence_batch(w)
    labels = np.arange(len(w)) % 2
    classifier = TemporalClassifier("cnn_gru", options(), 5412).fit(x, lengths, labels, fixed_epochs=1)
    model = TemporalFoG(classifier, "cnn_gru", GYRO_CHANNELS, (), 0.5, (1,), (1,))
    model.save(tmp_path / "model.joblib")
    restored = TemporalFoG.load(tmp_path / "model.joblib")
    batch = restored.predict_windows(w)
    stream = restored.new_stream()
    outputs = pd.concat([stream.push(v) for v in w], ignore_index=True)
    np.testing.assert_allclose(batch.fog_probability, outputs.fog_probability, atol=1e-6, rtol=0)
    changed = w[:9] + [replace(v, samples=v.samples * 1000, fog_label=1, activity_label=7) for v in w[9:]]
    pd.testing.assert_frame_equal(batch.iloc[:9], restored.predict_windows(changed).iloc[:9])
    reset = stream.push(replace(w[0], task_id="other"))
    assert reset.context_windows.iloc[0] == 1
    assert not any(c.startswith("truth_") for c in batch)


def test_inner_early_stop_restores_checkpoint_and_evaluator_cannot_fit():
    w = windows(180)
    x, lengths, _ = sequence_batch(w)
    y = [0, 1, 0]
    classifier = TemporalClassifier("cnn_control", options(), 4).fit(x, lengths, y, validation=(x, lengths, y))
    assert classifier.best_epoch_ == 1 and len(classifier.history_) == 2
    with torch.inference_mode():
        loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
            classifier.network_(torch.from_numpy(x), torch.as_tensor(lengths)), torch.as_tensor(y, dtype=torch.float32)))
        loss += options()["kernel_l2"] * float(classifier.network_.kernel_penalty())
    assert loss == pytest.approx(classifier.history_[0]["checkpoint_loss"], abs=1e-6)
    with pytest.raises(ValueError, match="OR fixed"):
        TemporalClassifier("cnn_control", options(), 4).fit(x, lengths, y)
    source = (Path(__file__).parents[1] / "scripts/evaluate_a1_temporal_cnn.py").read_text("utf-8")
    assert not [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in {"fit", "fit_transform", "partial_fit"}]
