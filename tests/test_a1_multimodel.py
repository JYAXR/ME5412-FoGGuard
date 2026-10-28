"""Check same-input preparation, training-only normalization, folds and persistence."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fogguard.data.fogstar import FogStarWindow
from fogguard.evaluation.a1_multimodel import inner_grouped_scores
from fogguard.modeling.a1_multimodel import (
    MultiModelFoG,
    make_classifier,
    model_scores,
    waveform_matrix,
)


def config():
    cfg = json.loads((Path(__file__).resolve().parents[1]
                      / "config/a1_multimodel_v1.json").read_text(encoding="utf-8"))
    cfg["rf"]["n_estimators"] = 4
    cfg["hist_gb"]["max_iter"] = 5
    cfg["hist_gb"]["min_samples_leaf"] = 2
    cfg["cnn"]["epochs"] = 1
    cfg["threshold_grid"]["count"] = 5
    return cfg


def window():
    names = tuple(f"{p}_{s}_{a}" for p in ("ankleL", "back")
                  for s in ("acc", "gyro") for a in "xyz")
    return FogStarWindow(np.random.default_rng(1).normal(size=(120, 12)), names, 60,
                         0, 119 / 60, "1", "1", "1", 1, 1, "left", 1, 7, 1)


def test_waveform_uses_same_channels_and_does_not_use_truth():
    w = window()
    x = waveform_matrix([w])
    assert x.shape == (1, 12, 120)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x, waveform_matrix([replace(w, fog_label=0, activity_label=1)]))
    with pytest.raises(ValueError, match="ordered"):
        waveform_matrix([w, replace(w, channel_names=w.channel_names[::-1])])


@pytest.mark.parametrize("family", ["rf", "svm", "hist_gb"])
def test_tabular_models_return_bounded_scores_and_preserve_scaler(family):
    cfg = config()
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(32, 132)), np.tile([0, 1], 16)
    model = make_classifier(family, cfg).fit(x, y)
    scores = model_scores(model, family, x)
    assert len(scores) == len(x) and ((scores >= 0) & (scores <= 1)).all()
    if family == "svm":
        scaler = model.steps[0][1]
        np.testing.assert_allclose(scaler.mean_, x.mean(axis=0))
        before = scaler.mean_.copy()
        model_scores(model, family, x + 10000)
        np.testing.assert_array_equal(before, scaler.mean_)
        assert model.steps[-1][1].probability is not True


def test_inner_group_assignment_identical_across_algorithms():
    cfg = config()
    rng = np.random.default_rng(3)
    x, y, g = rng.normal(size=(48, 8)), np.tile([0, 1], 24), np.repeat(np.arange(6), 8)
    _, a = inner_grouped_scores(x, y, g, "rf", cfg)
    _, b = inner_grouped_scores(x, y, g, "svm", cfg)
    assert a["folds"] == b["folds"]
    for fold in a["folds"]:
        assert not set(fold["train"]) & set(fold["validation"])
    assert a["selected"]["fog_f1"] == max(r["fog_f1"] for r in a["threshold_scores"])


def test_cnn_normalization_only_fitted_on_training_and_persists(tmp_path):
    pytest.importorskip("torch")
    cfg = config()
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(16, 12, 120)).astype(np.float32), np.tile([0, 1], 8)
    classifier = make_classifier("cnn", cfg).fit(x, y)
    np.testing.assert_allclose(classifier.mean_, x.mean(axis=(0, 2), keepdims=True))
    before = classifier.mean_.copy()
    model_scores(classifier, "cnn", x + 1000)
    np.testing.assert_array_equal(before, classifier.mean_)
    w = window()
    model = MultiModelFoG(classifier, "cnn", w.channel_names, 0.5)
    path = tmp_path / "cnn.joblib"
    model.save(path)
    restored = MultiModelFoG.load(path)
    pd.testing.assert_frame_equal(model.predict_windows([w]), restored.predict_windows([w]))
    assert not any(c.startswith("truth_") for c in model.predict_windows([w]))


def test_unexpected_channels_rejected_before_prediction():
    w = window()
    model = MultiModelFoG(None, "cnn", w.channel_names, 0.5)
    with pytest.raises(ValueError, match="unexpected channel"):
        model.predict_windows([replace(w, channel_names=w.channel_names[::-1])])
