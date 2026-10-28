"""A1 feature equivalence, runtime schema, group isolation and model persistence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fogguard.data.fogstar import FogStarWindow
from fogguard.evaluation.a1_validation import grouped_threshold, make_forest
from fogguard.modeling.a1_fog import A1FogModel, a1_features, a1_matrix
from fogguard.modeling.fogstar_baseline import extract_window_features


def window():
    rng = np.random.default_rng(5412)
    names = tuple(f"{p}_{s}_{a}" for p in ("ankleL", "back")
                  for s in ("acc", "gyro") for a in "xyz")
    return FogStarWindow(rng.normal(size=(120, 12)), names, 60, 0, 119 / 60,
                         "1", "1", "1", 1, 1.0, "left", 1.0, 7, 1.0)


def config():
    return {"n_estimators": 8, "min_samples_leaf": 2,
            "class_weight": "balanced_subsample", "random_state": 5412,
            "inner_folds": 3, "threshold_grid": {"start": 0.1, "stop": 0.9, "count": 9}}


def test_historical_subset_is_exact_and_candidate_adds_60_features():
    w = window()
    old, names = extract_window_features(w)
    expected = old[[i for i, name in enumerate(names)
                    if not name.endswith(("__raw_mean", "__raw_median"))]]
    baseline, _ = a1_features(w, "baseline_2s_v1")
    candidate, schema = a1_features(w, "recent_trend_v1")
    np.testing.assert_array_equal(baseline, expected)
    np.testing.assert_array_equal(candidate[:132], expected)
    assert len(candidate) == len(schema) == len(set(schema)) == 192
    assert np.isfinite(candidate).all()


def test_labels_never_change_features_and_short_windows_reject():
    w = window()
    np.testing.assert_array_equal(a1_features(w, "recent_trend_v1")[0], a1_features(
        replace(w, fog_label=0, direction_label="right", activity_label=6),
        "recent_trend_v1")[0])
    with pytest.raises(ValueError, match="120 samples"):
        a1_features(replace(w, samples=w.samples[:60]), "recent_trend_v1")


def test_channels_are_ordered_and_must_match():
    w = window()
    with pytest.raises(ValueError, match="Channel ordering"):
        a1_matrix([w, replace(w, channel_names=w.channel_names[::-1])], "baseline_2s_v1")


def test_grouped_threshold_never_shares_people_and_covers_all():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(48, 4))
    y = np.tile([0, 1], 24)
    groups = np.repeat(np.arange(6), 8)
    threshold, audit = grouped_threshold(x, y, groups, config())
    assert 0.1 <= threshold <= 0.9
    seen = []
    for fold in audit["folds"]:
        assert not set(fold["train"]) & set(fold["validation"])
        seen.extend(fold["validation"])
    assert sorted(seen) == list(map(str, range(6)))


def test_round_trip_and_future_samples_cannot_change_completed_output(tmp_path):
    w = window()
    x, names = a1_matrix([w, replace(w, samples=w.samples * 0.4),
                          replace(w, samples=w.samples * 2),
                          replace(w, samples=w.samples * 3)], "recent_trend_v1")
    forest = make_forest(config()).fit(x, [0, 0, 1, 1])
    model = A1FogModel(forest, "recent_trend_v1", w.channel_names, names, 0.3)
    path = tmp_path / "a1.joblib"
    model.save(path)
    loaded = A1FogModel.load(path)
    pd.testing.assert_frame_equal(model.predict_windows([w]), loaded.predict_windows([w]))
    frame = pd.DataFrame(np.vstack([w.samples, w.samples[:60]]), columns=w.channel_names)
    frame["timestamp"] = np.arange(180) / 60
    frame["fog"], frame["activity"] = 1, 7
    initial = loaded.predict_frame(frame).iloc[0]
    frame.loc[120:, w.channel_names[0]] = 1e9
    frame["fog"], frame["activity"] = 0, 1
    changed = loaded.predict_frame(frame).iloc[0]
    pd.testing.assert_series_equal(initial, changed)
    assert not any(name.startswith("truth_") for name in initial.index)


def test_missing_future_sample_does_not_remove_prior_window(tmp_path):
    w = window()
    x, names = a1_matrix([w, replace(w, samples=w.samples * 2)], "baseline_2s_v1")
    model = A1FogModel(make_forest(config()).fit(x, [0, 1]), "baseline_2s_v1",
                       w.channel_names, names, 0.3)
    frame = pd.DataFrame(np.vstack([w.samples, w.samples[:60]]), columns=w.channel_names)
    frame["timestamp"] = np.arange(180) / 60
    original = model.predict_frame(frame).iloc[0]
    frame.loc[120, w.channel_names[0]] = np.nan
    pd.testing.assert_series_equal(original, model.predict_frame(frame).iloc[0])
