"""Raw low-frequency information, causal history and train-only model selection."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fogguard.data.fogstar import FogStarWindow
from fogguard.evaluation.a1_posture import nested_candidate_scores
from fogguard.modeling.a1_fog import A1FogModel
from fogguard.modeling.a1_posture import (
    POSTURE_VARIANT,
    PostureFoG,
    posture_matrix,
    posture_summary,
    search_classifier,
)


def windows():
    rng = np.random.default_rng(5412)
    names = tuple(f"{p}_{s}_{a}" for p in ("ankleL", "back") for s in ("acc", "gyro") for a in "xyz")
    data = rng.normal(size=(450, 12))
    return [FogStarWindow(data[30 * i:30 * i + 120], names, 60, i * 0.5, i * 0.5 + 119 / 60,
                           "1", "1", "1") for i in range(12)]


def config():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "config/a1_posture_search_v1.json").read_text("utf-8"))
    for candidate in cfg["candidates"]:
        if "n_estimators" in candidate["params"]:
            candidate["params"]["n_estimators"] = 4
        else:
            candidate["params"]["max_iter"] = 3
    cfg["threshold_grid"]["count"] = 5
    return cfg


def test_posture_restores_raw_slow_components_and_has_finite_zero_encoding():
    ws = windows()
    base, names = posture_summary(ws[0])
    changed = ws[0].samples.copy()
    changed[:, ws[0].channel_names.index("back_gyro_z")] += 7
    after, schema = posture_summary(replace(ws[0], samples=changed))
    assert len(base) == len(names) == 32 and names == schema
    i = names.index("back__raw_gyro_mean_z")
    assert after[i] - base[i] == pytest.approx(7)
    assert np.isfinite(posture_summary(replace(ws[0], samples=np.zeros_like(changed)))[0]).all()


def test_438_features_use_only_current_and_past_labels_never_enter():
    ws = windows()
    values, names, links = posture_matrix(ws)
    assert values.shape == (12, 438) and len(names) == len(set(names)) == 438
    assert links[8].tolist() == [4, 0]
    current = names.index("back__raw_gyro_mean_z")
    lag = names.index("back__raw_gyro_mean_z__lag_4s")
    assert values[8, lag] == values[0, current]
    changed = [replace(w, fog_label=1, fog_fraction=1, activity_label=7, direction_label="left") for w in ws]
    changed[9:] = [replace(w, samples=w.samples * 1000) for w in changed[9:]]
    np.testing.assert_array_equal(values[:9], posture_matrix(changed)[0][:9])
    for i in range(9, len(changed)):
        changed[i] = replace(changed[i], task_id="2")
    assert posture_matrix(changed)[2][9].tolist() == [-1, -1]


def test_search_uses_same_participant_splits_for_all_candidates_and_checks_forbidden():
    cfg = config()
    rng = np.random.default_rng(5)
    matrices = {v: rng.normal(size=(48, 8)) for v in cfg["feature_variants"]}
    labels, groups = np.tile([0, 1], 24), np.repeat(np.arange(6), 8)
    audit, scores = nested_candidate_scores(matrices, labels, groups, cfg)
    assert len(audit["candidates"]) == len(scores) == 6
    for fold in audit["inner_folds"]:
        assert not set(fold["train"]) & set(fold["validation"])
    best = max(r["selected_threshold"]["fog_f1"] for r in audit["candidates"])
    selected = next(r for r in audit["candidates"] if r["candidate_id"] == audit["selected_candidate_id"])
    assert selected["selected_threshold"]["fog_f1"] == best
    with pytest.raises(ValueError, match="Test people"):
        nested_candidate_scores(matrices, labels, groups, cfg, forbidden=[0])


def test_model_roundtrip_streaming_and_sensor_only_frame_future_invariance(tmp_path):
    ws, cfg = windows(), config()
    values, names, _ = posture_matrix(ws)
    candidate = cfg["candidates"][1]
    forest = search_classifier(candidate, cfg["random_state"]).fit(values, np.arange(12) % 2)
    model = PostureFoG(forest, POSTURE_VARIANT, ws[0].channel_names, names, 0.3, candidate_id=candidate["id"])
    file = tmp_path / "posture.joblib"
    model.save(file)
    loaded = A1FogModel.load(file)
    stream = loaded.new_stream()
    pd.testing.assert_frame_equal(loaded.predict_windows(ws),
                                   pd.concat([stream.push(w) for w in ws], ignore_index=True))
    raw = np.vstack([ws[0].samples, *(w.samples[-30:] for w in ws[1:])])
    frame = pd.DataFrame(raw, columns=ws[0].channel_names)
    frame["timestamp"] = np.arange(len(raw)) / 60
    before = loaded.predict_frame(frame).iloc[:9]
    frame.loc[360:, ws[0].channel_names[0]] = np.nan
    frame["fog"], frame["activity"] = 1, 7
    pd.testing.assert_frame_equal(before, loaded.predict_frame(frame).iloc[:9])
    assert not any(c.startswith("truth_") for c in before)
