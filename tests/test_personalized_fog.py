"""Block separation, adaptation-only choices and label-free saved inference."""

import json

import numpy as np
import pandas as pd
import pytest

from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_fog import a1_matrix
from fogguard.modeling.personalized_fog import (
    KEYS,
    PersonalStudyModel,
    adaptation_status,
    choose_calibration,
    fit_model,
    plan_blocks,
    record_mask,
)


def records(rows):
    return pd.DataFrame(rows, columns=KEYS)


def test_multiple_sessions_are_held_out_as_whole_blocks():
    source = records([(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2)])
    folds = plan_blocks(source)
    assert len(folds) == 2
    assert all(f["split_kind"] == "session_held_out" for f in folds)
    for fold in folds:
        assert {r[1] for r in fold["adaptation_records"]}.isdisjoint(
            {r[1] for r in fold["evaluation_records"]})


def test_one_session_uses_whole_records_and_each_record_is_evaluated_once():
    source = records([(1, 1, 1), (1, 1, 2), (1, 1, 3), (2, 1, 1), (2, 1, 2)])
    folds = plan_blocks(source)
    evaluated = [tuple(r) for f in folds for r in f["evaluation_records"]]
    assert len(evaluated) == len(set(evaluated)) == len(source)
    for fold in folds:
        assert fold["split_kind"] == "record_held_out"
        assert not set(map(tuple, fold["adaptation_records"])) & set(map(tuple, fold["evaluation_records"]))
        assert all(r[0] == fold["subject_id"] for r in fold["adaptation_records"])


def test_eligibility_cannot_depend_on_evaluation_labels():
    meta = records([(1, 1, 1), (1, 1, 1), (1, 1, 2), (1, 1, 2)]).astype(str)
    meta["truth_fog_label"] = [0, 1, 0, 0]
    fold = plan_blocks(meta)[1]
    config = {"minimum_adaptation_positive_windows": 1, "minimum_adaptation_negative_windows": 1}
    first = adaptation_status(meta, fold, config)
    meta.loc[record_mask(meta, fold["evaluation_records"]), "truth_fog_label"] = 1
    assert first == adaptation_status(meta, fold, config)
    assert first["eligible"]


def test_missing_positive_adaptation_data_is_explicitly_ineligible():
    meta = records([(1, 1, 1), (1, 1, 2)]).astype(str).assign(truth_fog_label=0)
    config = {"minimum_adaptation_positive_windows": 1, "minimum_adaptation_negative_windows": 1}
    result = adaptation_status(meta, plan_blocks(meta)[0], config)
    assert not result["eligible"]
    assert result["reason"] == "insufficient_adaptation_positive_windows"


def test_calibration_requires_both_classes_and_uses_fixed_tie_break():
    selected, _ = choose_calibration([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9],
                                     {"start": 0.25, "stop": 0.75, "count": 3})
    assert selected["threshold"] == 0.75 and selected["f1"] == 1
    with pytest.raises(ValueError):
        choose_calibration([0, 0], [0.1, 0.2], {"start": 0.1, "stop": 0.9, "count": 3})


def test_duplicate_sample_times_in_distinct_records_do_not_join_blocks():
    meta = records([(1, 1, 1), (1, 1, 2), (2, 1, 1)]).astype(str).assign(window_end_s=2)
    np.testing.assert_array_equal(record_mask(meta, [["1", "1", "1"]]), [True, False, False])


def test_fitted_models_and_calibration_roundtrip_ignore_annotation_columns(tmp_path):
    signal = pd.DataFrame(np.random.default_rng(42).normal(size=(360, 6)), columns=DUAL)
    signal = signal.assign(timestamp=np.arange(360) / 60, subjectID=1, sessionID=1, taskID=1)
    x, names = a1_matrix(anchored_gyro_windows(signal, require_right=True), "baseline_2s_v1")
    config = {"rf": {"n_estimators": 4, "min_samples_leaf": 1, "n_jobs": 1},
              "random_state": 1, "fixed_threshold": 0.23}
    model = fit_model(x, np.arange(len(x)) % 2, names, config,
                      strategy="population", fit_people=[2, 3])
    model.save(tmp_path / "base.joblib")
    loaded = PersonalStudyModel.load(tmp_path / "base.joblib")
    pd.testing.assert_frame_equal(model.predict_frame(signal),
                                  loaded.predict_frame(signal.assign(fog=1, activity=7)))
    # A calibration file supplies no target truth to inference.
    (tmp_path / "calibration.json").write_text(json.dumps({"base_model": "base.joblib",
        "selected": {"threshold": 0.8}, "adaptation_records": [["1", "2", "1"]]}), encoding="utf-8")
    calibrated = PersonalStudyModel.load(tmp_path / "calibration.json")
    expected, actual = model.predict_frame(signal), calibrated.predict_frame(signal)
    np.testing.assert_array_equal(expected.fog_probability, actual.fog_probability)
    assert actual.fog_threshold.eq(0.8).all()
    assert (actual.is_fog == actual.fog_probability.ge(0.8)).all()
    assert not any(c.startswith("truth_") for c in actual)
