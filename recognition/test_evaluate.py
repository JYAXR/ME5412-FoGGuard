import copy
from types import SimpleNamespace

import pandas as pd
import pytest
from evaluate import (
    acceleration_magnitude,
    metrics,
    validate_predictions,
    validate_raw_windows,
)


def fixture():
    data = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "session_id": [1] * 4,
            "task_id": [1] * 4,
            "window_start_s": [0, 0.5, 2, 2.5],
            "window_end_s": [1.98333, 2.48333, 3.98333, 4.48333],
            "truth_fog_label": [1, 1, 0, 0],
            "truth_fog_fraction": [1, 1, 0, 0],
            "fog_probability": [0.8, 0.1, 0.8, 0.1],
            "fog_threshold": [0.3] * 4,
            "is_fog": [True, False, True, False],
        }
    )
    report = {
        "validation": {
            "fold_subjects": [
                {"train": ["2"], "test": ["1"]},
                {"train": ["1"], "test": ["2"]},
            ],
            "fold_fog_thresholds": [0.3, 0.3],
        }
    }
    return data, report


def test_known_confusion_and_folds():
    frame, report = fixture()
    checked = validate_predictions(frame, report)
    m = metrics(checked)
    assert [m[k] for k in ["TP", "TN", "FP", "FN"]] == [1, 1, 1, 1]
    assert m["fog_f1"] == m["macro_f1"] == m["recall"] == 0.5
    assert checked.held_out_fold.tolist() == [1, 1, 2, 2]
    assert checked.outcome.tolist() == ["TP", "FN", "FP", "TN"]


def test_absent_positive_recall_is_not_perfect():
    frame, _ = fixture()
    m = metrics(frame.iloc[2:])
    assert m["recall"] is None
    assert m["auc"] is None
    assert m["balanced_accuracy"] is None


def test_missing_sensor_is_never_displayed_as_zero():
    frame = pd.DataFrame(
        {
            "ankleL_acc_x": [3, 0, None, None],
            "ankleL_acc_y": [4, 0, None, 4],
            "ankleL_acc_z": [0, 0, None, 0],
        }
    )
    result = acceleration_magnitude(frame, "ankleL")
    assert result.iloc[:2].tolist() == [5, 0]
    assert result.iloc[2:].isna().all()


@pytest.mark.parametrize(
    "mutation", ["leak", "duplicate_test", "threshold", "missing_fold"]
)
def test_invalid_fold_rejected(mutation):
    frame, report = fixture()
    report = copy.deepcopy(report)
    if mutation == "leak":
        report["validation"]["fold_subjects"][0]["train"].append("1")
    elif mutation == "duplicate_test":
        report["validation"]["fold_subjects"][1]["test"] = ["1"]
    elif mutation == "threshold":
        report["validation"]["fold_fog_thresholds"][0] = 0.4
    else:
        report["validation"]["fold_subjects"].pop()
    with pytest.raises(ValueError):
        validate_predictions(frame, report)


@pytest.mark.parametrize(
    "column,value",
    [
        ("fog_probability", float("nan")),
        ("fog_probability", 1.1),
        ("is_fog", "maybe"),
        ("truth_fog_label", 2),
        ("subject_id", 1.5),
        ("window_end_s", -1),
        ("fog_threshold", 0.9),
    ],
)
def test_invalid_prediction_rejected(column, value):
    frame, report = fixture()
    if isinstance(value, str):
        frame[column] = frame[column].astype(object)
    elif column == "subject_id":
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_predictions(frame, report)


def test_duplicate_window_rejected():
    frame, report = fixture()
    frame.loc[1, "window_start_s"] = 0
    with pytest.raises(ValueError):
        validate_predictions(frame, report)


@pytest.mark.parametrize(
    "mutation", [None, "label", "fraction", "end", "missing", "duplicate"]
)
def test_raw_alignment(mutation):
    frame, report = fixture()
    frame = validate_predictions(frame, report)
    windows = [
        SimpleNamespace(
            subject_id=r.subject_id,
            session_id=r.session_id,
            task_id=r.task_id,
            start_s=r.window_start_s,
            end_s=r.window_end_s,
            fog_label=r.truth_fog_label,
            fog_fraction=r.truth_fog_fraction,
        )
        for r in frame.itertuples()
    ]
    if mutation is None:
        assert validate_raw_windows(windows, frame) == 4
        return
    if mutation == "label":
        windows[0].fog_label = 0
    elif mutation == "fraction":
        windows[0].fog_fraction = 0.8
    elif mutation == "end":
        windows[0].end_s += 0.1
    elif mutation == "missing":
        windows.pop()
    else:
        windows.append(windows[0])
    with pytest.raises(ValueError):
        validate_raw_windows(windows, frame)
