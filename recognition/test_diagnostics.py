import numpy as np
import pandas as pd
import pytest
from diagnostics import (
    annotate_windows,
    build_diagnostics,
    clustered_intervals,
    decision_intervals,
    event_rows,
    intersection_duration,
    majority,
    merge_intervals,
)


def signal(labels):
    n = len(labels)
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) / 60,
            "fog": labels,
            "fog_severity": np.ones(n),
            "subjectID": np.ones(n),
            "sessionID": np.ones(n),
            "taskID": np.ones(n),
            "activity": np.ones(n),
        }
    )


def predictions(ends, flags):
    return pd.DataFrame({"window_end_s": ends, "is_fog": flags})


def test_decision_never_backfills_input_window_or_long_gap():
    alarms, support = decision_intervals(predictions([2, 4], [True, True]), 5)
    assert alarms == support == [(2, 2.5), (4, 4.5)]
    assert intersection_duration(alarms, [(0, 2)]) == 0


def test_hold_stops_at_next_decision_and_record_end():
    alarms, support = decision_intervals(predictions([2, 2.2], [True, False]), 2.3)
    assert alarms == [(2, 2.2)]
    assert support == [(2, 2.3)]


def test_early_event_is_missed_even_with_later_positive_window():
    raw = signal([1] * 60 + [0] * 120)
    events, alarms, summary = event_rows(raw, predictions([2], [True]))
    assert len(events) == 1 and not events[0]["covered"]
    assert events[0]["model_support_s"] == 0
    assert not alarms[0]["overlaps_truth"]
    assert summary["false_alarm_s"] == pytest.approx(0.5)
    assert summary["missed_fog_s"] == pytest.approx(1)


def test_existing_alarm_is_not_new_detection():
    raw = signal([0] * 132 + [1] * 12 + [0] * 36)
    events, _, _ = event_rows(raw, predictions([2], [True]))
    assert events[0]["covered"]
    assert events[0]["alarm_already_active_at_onset"]
    assert events[0]["onset_delay_s"] == 0


def test_new_alarm_latency_uses_window_end():
    raw = signal([0] * 120 + [1] * 60)
    events, _, _ = event_rows(raw, predictions([2.25], [True]))
    assert events[0]["covered"]
    assert not events[0]["alarm_already_active_at_onset"]
    assert events[0]["onset_delay_s"] == pytest.approx(0.25)


def test_no_predictions_remains_uncovered():
    events, alarms, summary = event_rows(signal([1] * 180), predictions([], []))
    assert not alarms
    assert summary["model_support_s"] == 0
    assert not events[0]["covered"]


def test_no_truth_events_has_false_alarm_duration():
    events, alarms, summary = event_rows(signal([0] * 180), predictions([2], [True]))
    assert not events and len(alarms) == 1
    assert summary["truth_fog_s"] == 0
    assert summary["false_alarm_s"] == pytest.approx(0.5)


def test_boundary_touch_does_not_cover_event():
    events, _, _ = event_rows(signal([1] * 120 + [0] * 60), predictions([2], [True]))
    assert not events[0]["covered"]


def test_missing_timestamp_support_is_not_invented():
    raw = signal([0] * 180)
    raw = raw[(raw.timestamp < 2) | (raw.timestamp >= 2.5)]
    _, alarms, summary = event_rows(raw, predictions([2], [True]))
    assert not alarms
    assert summary["model_support_s"] == 0


def test_one_alarm_can_cover_two_events_but_not_gap_time_as_true():
    raw = signal([0] * 120 + [1] * 6 + [0] * 6 + [1] * 6 + [0] * 42)
    events, alarms, summary = event_rows(raw, predictions([2], [True]))
    assert len(events) == 2 and all(r["covered"] for r in events)
    assert len(alarms) == 1
    assert summary["true_alarm_s"] == pytest.approx(0.2)
    assert summary["false_alarm_s"] == pytest.approx(0.3)


def test_overlapping_intervals_not_double_counted():
    assert merge_intervals([(0, 2), (1, 3), (4, 4)]) == [(0.0, 3.0)]
    assert intersection_duration([(0, 2), (1, 3)], [(1, 2)]) == 1


def test_manifestation_tie_is_mixed_and_not_decided_by_prediction():
    raw = signal([1] * 120)
    raw.loc[:59, "fog_severity"] = 2
    pred = pd.DataFrame(
        {
            "subject_id": [1],
            "session_id": [1],
            "task_id": [1],
            "window_start_s": [0],
            "window_end_s": [119 / 60],
            "truth_fog_fraction": [1],
            "truth_fog_label": [1],
            "is_fog": [False],
        }
    )
    result = annotate_windows(raw, pred)
    assert result.truth_manifestation.iloc[0] == "混合表现"
    raw.loc[60, "fog_severity"] = 2
    assert annotate_windows(raw, pred).truth_manifestation.iloc[0] == "颤动 Trembling"
    pred.is_fog = True
    assert annotate_windows(raw, pred).truth_manifestation.iloc[0] == "颤动 Trembling"


def test_manifestation_uses_only_frozen_samples():
    raw = signal([1] * 60 + [0] * 60)
    raw.loc[60:, "fog_severity"] = 0
    pred = pd.DataFrame(
        {
            "subject_id": [1],
            "session_id": [1],
            "task_id": [1],
            "window_start_s": [0],
            "window_end_s": [119 / 60],
            "truth_fog_fraction": [0.5],
            "truth_fog_label": [1],
        }
    )
    assert annotate_windows(raw, pred).truth_manifestation.iloc[0] == "拖步 Shuffling"
    raw.loc[0, "fog_severity"] = 0
    with pytest.raises(ValueError):
        annotate_windows(raw, pred)


def test_annotation_rejects_misaligned_truth():
    pred = pd.DataFrame(
        {
            "subject_id": [1],
            "session_id": [1],
            "task_id": [1],
            "window_start_s": [0],
            "window_end_s": [119 / 60],
            "truth_fog_fraction": [0],
            "truth_fog_label": [0],
        }
    )
    with pytest.raises(ValueError):
        annotate_windows(signal([1] * 120), pred)


def test_cluster_bootstrap_is_deterministic_and_keeps_no_positive_undefined():
    frame = pd.DataFrame({"subject_id": [1, 1, 2, 2], "outcome": ["TN"] * 4})
    first = clustered_intervals(frame, repetitions=100)
    assert first == clustered_intervals(frame, repetitions=100)
    assert first["metrics"]["recall"]["low"] is None
    assert first["metrics"]["accuracy"]["low"] == 1


def test_unique_majority_is_required():
    assert majority([1, 1, 2, 3]) == 1
    assert majority([1, 1, 2, 3], strictly=True) is None
    assert majority([1, 2]) is None


def test_record_without_predictions_is_included_in_full_event_evaluation(tmp_path):
    from evaluate import metrics

    first = signal([0] * 120)
    second = signal([1] * 120)
    second.timestamp += 3
    second.taskID = 7
    raw = pd.concat([first, second], ignore_index=True)
    for pos in ["ankleL", "back"]:
        for kind in ["acc", "gyro"]:
            for axis in "xyz":
                raw[f"{pos}_{kind}_{axis}"] = 1.0
    raw.loc[raw.taskID == 7, "ankleL_acc_x"] = np.nan
    pred = pd.DataFrame(
        {
            "subject_id": [1],
            "session_id": [1],
            "task_id": [1],
            "window_start_s": [0],
            "window_end_s": [119 / 60],
            "truth_fog_fraction": [0],
            "truth_fog_label": [0],
            "is_fog": [False],
            "outcome": ["TN"],
            "fog_probability": [0.1],
        }
    )
    diagnostic, _ = build_diagnostics(raw, pred, metrics, tmp_path)
    assert diagnostic["event_summary"]["truth_events"] == 1
    assert diagnostic["event_summary"]["missed_events"] == 1
    event = diagnostic["missed_events"][0]
    assert event["record"] == 1 and event["closest_window"] is None
    assert event["support_reason"] == "整段无有效预测窗口"
    assert event["incomplete_imu_samples"] == 120
