"""Sensor validity, time and provenance for the uncalibrated motion proxy."""

import numpy as np
import pandas as pd
import pytest

from fogguard.motion.proxy_feedback import GYRO, estimate_motion_feedback


def signal(values):
    n = len(values)
    result = pd.DataFrame({"timestamp": np.arange(n)/60, "subjectID": 1,
                           "sessionID": 1, "taskID": 1})
    for name in GYRO:
        result[name] = values
    return result


def test_low_high_and_uncertain_without_asserting_a_step():
    raw = signal(np.r_[np.zeros(30), np.full(30, 20), np.full(30, 6)])
    output = estimate_motion_feedback(raw)
    assert output.feedback_state.tolist() == ["no_movement_evidence", "movement_evidence", "unknown"]
    assert output.feedback_valid.tolist() == [True, True, False]
    assert output.feedback_source.eq("derived_proxy").all()
    assert output.feedback_kind.eq("movement_only").all()
    assert np.allclose(output.observed_duration_s, [29/60, 29/60, 0])
    assert np.allclose(output.signal_span_s, 29/60)
    assert (output.feedback_available_at_s == output.feedback_window_end_s).all()
    assert not {"foot", "forward_m", "lateral_m"} & set(output)


def test_missing_incomplete_and_gap_become_unknown_without_filling():
    raw = signal(np.zeros(75))
    raw.loc[2, GYRO[0]] = np.nan
    raw.loc[40:, "timestamp"] += .1
    out = estimate_motion_feedback(raw)
    assert out.feedback_state.tolist() == ["unknown", "unknown", "unknown"]
    assert out.feedback_reason.tolist() == ["missing_gyro", "incomplete_or_irregular_window",
                                             "incomplete_or_irregular_window"]
    assert out.observed_duration_s.eq(0).all()


def test_missing_channel_unknown_and_reversed_or_duplicate_times_rejected():
    raw = signal(np.zeros(60))
    assert estimate_motion_feedback(raw.drop(columns=GYRO[0])).feedback_state.eq("unknown").all()
    raw.loc[31, "timestamp"] = raw.loc[30, "timestamp"]
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_motion_feedback(raw)


def test_interleaved_records_rejected_and_labels_do_not_change_result():
    raw = signal(np.zeros(60))
    baseline = estimate_motion_feedback(raw)
    with_labels = estimate_motion_feedback(raw.assign(fog=1, activity=7))
    pd.testing.assert_frame_equal(baseline, with_labels)
    raw.loc[30:44, "subjectID"] = 2
    with pytest.raises(ValueError, match="interleaved"):
        estimate_motion_feedback(raw)
