"""Continuous raw-event disagreements preserve causal holds and unknown input."""

import numpy as np
import pandas as pd
import pytest

from fogguard.evaluation.a1_disagreement import record_audit, segments, select_cases, timeline


def predictions(ends, scores):
    return pd.DataFrame({"subject_id": 1, "session_id": 1, "task_id": 1,
        "window_end_s": ends, "available_at_s": ends, "fog_probability": scores,
        "fog_threshold": 0.5, "is_fog": np.array(scores) >= 0.5})


def test_decisions_begin_at_end_expire_and_never_bridge_missing_windows():
    t = np.arange(240) / 60
    frame = predictions([119 / 60, 179 / 60], [0.8, 0.2])
    state, score = timeline(t, frame)
    assert np.all(state[:119] == -1) and np.all(state[119:149] == 1)
    assert np.all(state[149:179] == -1) and np.all(state[179:209] == 0)
    assert np.all(state[209:] == -1) and np.isnan(score[149:179]).all()
    with pytest.raises(ValueError, match="backdated"):
        timeline(t, frame.assign(available_at_s=0))


def test_events_loss_gain_partition_and_negative_record_are_all_preserved():
    n = 600
    raw = pd.DataFrame({"timestamp": np.arange(n) / 60, "subjectID": 1, "sessionID": 1, "taskID": 1, "fog": 0})
    raw.loc[120:479, "fog"] = 1  # six-second event
    ends = np.arange(0, n, 30) / 60
    base = predictions(ends, np.ones(len(ends)))
    candidate = predictions(ends, np.where((ends >= 4) & (ends < 5), 0.1, 0.9))
    negative = raw.assign(subjectID=2, fog=0)
    records, events, pieces, _ = record_audit(pd.concat([raw, negative]), base, candidate)
    assert len(records) == 2 and len(events) == 1
    assert events.iloc[0].lost_fog_s == pytest.approx(1)
    assert events.iloc[0].onset_lost_s == pytest.approx(1)
    assert events.iloc[0].interior_lost_s == pytest.approx(0)
    assert events.iloc[0].ending_lost_s == pytest.approx(0)
    assert records.iloc[1].support_s == 0 and records.iloc[1].truth_events == 0
    assert len(pieces.loc[pieces.kind.eq("lost_fog")]) == 1
    assert select_cases(events, pieces)[0]["case"] == "largest_fog_time_loss"


def test_raw_gaps_split_events_and_no_prediction_means_unknown():
    t = np.r_[np.arange(60) / 60, 2 + np.arange(60) / 60]
    assert segments(t, np.ones(len(t), bool)) == [(0, 60), (60, 120)]
    state, score = timeline(t, predictions([], []))
    assert np.all(state == -1) and np.isnan(score).all()


def test_sample_alignment_and_mismatched_support_are_rejected():
    t = np.arange(180) / 60
    with pytest.raises(ValueError, match="aligned"):
        timeline(t, predictions([1.001], [0.9]))
    raw = pd.DataFrame({"timestamp": t, "subjectID": 1, "sessionID": 1, "taskID": 1, "fog": 0})
    with pytest.raises(ValueError, match="identical support"):
        record_audit(raw, predictions([1], [0.8]), predictions([1.5], [0.8]))
