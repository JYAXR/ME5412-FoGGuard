"""Causal confirmation, exact decision support and training-only selection."""

import numpy as np
import pandas as pd
import pytest

from fogguard.evaluation.a1_event import EventSupport, choose_event_policy, grouped_inner_scores
from fogguard.modeling.a1_event import EventGate, EventPolicy, apply_event_policy


def sample():
    time = np.arange(240) / 60
    raw = pd.DataFrame({"subjectID": 1, "sessionID": 1, "taskID": 1,
                        "timestamp": time, "fog": ((time >= 2) & (time < 3)).astype(int)})
    missing = raw.copy()
    missing["taskID"], missing["timestamp"], missing["fog"] = 2, time + 10, 1
    raw = pd.concat([raw, missing], ignore_index=True)
    frame = pd.DataFrame({"subject_id": 1, "session_id": 1, "task_id": 1,
                          "window_start_s": np.arange(5) * 0.5,
                          "window_end_s": 119 / 60 + np.arange(5) * 0.5,
                          "fog_probability": [0.1, 0.6, 0.9, 0.4, 0.1],
                          "truth_fog_label": [0, 1, 1, 0, 0]})
    return raw, frame


def test_confirmation_is_causal_and_resets_without_backdating():
    gate = EventGate(EventPolicy(0.3, 2))
    assert not gate.push(0.7, (1, 1, 1), 2)
    assert gate.push(0.8, (1, 1, 1), 2.5)
    assert not gate.push(0.1, (1, 1, 1), 3)
    assert not gate.push(0.9, (1, 1, 1), 3.5)
    assert not gate.push(0.9, (1, 1, 1), 5)
    assert not gate.push(0.9, (1, 1, 2), 5.5)
    assert gate.push(0.9, (1, 1, 2), 6)


def test_gate_preserves_raw_threshold_decision_and_needs_no_truth():
    _, frame = sample()
    policy = EventPolicy(0.3, 2)
    result = apply_event_policy(frame.drop(columns="truth_fog_label"), policy)
    assert result.is_fog.tolist() == [False, True, True, True, False]
    assert result.event_fog.tolist() == [False, False, True, True, False]
    future = frame.copy()
    future.loc[3:, "fog_probability"] = 1
    pd.testing.assert_frame_equal(apply_event_policy(frame, policy).iloc[:3],
                                   apply_event_policy(future, policy).iloc[:3])


def test_support_counts_missing_record_and_exact_partial_sample_overlap():
    raw, frame = sample()
    support = EventSupport.build(raw, frame, allowed=[1])
    positive = np.array([False, True, True, False, False])
    metrics = support.metrics(positive)
    assert metrics["truth_events"] == 2 and metrics["supported_events"] == 1
    assert metrics["covered_raw_events"] == 1
    assert metrics["true_alarm_s"] == pytest.approx(31 / 60)
    assert metrics["false_alarm_s"] == pytest.approx(29 / 60)
    assert support.truth_s == pytest.approx(5)
    assert support.metrics(np.zeros(5, bool))["covered_raw_events"] == 0


def test_raw_test_people_and_wrong_allowed_people_rejected():
    raw, frame = sample()
    with pytest.raises(ValueError, match="leakage"):
        EventSupport.build(raw, frame, allowed=[1], forbidden=[1])
    wrong = raw.copy()
    wrong.loc[0, "subjectID"] = 2
    with pytest.raises(ValueError, match="leakage"):
        EventSupport.build(wrong, frame, allowed=[1])


def test_policy_selection_respects_training_duration_budget_and_coverage():
    raw, frame = sample()
    support = EventSupport.build(raw, frame, allowed=[1])
    cfg = {"threshold_grid": {"start": 0.1, "stop": 0.9, "count": 9},
           "confirmation_choices": [1, 2], "false_duration_budget_multiplier": 1.1,
           "require_control_time_coverage": True}
    audit = choose_event_policy(frame, support, cfg)
    assert len(audit["options"]) == 18
    assert audit["selected"]["false_alarm_s"] <= audit["false_alarm_budget_s"] + 1e-6
    assert audit["selected"]["true_alarm_s"] >= audit["control"]["true_alarm_s"] - 1e-6
    assert audit["selected"]["covered_raw_events"] >= audit["control"]["covered_raw_events"]


def test_inner_scores_isolate_people_and_reject_forbidden():
    rng = np.random.default_rng(5412)
    x, y, groups = rng.normal(size=(48, 4)), np.tile([0, 1], 24), np.repeat(np.arange(6), 8)
    cfg = {"inner_folds": 3, "random_state": 5412,
           "rf": {"n_estimators": 6, "min_samples_leaf": 2, "class_weight": "balanced_subsample"}}
    scores, folds = grouped_inner_scores(x, y, groups, cfg)
    assert np.isfinite(scores).all()
    assert sorted(s for f in folds for s in f["validation"]) == list(map(str, range(6)))
    assert all(not set(f["train"]) & set(f["validation"]) for f in folds)
    with pytest.raises(ValueError, match="test people"):
        grouped_inner_scores(x, y, groups, cfg, forbidden=[0])


def test_bad_policy_and_out_of_order_input_rejected():
    with pytest.raises(ValueError, match="threshold"):
        EventPolicy(float("nan"))
    with pytest.raises(ValueError, match="Confirmation"):
        EventPolicy(0.3, 0)
    gate = EventGate(EventPolicy(0.3))
    gate.push(0.4, (1, 1, 1), 2)
    with pytest.raises(ValueError, match="chronological"):
        gate.push(0.4, (1, 1, 1), 2)
