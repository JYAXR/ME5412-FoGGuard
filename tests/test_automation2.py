"""Independent evaluation checks with hand-computed expected outcomes."""

import json

import pandas as pd
import pytest

from fogguard.cueing.replay import replay_guidance
from fogguard.cueing.scenarios import build_guidance_cases
from fogguard.evaluation.automation2 import (
    audit_commands,
    confusion,
    evaluate_timeline,
    match_onsets,
    truth_runs,
)
from scripts.run_automation2_evaluation import sensor_candidates, validate_oof_provenance


def test_confusion_counts_and_missing_denominators():
    result = confusion([1, 1, 0, 0], [1, 0, 1, 0])
    assert [result[k] for k in ("tp", "tn", "fp", "fn")] == [1, 1, 1, 1]
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert confusion([0, 0], [0, 0])["recall"] is None
    with pytest.raises(ValueError):
        confusion([0], [2])


def test_pre_onset_false_alarm_cannot_count_as_a_detected_event():
    result = match_onsets([{"start_s": 5.0, "end_s": 6.0}], [4.0])
    assert result["hits"] == 0
    assert result["misses"] == result["unmatched_starts"] == 1


def test_event_matching_is_one_to_one_and_honors_grace():
    truth = [{"start_s": 5.0, "end_s": 6.0}, {"start_s": 7.0, "end_s": 8.0}]
    result = match_onsets(truth, [7.5])
    assert result["hits"] == 1
    assert result["mean_delay_s"] == 2.5
    assert match_onsets(truth, [10.1])["hits"] == 0


def test_truth_runs_split_on_negative_and_missing_time():
    group = pd.DataFrame({"window_end_s": [2., 2.5, 3., 5., 5.5],
                          "truth_fog_label": [0, 1, 1, 1, 0]})
    assert truth_runs(group) == [{"start_s": 2.5, "end_s": 3.},
                                 {"start_s": 5., "end_s": 5.}]
    with pytest.raises(ValueError):
        truth_runs(group.iloc[::-1])


def test_terminal_stop_is_censored_and_not_recovery():
    events, _ = build_guidance_cases()["straight_fog"]
    events = events.iloc[:10].copy()
    commands = replay_guidance(events)
    report = evaluate_timeline(events, commands)
    assert report["terminal_stops"] == 1
    assert report["fade_completed_stops"] == report["upstream_recovery_stops"] == 0
    assert report["raw"]["windows"] == len(events)
    assert audit_commands(commands) == []


def test_lifecycle_auditor_catches_clock_reset_and_leaking_stop():
    events, _ = build_guidance_cases()["straight_fog"]
    commands = replay_guidance(events.iloc[:10])
    bad = commands.copy()
    hold = bad.index[bad.guidance_action.eq("hold")][0]
    bad.loc[hold, "guidance_cue_started_at_s"] += 1
    bad.loc[bad.index[-1], "guidance_audio_enabled"] = True
    codes = {row["code"] for row in audit_commands(bad)}
    assert {"episode_clock_reset", "disabled_output_still_active"} <= codes


def test_lifecycle_checks_do_not_merge_identical_ids_across_subjects():
    events, _ = build_guidance_cases()["straight_fog"]
    one = replay_guidance(events)
    two = one.copy()
    two["subject_id"] = "another-subject"
    assert audit_commands(pd.concat([one, two], ignore_index=True)) == []


def test_evaluation_rejects_misaligned_command_rows():
    events, _ = build_guidance_cases()["straight_fog"]
    commands = replay_guidance(events)
    commands.loc[0, "window_end_s"] += 0.01
    with pytest.raises(ValueError, match="alignment"):
        evaluate_timeline(events, commands)


def test_adapter_does_not_use_truth_to_build_candidates():
    frame = pd.DataFrame({"fog_detected": [True], "turn": ["left"],
                          "turn_confidence": [.9], "truth_fog_label": [0]})
    first = sensor_candidates(frame, "test")
    frame["truth_fog_label"] = 1
    second = sensor_candidates(frame, "test")
    pd.testing.assert_frame_equal(first.drop(columns="truth_fog_label"),
                                  second.drop(columns="truth_fog_label"))


def test_oof_report_rejects_subject_leakage():
    predictions = pd.DataFrame({"subject_id": ["A", "B"]})
    report = {"validation": {"fold_subjects": [
        {"train": ["B"], "test": ["A"]}, {"train": ["A"], "test": ["B"]}]},
        "dataset": {"windows": 2}}
    validate_oof_provenance(predictions, report)
    leaked = json.loads(json.dumps(report))
    leaked["validation"]["fold_subjects"][0]["train"].append("A")
    with pytest.raises(ValueError, match="leakage"):
        validate_oof_provenance(predictions, leaked)
