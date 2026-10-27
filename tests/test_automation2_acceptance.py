"""Regression gates must detect broken behaviors, even with valid command syntax."""

import json

import pandas as pd
import pytest

from fogguard.cueing.contracts import GuidanceConfig
from fogguard.cueing.replay import replay_guidance, summarize_guidance
from fogguard.cueing.scenarios import build_guidance_cases
from fogguard.evaluation.automation2 import confusion, evaluate_timeline
from fogguard.evaluation.automation2_acceptance import acceptance_summary, check_acceptance
from scripts.build_automation2_review import build_review


@pytest.mark.parametrize("value", [.9, -0.1, float("nan"), float("inf")])
def test_fractional_or_nonfinite_labels_are_not_silently_truncated(value):
    with pytest.raises(ValueError):
        confusion([value], [0])


def test_absent_evidence_cannot_pass_the_acceptance_gate(tmp_path):
    result = acceptance_summary(check_acceptance(tmp_path))
    assert not result["regression_gate_passed"]
    assert result["counts"]["FAIL"] > 0


def test_known_limitation_is_distinct_from_a_pass():
    result = acceptance_summary([{"status": "KNOWN_LIMITATION"}])
    assert result["regression_gate_passed"]
    assert result["counts"] == {"PASS": 0, "FAIL": 0, "KNOWN_LIMITATION": 1}


def test_valid_lifecycle_without_fading_fails_the_fading_gate(tmp_path):
    events, responses = build_guidance_cases()["successful_recovery"]
    config = GuidanceConfig(enable_fading=False)
    commands = replay_guidance(events, config=config, responses=responses)
    folder = tmp_path / "timelines/constructed_successful_recovery"
    folder.mkdir(parents=True)
    commands.to_csv(folder / "response.csv", index=False)
    (folder / "metrics.json").write_text(
        json.dumps({"response": {"config": config.as_dict()}}), encoding="utf-8")
    result = next(row for row in check_acceptance(tmp_path) if row["check"] == "fading")
    assert result["status"] == "FAIL"


def test_wrong_turn_targets_fail_gate_even_with_a_start_and_stop(tmp_path):
    events, _ = build_guidance_cases()["left_fog"]
    commands = replay_guidance(events)
    for index in commands.index:
        targets = json.loads(commands.loc[index, "guidance_targets"])
        for item in targets:
            item["yaw_deg"] = -abs(item["yaw_deg"])
        commands.loc[index, "guidance_targets"] = json.dumps(targets)
    folder = tmp_path / "timelines/constructed_left_fog"
    folder.mkdir(parents=True)
    commands.to_csv(folder / "response.csv", index=False)
    (folder / "metrics.json").write_text(
        json.dumps({"response": {"config": GuidanceConfig().as_dict()}}), encoding="utf-8")
    result = next(row for row in check_acceptance(tmp_path) if row["check"] == "direction_left_fog")
    assert result["status"] == "FAIL"


def test_no_negative_case_file_is_not_reported_as_no_false_alarms(tmp_path):
    folder = tmp_path / "timelines/sensor_normal"
    folder.mkdir(parents=True)
    pd.DataFrame({"guidance_action": []}).to_csv(folder / "response.csv", index=False)
    (folder / "metrics.json").write_text(
        json.dumps({"response": {"config": GuidanceConfig().as_dict()}}), encoding="utf-8")
    result = next(row for row in check_acceptance(tmp_path)
                  if row["check"] == "sensor_negative_normal")
    assert result["status"] == "FAIL"


def test_early_confirmation_is_distinct_from_later_guidance_coverage():
    events, _ = build_guidance_cases()["straight_fog"]
    events = events.copy()
    events.loc[events.window_end_s.lt(5.0), "truth_fog_label"] = 0
    events.loc[events.window_end_s.eq(4.5), "turn"] = "unknown"
    commands = replay_guidance(events)
    result = evaluate_timeline(events, commands)
    event = result["events"][0]
    assert not event["detected"]  # confirmation started before this truth run
    assert event["guidance_start_matched"]
    assert event["guidance_active_truth_windows"] > 0


def test_review_excludes_real_data_left_over_from_an_earlier_run(tmp_path):
    names = ["sensor_voluntary_stop", "sensor_fog_recovery"]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "case_names": names, "real_data_status": "not_run"}), encoding="utf-8")
    (tmp_path / "acceptance.json").write_text(json.dumps({
        "counts": {"PASS": 1, "FAIL": 0, "KNOWN_LIMITATION": 0},
        "regression_gate_passed": True,
        "checks": [{"check": "fixture", "status": "PASS", "meaning": "fixture"}]}),
        encoding="utf-8")
    events, _ = build_guidance_cases()["straight_fog"]
    commands = replay_guidance(events)
    for name in names:
        folder = tmp_path / "timelines" / name
        folder.mkdir(parents=True)
        commands.to_csv(folder / "response.csv", index=False)
        metrics = {"lifecycle": summarize_guidance(commands)}
        (folder / "metrics.json").write_text(json.dumps({"fixed": metrics, "response": metrics}),
                                             encoding="utf-8")
        commands.to_csv(folder / "fixed.csv", index=False)
    stale = tmp_path / "timelines/real_oof"
    stale.mkdir()
    (stale / "metrics.json").write_text("invalid stale report", encoding="utf-8")
    result = build_review(tmp_path)
    assert len(result["failure_packets"]) == 2
    assert result["playback_case_strategies"] == 4
