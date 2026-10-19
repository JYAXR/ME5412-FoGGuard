from pathlib import Path

import pytest

from fogguard.cueing import AdaptiveCueingController
from fogguard.pipeline import RESULT_COLUMNS, replay_frame
from scripts.generate_demo_data import build_scenario
from scripts.run_end_to_end import run_scenarios


@pytest.mark.parametrize(
    ("scenario", "expected_turn", "forbidden_turn"),
    [
        ("normal", "straight", "left"),
        ("straight_fog", "straight", "right"),
        ("turning_fog", "left", "right"),
        ("right_turning_fog", "right", "left"),
        ("normal_right_turn", "right", "left"),
    ],
)
def test_core_scenarios_reach_the_shared_pipeline(
    scenario: str,
    expected_turn: str,
    forbidden_turn: str,
) -> None:
    result = replay_frame(build_scenario(scenario))

    assert len(result) == 57
    assert expected_turn in set(result["turn"])
    assert forbidden_turn not in set(result["turn"])


def test_voluntary_stop_is_not_misclassified_as_turning() -> None:
    result = replay_frame(build_scenario("voluntary_stop"))

    assert set(result["turn"]) == {"straight"}
    assert set(result["truth_turn_label"]) == {"straight"}


def test_missing_gyroscope_safely_disables_cues() -> None:
    frame = build_scenario("straight_fog")
    frame["gyro_available"] = "false"

    result = replay_frame(frame, controller=AdaptiveCueingController(min_confidence=0.21))

    assert set(result["turn"]) == {"unknown"}
    assert set(result["turn_confidence"]) == {0.0}
    assert set(result["cue"]) == {"none"}
    fog_rows = result[result["fog_detected"]]
    assert set(fog_rows["cue_reason"]) == {"Turning context unavailable"}


def test_pipeline_exposes_ui_and_evaluation_contract() -> None:
    result = replay_frame(build_scenario("fog_recovery"))

    assert tuple(result.columns) == RESULT_COLUMNS
    assert result["window_start_s"].is_monotonic_increasing
    assert result["window_end_s"].is_monotonic_increasing
    assert result["turn_confidence"].between(0.0, 1.0).all()
    assert set(result["truth_fog_label"]) <= {"0", "1"}
    assert result["fog_matches_truth"].dtype == bool
    assert result["turn_matches_truth"].dtype == bool


def test_batch_runner_returns_machine_readable_scenario_ids(tmp_path: Path) -> None:
    scenarios = {}
    for scenario in ("normal", "right_turning_fog"):
        path = tmp_path / f"{scenario}.csv"
        build_scenario(scenario).to_csv(path, index=False)
        scenarios[scenario] = path

    result = run_scenarios(scenarios)

    assert len(result) == 114
    assert set(result["scenario_id"]) == set(scenarios)
    assert {"source_file", *RESULT_COLUMNS} <= set(result.columns)
    assert (result.loc[result["scenario_id"] == "right_turning_fog", "cue"] != "none").any()
