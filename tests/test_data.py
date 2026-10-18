from pathlib import Path

import pandas as pd
import pytest

from fogguard.data import WindowReplay, load_csv
from scripts.generate_demo_data import (
    SAMPLE_RATE_HZ,
    SCENARIO_FILENAMES,
    build_scenario,
    write_scenarios,
)


@pytest.mark.parametrize(
    (
        "scenario",
        "expected_fog_samples",
        "expected_left_samples",
        "expected_right_samples",
    ),
    [
        ("normal", 0, 0, 0),
        ("straight_fog", 250, 0, 0),
        ("turning_fog", 250, 450, 0),
        ("right_turning_fog", 250, 0, 450),
        ("normal_right_turn", 0, 0, 450),
        ("voluntary_stop", 0, 0, 0),
        ("fog_recovery", 250, 0, 0),
    ],
)
def test_scenario_labels(
    scenario: str,
    expected_fog_samples: int,
    expected_left_samples: int,
    expected_right_samples: int,
) -> None:
    frame = build_scenario(scenario)

    assert len(frame) == 30 * SAMPLE_RATE_HZ
    assert int(frame["fog_label"].sum()) == expected_fog_samples
    assert int((frame["turn_label"] == "left").sum()) == expected_left_samples
    assert int((frame["turn_label"] == "right").sum()) == expected_right_samples
    if scenario in {"turning_fog", "right_turning_fog"}:
        expected_turn = "left" if scenario == "turning_fog" else "right"
        assert (frame.loc[frame["fog_label"] == 1, "turn_label"] == expected_turn).all()


def test_scenarios_are_deterministic() -> None:
    pd.testing.assert_frame_equal(build_scenario("turning_fog"), build_scenario("turning_fog"))


def test_written_scenarios_load_and_replay(tmp_path: Path) -> None:
    write_scenarios(tmp_path)

    for filename in SCENARIO_FILENAMES.values():
        frame = load_csv(tmp_path / filename)
        windows = list(WindowReplay(frame).windows())

        assert len(windows) == 57
        assert windows[0].accel_mps2.shape == (100, 3)
        assert windows[0].gyro_dps.shape == (100, 3)


def test_replay_aligns_sample_labels_to_fog_windows() -> None:
    windows = list(WindowReplay(build_scenario("straight_fog")).windows())
    positive_windows = [window for window in windows if window.metadata["fog_label"] == "1"]

    assert len(positive_windows) == 11
    assert windows[0].metadata["fog_fraction"] == "0.000000"
    assert windows[0].metadata["fog_label"] == "0"
    assert windows[0].metadata["turn_label"] == "straight"
    assert max(float(window.metadata["fog_fraction"]) for window in windows) == 1.0


@pytest.mark.parametrize(
    ("scenario", "expected_turn"),
    [("turning_fog", "left"), ("right_turning_fog", "right")],
)
def test_replay_aggregates_window_turn_labels(
    scenario: str,
    expected_turn: str,
) -> None:
    windows = list(WindowReplay(build_scenario(scenario)).windows())

    assert any(window.metadata["turn_label"] == expected_turn for window in windows)
    assert all("turn_fraction" in window.metadata for window in windows)


def test_replay_preserves_optional_scenario_id() -> None:
    frame = build_scenario("normal")
    frame["scenario_id"] = "normal"

    first_window = next(WindowReplay(frame).windows())

    assert first_window.metadata["scenario_id"] == "normal"


def test_replay_accepts_unlabelled_data() -> None:
    frame = build_scenario("normal").drop(columns="fog_label")

    first_window = next(WindowReplay(frame).windows())

    assert "fog_label" not in first_window.metadata
    assert "fog_fraction" not in first_window.metadata


def test_load_csv_rejects_invalid_fog_labels(tmp_path: Path) -> None:
    frame = build_scenario("normal")
    frame.loc[0, "fog_label"] = 2
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="fog_label must contain only 0 or 1"):
        load_csv(path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("az", float("nan"), "finite numbers"),
        ("gx", float("inf"), "finite numbers"),
        ("turn_label", "clockwise", "turn_label must contain only"),
        ("scenario_id", "", "scenario_id must contain non-empty"),
    ],
)
def test_load_csv_rejects_unreliable_values(
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
) -> None:
    frame = build_scenario("normal")
    frame.loc[0, column] = value
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_csv(path)


def test_load_csv_requires_strictly_increasing_timestamps(tmp_path: Path) -> None:
    frame = build_scenario("normal")
    frame.loc[1, "timestamp_s"] = frame.loc[0, "timestamp_s"]
    path = tmp_path / "duplicate-time.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="strictly increasing"):
        load_csv(path)


def test_replay_rejects_sample_rate_mismatch() -> None:
    frame = build_scenario("normal")

    with pytest.raises(ValueError, match="Observed sample rate"):
        list(WindowReplay(frame, sample_rate_hz=100.0).windows())
