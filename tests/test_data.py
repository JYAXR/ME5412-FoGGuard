from pathlib import Path

import pandas as pd
import pytest

from fogguard.data import WindowReplay, load_csv
from scripts.generate_demo_data import SAMPLE_RATE_HZ, build_scenario, write_scenarios


@pytest.mark.parametrize(
    ("scenario", "expected_fog_samples", "expected_left_samples"),
    [
        ("normal", 0, 0),
        ("straight_fog", 250, 0),
        ("turning_fog", 250, 450),
    ],
)
def test_scenario_labels(
    scenario: str,
    expected_fog_samples: int,
    expected_left_samples: int,
) -> None:
    frame = build_scenario(scenario)

    assert len(frame) == 30 * SAMPLE_RATE_HZ
    assert int(frame["fog_label"].sum()) == expected_fog_samples
    assert int((frame["turn_label"] == "left").sum()) == expected_left_samples
    if scenario == "turning_fog":
        assert (frame.loc[frame["fog_label"] == 1, "turn_label"] == "left").all()


def test_scenarios_are_deterministic() -> None:
    pd.testing.assert_frame_equal(build_scenario("turning_fog"), build_scenario("turning_fog"))


def test_written_scenarios_load_and_replay(tmp_path: Path) -> None:
    write_scenarios(tmp_path)

    for filename in ("normal_gait.csv", "straight_fog.csv", "turning_fog.csv"):
        frame = load_csv(tmp_path / filename)
        windows = list(WindowReplay(frame).windows())

        assert len(windows) == 57
        assert windows[0].accel_mps2.shape == (100, 3)
        assert windows[0].gyro_dps.shape == (100, 3)


def test_replay_aligns_sample_labels_to_fog_windows() -> None:
    windows = list(WindowReplay(build_scenario("straight_fog")).windows())
    positive_windows = [window for window in windows if window.metadata["fog_label"] == "1"]

    assert len(positive_windows) == 11
    assert windows[0].metadata == {"fog_fraction": "0.000000", "fog_label": "0"}
    assert max(float(window.metadata["fog_fraction"]) for window in windows) == 1.0


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
