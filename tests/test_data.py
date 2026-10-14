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
