"""Generate deterministic synthetic IMU scenarios for testing and demos."""

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 50
DURATION_S = 30
SCENARIO_FILENAMES = {
    "normal": "normal_gait.csv",
    "straight_fog": "straight_fog.csv",
    "turning_fog": "turning_fog.csv",
    "right_turning_fog": "right_turning_fog.csv",
    "normal_right_turn": "normal_right_turn.csv",
    "voluntary_stop": "voluntary_stop.csv",
    "fog_recovery": "fog_recovery.csv",
    # Keep the original mixed timeline for the existing Streamlit default.
    "synthetic_gait": "synthetic_gait.csv",
}


def build_scenario(scenario: str) -> pd.DataFrame:
    if scenario not in SCENARIO_FILENAMES:
        raise ValueError(f"Unknown scenario: {scenario}")

    rng = np.random.default_rng(5412)
    t = np.arange(0, DURATION_S, 1 / SAMPLE_RATE_HZ)
    fog = np.zeros(len(t), dtype=bool)
    turn_label = np.full(len(t), "straight", dtype=object)
    pause = np.zeros(len(t), dtype=bool)
    recovery = np.zeros(len(t), dtype=bool)

    if scenario in {
        "straight_fog",
        "turning_fog",
        "right_turning_fog",
        "synthetic_gait",
    }:
        fog = (t >= 12) & (t < 17)
    elif scenario == "fog_recovery":
        fog = (t >= 8) & (t < 13)
        recovery = (t >= 13) & (t < 18)
    elif scenario == "voluntary_stop":
        pause = (t >= 12) & (t < 17)

    if scenario == "turning_fog":
        turn_label[(t >= 10) & (t < 19)] = "left"
    elif scenario in {"right_turning_fog", "normal_right_turn"}:
        turn_label[(t >= 10) & (t < 19)] = "right"
    elif scenario == "synthetic_gait":
        turn_label[(t >= 21) & (t < 25)] = "left"

    gait_amplitude = np.ones(len(t))
    gait_amplitude[pause] = 0.0
    gait_amplitude[recovery] = 0.6
    gait = gait_amplitude * np.sin(2 * np.pi * 1.6 * t)
    tremble = fog * 0.8 * np.sin(2 * np.pi * 5.0 * t)
    noise = 0.04 * rng.standard_normal((len(t), 6))
    yaw_rate = np.zeros(len(t))
    yaw_rate[turn_label == "left"] = 38.0
    yaw_rate[turn_label == "right"] = -38.0
    return pd.DataFrame(
        {
            "timestamp_s": t,
            "ax": 0.1 * gait + noise[:, 0],
            "ay": noise[:, 1],
            "az": 9.81 + gait + tremble + noise[:, 2],
            "gx": noise[:, 3],
            "gy": noise[:, 4],
            "gz": yaw_rate + noise[:, 5],
            "fog_label": fog.astype(int),
            "turn_label": turn_label,
        }
    )


def write_scenarios(destination_dir: Path = Path("data/demo")) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for scenario, filename in SCENARIO_FILENAMES.items():
        frame = build_scenario(scenario)
        destination = destination_dir / filename
        frame.to_csv(destination, index=False)
        print(f"Wrote {destination} ({len(frame)} samples, scenario={scenario})")


def main() -> None:
    write_scenarios()


if __name__ == "__main__":
    main()
