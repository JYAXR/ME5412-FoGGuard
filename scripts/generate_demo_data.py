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
    # Keep the original mixed timeline for the existing Streamlit default.
    "synthetic_gait": "synthetic_gait.csv",
}


def build_scenario(scenario: str) -> pd.DataFrame:
    if scenario not in SCENARIO_FILENAMES:
        raise ValueError(f"Unknown scenario: {scenario}")

    rng = np.random.default_rng(5412)
    t = np.arange(0, DURATION_S, 1 / SAMPLE_RATE_HZ)
    fog = np.zeros(len(t), dtype=bool)
    turn = np.zeros(len(t), dtype=bool)

    if scenario in {"straight_fog", "turning_fog", "synthetic_gait"}:
        fog = (t >= 12) & (t < 17)
    if scenario == "turning_fog":
        turn = (t >= 10) & (t < 19)
    elif scenario == "synthetic_gait":
        turn = (t >= 21) & (t < 25)

    gait = np.sin(2 * np.pi * 1.6 * t)
    tremble = fog * 0.8 * np.sin(2 * np.pi * 5.0 * t)
    noise = 0.04 * rng.standard_normal((len(t), 6))
    return pd.DataFrame(
        {
            "timestamp_s": t,
            "ax": 0.1 * gait + noise[:, 0],
            "ay": noise[:, 1],
            "az": 9.81 + gait + tremble + noise[:, 2],
            "gx": noise[:, 3],
            "gy": noise[:, 4],
            "gz": turn * 38.0 + noise[:, 5],
            "fog_label": fog.astype(int),
            "turn_label": np.where(turn, "left", "straight"),
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
