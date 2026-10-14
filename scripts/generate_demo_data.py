"""Generate deterministic synthetic IMU data for integration testing and demos."""

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 50
DURATION_S = 30


def main() -> None:
    rng = np.random.default_rng(5412)
    t = np.arange(0, DURATION_S, 1 / SAMPLE_RATE_HZ)
    fog = (t >= 12) & (t < 17)
    turn = (t >= 21) & (t < 25)
    gait = np.sin(2 * np.pi * 1.6 * t)
    tremble = fog * 0.8 * np.sin(2 * np.pi * 5.0 * t)
    noise = 0.04 * rng.standard_normal((len(t), 6))
    frame = pd.DataFrame(
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
    destination = Path("data/demo/synthetic_gait.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Wrote {destination} ({len(frame)} samples)")


if __name__ == "__main__":
    main()
