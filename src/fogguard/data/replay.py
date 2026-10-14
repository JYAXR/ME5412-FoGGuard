"""Convert a canonical dataframe into deterministic overlapping windows."""

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from fogguard.types import SensorWindow


@dataclass
class WindowReplay:
    frame: pd.DataFrame
    sample_rate_hz: float = 50.0
    window_seconds: float = 2.0
    step_seconds: float = 0.5
    fog_positive_fraction: float = 0.5

    def windows(self) -> Iterable[SensorWindow]:
        if not 0.0 <= self.fog_positive_fraction <= 1.0:
            raise ValueError("fog_positive_fraction must be between 0 and 1")

        window_n = max(1, round(self.sample_rate_hz * self.window_seconds))
        step_n = max(1, round(self.sample_rate_hz * self.step_seconds))
        for start in range(0, len(self.frame) - window_n + 1, step_n):
            chunk = self.frame.iloc[start : start + window_n]
            metadata: dict[str, str] = {}
            if "fog_label" in chunk.columns:
                fog_fraction = float(chunk["fog_label"].mean())
                metadata["fog_fraction"] = f"{fog_fraction:.6f}"
                metadata["fog_label"] = str(
                    int(fog_fraction >= self.fog_positive_fraction)
                )
            window = SensorWindow(
                timestamp_s=chunk["timestamp_s"].to_numpy(dtype=float),
                accel_mps2=chunk[["ax", "ay", "az"]].to_numpy(dtype=float),
                gyro_dps=chunk[["gx", "gy", "gz"]].to_numpy(dtype=float),
                sample_rate_hz=self.sample_rate_hz,
                metadata=metadata,
            )
            window.validate()
            yield window
