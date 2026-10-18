"""Convert a canonical dataframe into deterministic overlapping windows."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fogguard.data.dataset import validate_frame
from fogguard.types import SensorWindow


@dataclass
class WindowReplay:
    frame: pd.DataFrame
    sample_rate_hz: float = 50.0
    window_seconds: float = 2.0
    step_seconds: float = 0.5
    fog_positive_fraction: float = 0.5
    turn_positive_fraction: float = 0.5
    sample_rate_tolerance_fraction: float = 0.05

    def windows(self) -> Iterable[SensorWindow]:
        if not 0.0 <= self.fog_positive_fraction <= 1.0:
            raise ValueError("fog_positive_fraction must be between 0 and 1")
        if not 0.0 <= self.turn_positive_fraction <= 1.0:
            raise ValueError("turn_positive_fraction must be between 0 and 1")
        if self.sample_rate_tolerance_fraction < 0.0:
            raise ValueError("sample_rate_tolerance_fraction must not be negative")

        frame = validate_frame(self.frame)
        timestamps = frame["timestamp_s"].to_numpy(dtype=float)
        if len(timestamps) > 1:
            observed_rate_hz = 1.0 / float(np.median(np.diff(timestamps)))
            relative_error = abs(observed_rate_hz - self.sample_rate_hz) / self.sample_rate_hz
            if relative_error > self.sample_rate_tolerance_fraction:
                raise ValueError(
                    "Observed sample rate does not match sample_rate_hz "
                    f"({observed_rate_hz:.3f} Hz vs {self.sample_rate_hz:.3f} Hz)"
                )

        window_n = max(1, round(self.sample_rate_hz * self.window_seconds))
        step_n = max(1, round(self.sample_rate_hz * self.step_seconds))
        for start in range(0, len(frame) - window_n + 1, step_n):
            chunk = frame.iloc[start : start + window_n]
            metadata: dict[str, str] = {}
            if "fog_label" in chunk.columns:
                fog_fraction = float(chunk["fog_label"].mean())
                metadata["fog_fraction"] = f"{fog_fraction:.6f}"
                metadata["fog_label"] = str(
                    int(fog_fraction >= self.fog_positive_fraction)
                )
            if "turn_label" in chunk.columns:
                counts = chunk["turn_label"].value_counts()
                top_count = int(counts.iloc[0])
                tied = int((counts == top_count).sum()) > 1
                turn_fraction = top_count / len(chunk)
                turn_label = str(counts.index[0])
                if tied or turn_fraction < self.turn_positive_fraction:
                    turn_label = "unknown"
                metadata["turn_fraction"] = f"{turn_fraction:.6f}"
                metadata["turn_label"] = turn_label
            if "scenario_id" in chunk.columns:
                scenario_ids = chunk["scenario_id"].astype(str).unique()
                metadata["scenario_id"] = (
                    scenario_ids[0] if len(scenario_ids) == 1 else "mixed"
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
