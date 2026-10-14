"""Canonical CSV loader.

Required columns: timestamp_s, ax, ay, az, gx, gy, gz.
Optional labels such as fog_label and turn_label are preserved in metadata elsewhere.
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("timestamp_s", "ax", "ay", "az", "gx", "gy", "gz")


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if not frame["timestamp_s"].is_monotonic_increasing:
        raise ValueError("timestamp_s must be monotonically increasing")
    if "fog_label" in frame.columns and not frame["fog_label"].isin((0, 1)).all():
        raise ValueError("fog_label must contain only 0 or 1")
    return frame
