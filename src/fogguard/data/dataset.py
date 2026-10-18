"""Canonical CSV loader.

Required columns: timestamp_s, ax, ay, az, gx, gy, gz.
Optional labels such as fog_label and turn_label are preserved in metadata elsewhere.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp_s", "ax", "ay", "az", "gx", "gy", "gz")
VALID_TURN_LABELS = frozenset({"left", "right", "straight", "unknown"})


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of a canonical replay dataframe."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    validated = frame.copy()
    numeric = validated.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Required sensor and timestamp columns must be finite numbers")
    validated.loc[:, REQUIRED_COLUMNS] = numeric

    timestamps = numeric["timestamp_s"].to_numpy(dtype=float)
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamp_s must be strictly increasing")

    if "fog_label" in validated.columns:
        fog_labels = pd.to_numeric(validated["fog_label"], errors="coerce")
        if fog_labels.isna().any() or not fog_labels.isin((0, 1)).all():
            raise ValueError("fog_label must contain only 0 or 1")
        validated["fog_label"] = fog_labels.astype(int)

    if "turn_label" in validated.columns:
        turn_labels = validated["turn_label"]
        if turn_labels.isna().any() or not turn_labels.isin(VALID_TURN_LABELS).all():
            allowed = ", ".join(sorted(VALID_TURN_LABELS))
            raise ValueError(f"turn_label must contain only: {allowed}")

    if "scenario_id" in validated.columns:
        scenario_ids = validated["scenario_id"]
        if scenario_ids.isna().any() or scenario_ids.astype(str).str.strip().eq("").any():
            raise ValueError("scenario_id must contain non-empty values")

    return validated


def load_csv(path: str | Path) -> pd.DataFrame:
    return validate_frame(pd.read_csv(path))
