"""Validation helpers for explicitly simulated step-response inputs."""

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_RESPONSE_COLUMNS = (
    "scenario_id",
    "timestamp_s",
    "foot",
    "forward_m",
    "lateral_m",
    "response_status",
    "delay_s",
    "reliability",
    "source",
)
VALID_FEET = frozenset({"left", "right", "none"})
VALID_RESPONSE_STATUSES = frozenset(
    {"success", "delayed", "unable", "no_response", "missing_input"}
)


def validate_step_responses(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of simulated step-response records."""
    missing = sorted(set(REQUIRED_RESPONSE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Step responses are missing required columns: {missing}")

    validated = frame.copy()
    for column in ("timestamp_s", "forward_m", "lateral_m", "delay_s", "reliability"):
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    if validated["scenario_id"].isna().any() or validated["scenario_id"].astype(str).str.strip().eq("").any():
        raise ValueError("scenario_id must contain non-empty values")
    if validated["timestamp_s"].isna().any() or not np.isfinite(validated["timestamp_s"]).all():
        raise ValueError("timestamp_s must contain finite values")
    if (validated["timestamp_s"] < 0).any():
        raise ValueError("timestamp_s must not be negative")

    for scenario_id, group in validated.groupby("scenario_id", sort=False):
        timestamps = group["timestamp_s"].to_numpy(dtype=float)
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            raise ValueError(f"timestamp_s must increase within scenario {scenario_id}")

    if validated["foot"].isna().any() or not validated["foot"].isin(VALID_FEET).all():
        raise ValueError("foot must be left, right, or none")
    if validated["response_status"].isna().any() or not validated[
        "response_status"
    ].isin(VALID_RESPONSE_STATUSES).all():
        raise ValueError("response_status contains an unsupported value")
    if not validated["source"].eq("simulated").all():
        raise ValueError("All step-response rows must declare source=simulated")
    if validated["reliability"].isna().any() or not validated["reliability"].between(
        0.0, 1.0
    ).all():
        raise ValueError("reliability must be between 0 and 1")

    completed = validated["response_status"].isin({"success", "delayed"})
    if validated.loc[completed, "foot"].eq("none").any():
        raise ValueError("Completed responses require a left or right foot")
    coordinates = validated.loc[completed, ["forward_m", "lateral_m"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Completed responses require finite landing coordinates")
    delays = validated.loc[completed, "delay_s"].to_numpy(dtype=float)
    if not np.isfinite(delays).all() or (delays < 0).any():
        raise ValueError("Completed responses require a non-negative delay_s")
    if (validated.loc[validated["response_status"] == "delayed", "delay_s"] <= 0).any():
        raise ValueError("Delayed responses require delay_s greater than zero")

    return validated


def load_step_responses(path: str | Path) -> pd.DataFrame:
    """Load and validate a simulated step-response CSV."""
    return validate_step_responses(pd.read_csv(path))
