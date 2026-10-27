"""Load and window the public FoG-STAR multi-IMU dataset.

FoG-STAR keeps activity and freezing labels separate.  This module preserves
that distinction so a model can learn direction during a freezing episode
instead of forcing the two targets into one mutually exclusive label.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FOGSTAR_SAMPLE_RATE_HZ = 60.0
FOGSTAR_SENSOR_POSITIONS = ("ankleL", "ankleR", "back", "wrist")
FOGSTAR_ACTIVITY_TO_DIRECTION = {1: "straight", 6: "right", 7: "left"}
FOGSTAR_DIRECTION_LABELS = frozenset({"straight", "left", "right", "unknown"})


@dataclass(frozen=True)
class FogStarWindow:
    """One synchronized multi-IMU window and its independent targets."""

    samples: NDArray[np.float64]
    channel_names: tuple[str, ...]
    sample_rate_hz: float
    start_s: float
    end_s: float
    subject_id: str
    session_id: str
    task_id: str
    fog_label: int | None = None
    fog_fraction: float | None = None
    direction_label: str | None = None
    direction_fraction: float | None = None
    activity_label: int | None = None
    activity_fraction: float | None = None

    def validate(self) -> None:
        if self.samples.ndim != 2 or self.samples.shape[1] != len(self.channel_names):
            raise ValueError("samples must be an NxC array matching channel_names")
        if len(self.samples) == 0 or not np.isfinite(self.samples).all():
            raise ValueError("samples must contain finite values")
        if self.sample_rate_hz <= 0 or self.end_s <= self.start_s:
            raise ValueError("window timing and sample rate must be positive")
        if self.fog_label not in {None, 0, 1}:
            raise ValueError("fog_label must be 0, 1, or None")
        if self.direction_label not in FOGSTAR_DIRECTION_LABELS | {None}:
            raise ValueError("direction_label is invalid")
        if self.activity_label is not None and self.activity_label not in range(8):
            raise ValueError("activity_label is invalid")


def sensor_columns(sensor_positions: Iterable[str]) -> tuple[str, ...]:
    positions = tuple(sensor_positions)
    unknown = sorted(set(positions) - set(FOGSTAR_SENSOR_POSITIONS))
    if unknown:
        choices = ", ".join(FOGSTAR_SENSOR_POSITIONS)
        raise ValueError(f"Unknown FoG-STAR sensor positions {unknown}; choose from {choices}")
    if not positions:
        raise ValueError("At least one sensor position is required")
    return tuple(
        f"{position}_{sensor}_{axis}"
        for position in positions
        for sensor in ("acc", "gyro")
        for axis in ("x", "y", "z")
    )


def load_fogstar_frame(
    path: str | Path,
    *,
    sensor_positions: Iterable[str] = ("ankleL", "back"),
    require_labels: bool = True,
) -> pd.DataFrame:
    """Load only the requested signals and validate the available labels."""
    channels = sensor_columns(sensor_positions)
    available = set(pd.read_csv(path, nrows=0).columns)
    required = {"timestamp", *channels}
    if require_labels:
        required.update({"activity", "fog", "subjectID", "sessionID", "taskID"})
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"FoG-STAR data is missing required columns: {missing}")

    optional = ("activity", "fog", "fog_severity", "subjectID", "sessionID", "taskID")
    usecols = ["timestamp", *channels, *(name for name in optional if name in available)]
    frame = pd.read_csv(path, usecols=usecols)
    numeric = frame.apply(pd.to_numeric, errors="coerce")

    if numeric["timestamp"].isna().any():
        raise ValueError("FoG-STAR timestamps must be numeric")
    timestamps = numeric["timestamp"].to_numpy(dtype=float)
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("FoG-STAR timestamps must be strictly increasing")
    if "fog" in numeric and (numeric["fog"].isna().any() or not numeric["fog"].isin((0, 1)).all()):
        raise ValueError("FoG-STAR fog labels must contain only 0 or 1")
    if "activity" in numeric:
        valid_activities = set(range(8))
        values = set(numeric["activity"].dropna().astype(int).unique())
        if numeric["activity"].isna().any() or not values <= valid_activities:
            raise ValueError("FoG-STAR activity labels must contain integers from 0 to 7")
    return numeric


def _fill_short_gaps(series: pd.Series, max_gap_samples: int) -> pd.Series:
    missing = series.isna()
    if not missing.any() or max_gap_samples <= 0:
        return series
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_lengths = missing.groupby(groups).transform("sum")
    fillable = missing & run_lengths.le(max_gap_samples)
    interpolated = series.interpolate(method="linear", limit_area="inside")
    result = series.copy()
    result.loc[fillable] = interpolated.loc[fillable]
    return result


def _contiguous_runs(valid: NDArray[np.bool_], timestamps: NDArray[np.float64]) -> Iterable[slice]:
    starts: list[int] = []
    max_step_s = 1.5 / FOGSTAR_SAMPLE_RATE_HZ
    for index, is_valid in enumerate(valid):
        starts_new = is_valid and (
            index == 0
            or not valid[index - 1]
            or timestamps[index] - timestamps[index - 1] > max_step_s
        )
        if starts_new:
            starts.append(index)
        ends_run = is_valid and (
            index == len(valid) - 1
            or not valid[index + 1]
            or timestamps[index + 1] - timestamps[index] > max_step_s
        )
        if ends_run:
            yield slice(starts.pop(), index + 1)


def _direction_target(activity: NDArray[np.int64], threshold: float) -> tuple[str, float]:
    labels = np.array([FOGSTAR_ACTIVITY_TO_DIRECTION.get(int(value), "unknown") for value in activity])
    counts = pd.Series(labels).value_counts()
    top_count = int(counts.iloc[0])
    fraction = top_count / len(labels)
    tied = int((counts == top_count).sum()) > 1
    label = str(counts.index[0])
    if tied or fraction < threshold or label == "unknown":
        label = "unknown"
    return label, fraction


def _activity_target(activity: NDArray[np.int64], threshold: float) -> tuple[int | None, float]:
    counts = pd.Series(activity).value_counts()
    top_count = int(counts.iloc[0])
    fraction = top_count / len(activity)
    tied = int((counts == top_count).sum()) > 1
    if tied or fraction < threshold:
        return None, fraction
    return int(counts.index[0]), fraction


def iter_fogstar_windows(
    frame: pd.DataFrame,
    *,
    sensor_positions: Iterable[str] = ("ankleL", "back"),
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    fog_positive_fraction: float = 0.5,
    direction_positive_fraction: float = 0.5,
    max_interpolation_gap_s: float = 0.1,
) -> Iterable[FogStarWindow]:
    """Yield valid windows without crossing subject, session, or task boundaries."""
    channels = sensor_columns(sensor_positions)
    missing = sorted({"timestamp", *channels} - set(frame.columns))
    if missing:
        raise ValueError(f"FoG-STAR frame is missing required columns: {missing}")
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be positive")
    if not 0 <= fog_positive_fraction <= 1 or not 0 <= direction_positive_fraction <= 1:
        raise ValueError("label fractions must be between 0 and 1")

    prepared = frame.copy()
    for column, default in (("subjectID", "new"), ("sessionID", "1"), ("taskID", "0")):
        if column not in prepared:
            prepared[column] = default

    key_change = prepared[["subjectID", "sessionID", "taskID"]].ne(
        prepared[["subjectID", "sessionID", "taskID"]].shift()
    ).any(axis=1)
    group_ids = key_change.cumsum()
    window_n = max(1, round(FOGSTAR_SAMPLE_RATE_HZ * window_seconds))
    step_n = max(1, round(FOGSTAR_SAMPLE_RATE_HZ * step_seconds))
    max_gap_samples = round(FOGSTAR_SAMPLE_RATE_HZ * max_interpolation_gap_s)
    for _, recording in prepared.groupby(group_ids, sort=False):
        recording = recording.copy()
        for channel in channels:
            recording[channel] = _fill_short_gaps(recording[channel], max_gap_samples)
        timestamps = recording["timestamp"].to_numpy(dtype=float)
        valid = np.isfinite(recording.loc[:, channels].to_numpy(dtype=float)).all(axis=1)

        for run_slice in _contiguous_runs(valid, timestamps):
            run = recording.iloc[run_slice]
            if len(run) < window_n:
                continue
            signals = run.loc[:, channels].to_numpy(dtype=float)
            run_timestamps = run["timestamp"].to_numpy(dtype=float)

            for start in range(0, len(run) - window_n + 1, step_n):
                stop = start + window_n
                chunk = run.iloc[start:stop]
                fog_label: int | None = None
                fog_fraction: float | None = None
                if "fog" in chunk:
                    fog_fraction = float(chunk["fog"].mean())
                    fog_label = int(fog_fraction >= fog_positive_fraction)

                direction_label: str | None = None
                direction_fraction: float | None = None
                activity_label: int | None = None
                activity_fraction: float | None = None
                if "activity" in chunk:
                    activity = chunk["activity"].to_numpy(dtype=int)
                    direction_label, direction_fraction = _direction_target(
                        activity, direction_positive_fraction
                    )
                    activity_label, activity_fraction = _activity_target(
                        activity, direction_positive_fraction
                    )

                window = FogStarWindow(
                    samples=signals[start:stop].astype(float, copy=False),
                    channel_names=channels,
                    sample_rate_hz=FOGSTAR_SAMPLE_RATE_HZ,
                    start_s=float(run_timestamps[start]),
                    end_s=float(run_timestamps[stop - 1]),
                    subject_id=str(chunk["subjectID"].iloc[0]),
                    session_id=str(chunk["sessionID"].iloc[0]),
                    task_id=str(chunk["taskID"].iloc[0]),
                    fog_label=fog_label,
                    fog_fraction=fog_fraction,
                    direction_label=direction_label,
                    direction_fraction=direction_fraction,
                    activity_label=activity_label,
                    activity_fraction=activity_fraction,
                )
                window.validate()
                yield window


def load_fogstar_windows(
    path: str | Path,
    *,
    sensor_positions: Iterable[str] = ("ankleL", "back"),
    require_labels: bool = True,
    **window_options: object,
) -> list[FogStarWindow]:
    positions = tuple(sensor_positions)
    frame = load_fogstar_frame(
        path,
        sensor_positions=positions,
        require_labels=require_labels,
    )
    return list(iter_fogstar_windows(frame, sensor_positions=positions, **window_options))
