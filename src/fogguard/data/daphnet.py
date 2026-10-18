"""Load the public Daphnet FoG dataset without inventing sensor measurements."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.data.dataset import validate_frame

DAPHNET_SAMPLE_RATE_HZ = 64.0
MG_TO_MPS2 = 9.80665 / 1000.0
SENSOR_COLUMNS = {
    "ankle": ("ankle_forward_mg", "ankle_lateral_mg", "ankle_vertical_mg"),
    "thigh": ("thigh_forward_mg", "thigh_lateral_mg", "thigh_vertical_mg"),
    "trunk": ("trunk_forward_mg", "trunk_lateral_mg", "trunk_vertical_mg"),
}
RAW_COLUMNS = (
    "timestamp_ms",
    "ankle_forward_mg",
    "ankle_vertical_mg",
    "ankle_lateral_mg",
    "thigh_forward_mg",
    "thigh_vertical_mg",
    "thigh_lateral_mg",
    "trunk_forward_mg",
    "trunk_vertical_mg",
    "trunk_lateral_mg",
    "annotation",
)


def discover_daphnet_files(root: str | Path) -> list[Path]:
    """Return the Daphnet recording files below a dataset or extraction root."""
    root_path = Path(root)
    files = sorted(root_path.rglob("S??R??.txt"))
    if not files:
        raise FileNotFoundError(f"No Daphnet S??R??.txt files found below {root_path}")
    return files


def _recording_identity(path: Path) -> tuple[str, str]:
    match = re.fullmatch(r"S(?P<subject>\d{2})R(?P<run>\d{2})", path.stem)
    if match is None:
        raise ValueError(f"Unexpected Daphnet recording name: {path.name}")
    return f"S{match.group('subject')}", path.stem


def _load_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", names=RAW_COLUMNS, header=None)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Daphnet file contains non-finite values: {path}")
    annotations = numeric["annotation"]
    if not annotations.isin((0, 1, 2)).all():
        raise ValueError(f"Daphnet annotation must contain only 0, 1, or 2: {path}")
    timestamps = numeric["timestamp_ms"].to_numpy(dtype=float)
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"Daphnet timestamps must be strictly increasing: {path}")
    return numeric


def load_daphnet_segments(
    path: str | Path,
    *,
    sensor_position: str = "ankle",
) -> list[pd.DataFrame]:
    """Convert one recording into contiguous canonical, labelled segments.

    Daphnet has accelerometers but no gyroscopes. Canonical gyro columns are
    zero placeholders and every output row is marked ``gyro_available=false``.
    The turning recognizer rejects such windows so zeros cannot be mistaken for
    measured angular velocity.
    """
    if sensor_position not in SENSOR_COLUMNS:
        choices = ", ".join(sorted(SENSOR_COLUMNS))
        raise ValueError(f"sensor_position must be one of: {choices}")

    source_path = Path(path)
    raw = _load_raw(source_path)
    subject_id, recording_id = _recording_identity(source_path)
    valid = raw["annotation"].isin((1, 2))
    segment_ids = valid.ne(valid.shift(fill_value=False)).cumsum()
    forward, lateral, vertical = SENSOR_COLUMNS[sensor_position]

    segments: list[pd.DataFrame] = []
    for segment_index, (_, segment) in enumerate(raw[valid].groupby(segment_ids[valid]), 1):
        timestamp_s = segment["timestamp_ms"].to_numpy(dtype=float) / 1000.0
        frame = pd.DataFrame(
            {
                "timestamp_s": timestamp_s,
                "ax": segment[forward].to_numpy(dtype=float) * MG_TO_MPS2,
                "ay": segment[lateral].to_numpy(dtype=float) * MG_TO_MPS2,
                "az": segment[vertical].to_numpy(dtype=float) * MG_TO_MPS2,
                "gx": np.zeros(len(segment), dtype=float),
                "gy": np.zeros(len(segment), dtype=float),
                "gz": np.zeros(len(segment), dtype=float),
                "fog_label": (segment["annotation"].to_numpy(dtype=int) == 2).astype(int),
                "scenario_id": f"{recording_id}_segment_{segment_index:02d}",
                "subject_id": subject_id,
                "recording_id": recording_id,
                "sensor_position": sensor_position,
                "source_dataset": "daphnet",
                "source_license": "CC-BY-4.0",
                "gyro_available": "false",
            }
        )
        segments.append(validate_frame(frame))
    return segments


def load_daphnet_dataset(
    root: str | Path,
    *,
    sensor_position: str = "ankle",
) -> dict[str, list[pd.DataFrame]]:
    """Load all Daphnet recordings grouped by participant."""
    by_subject: dict[str, list[pd.DataFrame]] = {}
    for path in discover_daphnet_files(root):
        subject_id, _ = _recording_identity(path)
        by_subject.setdefault(subject_id, []).extend(
            load_daphnet_segments(path, sensor_position=sensor_position)
        )
    return by_subject
