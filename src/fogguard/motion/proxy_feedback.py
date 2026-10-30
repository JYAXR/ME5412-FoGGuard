"""Causal bilateral gyro activity proxy; it does not infer a step or displacement."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ("subject_id", "session_id", "task_id")
RAW_KEYS = ("subjectID", "sessionID", "taskID")
GYRO = tuple(f"{side}_gyro_{axis}" for side in ("ankleL", "ankleR") for axis in "xyz")


@dataclass(frozen=True)
class MotionProxyConfig:
    sample_rate_hz: float = 60.0
    window_samples: int = 30
    time_tolerance_s: float = 0.0001
    source_lifetime_s: float = 0.5
    no_movement_max_gyro_rms_deg_s: float = 5.0
    movement_min_gyro_rms_deg_s: float = 15.0
    valid_no_movement_to_fallback_s: float = 3.0
    fallback_duration_s: float = 3.0

    def __post_init__(self):
        if (self.sample_rate_hz != 60 or self.window_samples != 30
                or not 0 < self.time_tolerance_s < 1 / self.sample_rate_hz
                or not 0 < self.source_lifetime_s <= 0.5
                or not 0 <= self.no_movement_max_gyro_rms_deg_s
                < self.movement_min_gyro_rms_deg_s
                or self.valid_no_movement_to_fallback_s <= 0 or self.fallback_duration_s <= 0):
            raise ValueError("Invalid motion proxy timing or thresholds")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text("utf-8"))
        if payload.get("schema") != "fogguard.b1.motion_proxy.v1":
            raise ValueError("Unsupported proxy config")
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})

    def as_dict(self):
        return asdict(self)


def _record_keys(frame):
    if not set(RAW_KEYS) <= set(frame):
        raise ValueError("Raw input must include subjectID, sessionID and taskID")
    result = frame[list(RAW_KEYS)].copy()
    if result.isna().any().any() or result.astype(str).apply(lambda x: x.str.strip().eq("")).any().any():
        raise ValueError("Record identity must be present")
    def normalize(value):
        if isinstance(value, (int, float, np.number)):
            if not np.isfinite(value):
                raise ValueError("Record identity must be finite")
            if float(value).is_integer():
                return str(int(value))
        return str(value).strip()

    return result.apply(lambda column: column.map(normalize))


def estimate_motion_feedback(frame, config=None):
    """Nonoverlapping windows use only observations available at the last sample.

    Low/middle/high RMS states are engineering proxies, not detected steps.
    A missing channel is unknown; no measurements are invented or imputed.
    """
    config = config or MotionProxyConfig()
    if "timestamp" not in frame:
        raise ValueError("Raw timestamp is required")
    work = frame.copy()
    identity = _record_keys(work)
    work[list(RAW_KEYS)] = identity
    for channel in GYRO:
        if channel not in work:
            work[channel] = np.nan
        work[channel] = pd.to_numeric(work[channel], errors="coerce")
    work["timestamp"] = pd.to_numeric(work.timestamp, errors="raise")
    blocks = identity.ne(identity.shift()).any(axis=1).cumsum()
    seen = set()
    rows = []
    for _, record in work.groupby(blocks, sort=False):
        key = tuple(record[name].iloc[0] for name in RAW_KEYS)
        if key in seen:
            raise ValueError("Records must be contiguous, not interleaved")
        seen.add(key)
        times = record.timestamp.to_numpy(float)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Raw timestamps must be finite and strictly increasing per record")
        values = record[list(GYRO)].to_numpy(float)
        for begin in range(0, len(record), config.window_samples):
            stop = min(begin + config.window_samples, len(record))
            t, sample = times[begin:stop], values[begin:stop]
            count = len(t)
            regular = count == config.window_samples and np.all(
                np.abs(np.diff(t) - 1 / config.sample_rate_hz) <= config.time_tolerance_s)
            finite = bool(np.isfinite(sample).all())
            left = float(np.sqrt(np.mean(np.sum(sample[:, :3] ** 2, axis=1)))) if finite else None
            right = float(np.sqrt(np.mean(np.sum(sample[:, 3:] ** 2, axis=1)))) if finite else None
            if not regular or not finite:
                state, valid = "unknown", False
                reason = "missing_gyro" if not finite else "incomplete_or_irregular_window"
            elif max(left, right) <= config.no_movement_max_gyro_rms_deg_s:
                state, valid, reason = "no_movement_evidence", True, "low_bilateral_gyro_rms"
            elif max(left, right) >= config.movement_min_gyro_rms_deg_s:
                state, valid, reason = "movement_evidence", True, "high_gyro_rms"
            else:
                state, valid, reason = "unknown", False, "threshold_uncertainty_band"
            rows.append({**dict(zip(KEYS, key, strict=True)),
                         "feedback_window_start_s": float(t[0]),
                         "feedback_window_end_s": float(t[-1]),
                         "feedback_available_at_s": float(t[-1]),
                         "observed_duration_s": float(t[-1] - t[0]) if valid else 0.0,
                         "signal_span_s": float(t[-1] - t[0]) if regular and finite else 0.0,
                         "feedback_valid": valid, "feedback_state": state,
                         "feedback_source": "derived_proxy", "feedback_kind": "movement_only",
                         "feedback_reason": reason, "left_gyro_rms_deg_s": left,
                         "right_gyro_rms_deg_s": right, "sample_count": count})
    if not rows:
        raise ValueError("At least one sensor sample is required")
    return pd.DataFrame(rows)
