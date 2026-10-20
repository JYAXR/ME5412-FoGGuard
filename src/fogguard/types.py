"""Shared, dependency-light data contracts used across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class TurnDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"
    UNKNOWN = "unknown"


class MotionState(str, Enum):
    WALKING = "walking"
    STATIONARY = "stationary"
    UNKNOWN = "unknown"


class CueType(str, Enum):
    NONE = "none"
    AUDITORY = "auditory"
    VISUAL = "visual"
    HAPTIC = "haptic"


@dataclass(frozen=True)
class SensorWindow:
    timestamp_s: NDArray[np.float64]
    accel_mps2: NDArray[np.float64]
    gyro_dps: NDArray[np.float64]
    sample_rate_hz: float
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        n = len(self.timestamp_s)
        if n == 0 or self.accel_mps2.shape != (n, 3) or self.gyro_dps.shape != (n, 3):
            raise ValueError("SensorWindow requires N timestamps and Nx3 accel/gyro arrays")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")


@dataclass(frozen=True)
class DetectionResult:
    probability: float
    is_fog: bool
    model_name: str
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnIntent:
    direction: TurnDirection
    confidence: float
    horizon_s: float


@dataclass(frozen=True)
class MotionEstimate:
    state: MotionState
    confidence: float
    dynamic_accel_rms_mps2: float


@dataclass(frozen=True)
class ControllerContext:
    now_s: float
    fog: DetectionResult
    turn: TurnIntent
    seconds_since_last_cue: float | None = None
    motion: MotionEstimate | None = None


@dataclass(frozen=True)
class CueCommand:
    cue_type: CueType
    intensity: float
    duration_s: float
    reason: str
