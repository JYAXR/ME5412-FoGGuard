"""Conservative activity-state baseline for voluntary-stop suppression."""

from dataclasses import dataclass

import numpy as np

from fogguard.types import MotionEstimate, MotionState, SensorWindow


@dataclass
class AccelerationMotionRecognizer:
    """Separate stationary windows from active gait using dynamic acceleration RMS.

    The recognizer observes physical inactivity, not a person's intent. In the
    synthetic P0 scenarios, ``stationary`` is the measurable proxy for the
    labelled voluntary-stop interval.
    """

    stationary_rms_threshold_mps2: float = 0.15
    uncertainty_margin: float = 0.10

    def __post_init__(self) -> None:
        if self.stationary_rms_threshold_mps2 <= 0:
            raise ValueError("stationary_rms_threshold_mps2 must be positive")
        if not 0.0 <= self.uncertainty_margin < 1.0:
            raise ValueError("uncertainty_margin must be between 0 and 1")

    def predict(self, window: SensorWindow) -> MotionEstimate:
        window.validate()
        acceleration = window.accel_mps2
        if not np.isfinite(acceleration).all():
            return MotionEstimate(MotionState.UNKNOWN, 0.0, float("nan"))

        dynamic = acceleration - np.mean(acceleration, axis=0, keepdims=True)
        dynamic_rms = float(np.sqrt(np.mean(np.sum(dynamic**2, axis=1))))
        threshold = self.stationary_rms_threshold_mps2
        if dynamic_rms < threshold:
            state = MotionState.STATIONARY
            confidence = 1.0 - dynamic_rms / threshold
        else:
            state = MotionState.WALKING
            confidence = (dynamic_rms - threshold) / threshold

        if confidence < self.uncertainty_margin:
            state = MotionState.UNKNOWN
            confidence = 0.0
        return MotionEstimate(
            state=state,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            dynamic_accel_rms_mps2=dynamic_rms,
        )
