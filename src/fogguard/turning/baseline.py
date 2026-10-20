"""Simple turning-intent baseline based on mean yaw rate."""

from dataclasses import dataclass

import numpy as np

from fogguard.types import SensorWindow, TurnDirection, TurnIntent


@dataclass
class YawRateTurnRecognizer:
    threshold_dps: float = 20.0
    horizon_s: float = 0.5
    uncertainty_margin: float = 0.10

    def __post_init__(self) -> None:
        if self.threshold_dps <= 0:
            raise ValueError("threshold_dps must be positive")
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if not 0.0 <= self.uncertainty_margin < 1.0:
            raise ValueError("uncertainty_margin must be between 0 and 1")

    def _unknown(self) -> TurnIntent:
        return TurnIntent(
            direction=TurnDirection.UNKNOWN,
            confidence=0.0,
            horizon_s=self.horizon_s,
        )

    def predict(self, window: SensorWindow) -> TurnIntent:
        window.validate()
        if window.metadata.get("gyro_available", "true").lower() != "true":
            return self._unknown()

        yaw_rate = window.gyro_dps[:, 2]
        if not np.isfinite(yaw_rate).all():
            return self._unknown()

        mean_yaw = float(np.mean(yaw_rate))
        magnitude = abs(mean_yaw)
        if magnitude < self.threshold_dps:
            direction = TurnDirection.STRAIGHT
            confidence = 1.0 - magnitude / self.threshold_dps
        else:
            direction = TurnDirection.LEFT if mean_yaw > 0 else TurnDirection.RIGHT
            confidence = (magnitude - self.threshold_dps) / self.threshold_dps
        if confidence < self.uncertainty_margin:
            return self._unknown()
        return TurnIntent(
            direction=direction,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            horizon_s=self.horizon_s,
        )
