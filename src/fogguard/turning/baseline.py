"""Simple turning-intent baseline based on mean yaw rate."""

from dataclasses import dataclass

import numpy as np

from fogguard.types import SensorWindow, TurnDirection, TurnIntent


@dataclass
class YawRateTurnRecognizer:
    threshold_dps: float = 20.0

    def predict(self, window: SensorWindow) -> TurnIntent:
        window.validate()
        mean_yaw = float(np.mean(window.gyro_dps[:, 2]))
        magnitude = abs(mean_yaw)
        if magnitude < self.threshold_dps:
            direction = TurnDirection.STRAIGHT
        else:
            direction = TurnDirection.LEFT if mean_yaw > 0 else TurnDirection.RIGHT
        confidence = float(np.clip(magnitude / max(self.threshold_dps * 2, 1e-9), 0.0, 1.0))
        return TurnIntent(direction=direction, confidence=confidence, horizon_s=0.5)
