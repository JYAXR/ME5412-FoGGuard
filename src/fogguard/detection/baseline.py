"""Explainable frequency-domain FoG baseline for the vertical acceleration axis."""

from dataclasses import dataclass

import numpy as np

from fogguard.types import DetectionResult, SensorWindow


@dataclass
class FreezeIndexDetector:
    threshold: float = 0.60

    def predict(self, window: SensorWindow) -> DetectionResult:
        window.validate()
        signal = window.accel_mps2[:, 2] - np.mean(window.accel_mps2[:, 2])
        power = np.abs(np.fft.rfft(signal)) ** 2
        frequency = np.fft.rfftfreq(len(signal), d=1.0 / window.sample_rate_hz)
        locomotor = float(power[(frequency >= 0.5) & (frequency < 3.0)].sum())
        freeze = float(power[(frequency >= 3.0) & (frequency <= 8.0)].sum())
        freeze_index = freeze / (locomotor + freeze + 1e-9)
        probability = float(np.clip(freeze_index, 0.0, 1.0))
        return DetectionResult(
            probability=probability,
            is_fog=probability >= self.threshold,
            model_name="freeze-index-baseline",
            evidence={
                "freeze_index": freeze_index,
                "locomotor_power": locomotor,
                "freeze_power": freeze,
            },
        )
