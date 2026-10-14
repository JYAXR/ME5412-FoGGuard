import numpy as np

from fogguard.detection import FreezeIndexDetector
from fogguard.types import SensorWindow


def test_freeze_index_detector_reports_frequency_band_evidence() -> None:
    sample_rate_hz = 50.0
    timestamp_s = np.arange(100) / sample_rate_hz
    freeze_signal = np.sin(2 * np.pi * 4.0 * timestamp_s)
    window = SensorWindow(
        timestamp_s=timestamp_s,
        accel_mps2=np.column_stack(
            [np.zeros(100), np.zeros(100), freeze_signal]
        ),
        gyro_dps=np.zeros((100, 3)),
        sample_rate_hz=sample_rate_hz,
    )

    result = FreezeIndexDetector().predict(window)

    assert set(result.evidence) == {
        "freeze_index",
        "locomotor_power",
        "freeze_power",
    }
    assert 0.0 <= result.evidence["freeze_index"] <= 1.0
    assert result.evidence["freeze_power"] > result.evidence["locomotor_power"]
