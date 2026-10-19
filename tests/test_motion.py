import numpy as np
import pytest

from fogguard.motion import AccelerationMotionRecognizer
from fogguard.types import MotionState, SensorWindow


def _window(vertical_acceleration: np.ndarray) -> SensorWindow:
    sample_count = len(vertical_acceleration)
    return SensorWindow(
        timestamp_s=np.arange(sample_count) / 50.0,
        accel_mps2=np.column_stack(
            [np.zeros(sample_count), np.zeros(sample_count), vertical_acceleration]
        ),
        gyro_dps=np.zeros((sample_count, 3)),
        sample_rate_hz=50.0,
    )


def test_motion_recognizer_detects_stationary_window() -> None:
    result = AccelerationMotionRecognizer().predict(_window(np.full(100, 9.81)))

    assert result.state is MotionState.STATIONARY
    assert result.confidence == pytest.approx(1.0)
    assert result.dynamic_accel_rms_mps2 == pytest.approx(0.0, abs=1e-12)


def test_motion_recognizer_preserves_active_gait() -> None:
    timestamp_s = np.arange(100) / 50.0
    vertical = 9.81 + np.sin(2 * np.pi * 1.6 * timestamp_s)

    result = AccelerationMotionRecognizer().predict(_window(vertical))

    assert result.state is MotionState.WALKING
    assert result.confidence > 0.9
    assert result.dynamic_accel_rms_mps2 > 0.6
