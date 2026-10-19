import numpy as np
import pytest

from fogguard.turning import YawRateTurnRecognizer
from fogguard.types import SensorWindow, TurnDirection


def make_window(yaw_rate_dps: float, *, gyro_available: str = "true") -> SensorWindow:
    t = np.arange(100) / 50.0
    return SensorWindow(
        timestamp_s=t,
        accel_mps2=np.zeros((100, 3)),
        gyro_dps=np.column_stack([np.zeros(100), np.zeros(100), np.full(100, yaw_rate_dps)]),
        sample_rate_hz=50.0,
        metadata={"gyro_available": gyro_available},
    )


@pytest.mark.parametrize(
    ("yaw_rate_dps", "expected_direction", "expected_confidence"),
    [
        (0.0, TurnDirection.STRAIGHT, 1.0),
        (10.0, TurnDirection.STRAIGHT, 0.5),
        (38.0, TurnDirection.LEFT, 0.9),
        (-38.0, TurnDirection.RIGHT, 0.9),
    ],
)
def test_recognizer_classifies_direction_with_margin_confidence(
    yaw_rate_dps: float,
    expected_direction: TurnDirection,
    expected_confidence: float,
) -> None:
    result = YawRateTurnRecognizer(threshold_dps=20.0).predict(make_window(yaw_rate_dps))

    assert result.direction is expected_direction
    assert result.confidence == pytest.approx(expected_confidence)
    assert result.horizon_s == 0.5


def test_recognizer_returns_unknown_without_measured_gyroscope() -> None:
    result = YawRateTurnRecognizer().predict(make_window(0.0, gyro_available="false"))

    assert result.direction is TurnDirection.UNKNOWN
    assert result.confidence == 0.0


def test_recognizer_returns_unknown_for_non_finite_yaw_rate() -> None:
    window = make_window(0.0)
    window.gyro_dps[0, 2] = np.nan

    result = YawRateTurnRecognizer().predict(window)

    assert result.direction is TurnDirection.UNKNOWN
    assert result.confidence == 0.0


def test_recognizer_returns_unknown_near_the_decision_boundary() -> None:
    result = YawRateTurnRecognizer().predict(make_window(19.0))

    assert result.direction is TurnDirection.UNKNOWN
    assert result.confidence == 0.0


@pytest.mark.parametrize("field", ["threshold_dps", "horizon_s"])
def test_recognizer_rejects_non_positive_configuration(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        YawRateTurnRecognizer(**{field: 0.0})


@pytest.mark.parametrize("margin", [-0.1, 1.0])
def test_recognizer_rejects_invalid_uncertainty_margin(margin: float) -> None:
    with pytest.raises(ValueError, match="uncertainty_margin must be between 0 and 1"):
        YawRateTurnRecognizer(uncertainty_margin=margin)
