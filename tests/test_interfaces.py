import numpy as np

from fogguard.cueing import AdaptiveCueingController
from fogguard.detection import FreezeIndexDetector
from fogguard.turning import YawRateTurnRecognizer
from fogguard.types import ControllerContext, CueType, SensorWindow


def test_baselines_connect_end_to_end() -> None:
    t = np.arange(100) / 50.0
    window = SensorWindow(
        timestamp_s=t,
        accel_mps2=np.column_stack([np.zeros(100), np.zeros(100), np.sin(2 * np.pi * 5 * t)]),
        gyro_dps=np.column_stack([np.zeros(100), np.zeros(100), np.full(100, 30.0)]),
        sample_rate_hz=50.0,
    )
    fog = FreezeIndexDetector(threshold=0.5).predict(window)
    turn = YawRateTurnRecognizer(threshold_dps=20.0).predict(window)
    cue = AdaptiveCueingController(min_confidence=0.5).decide(ControllerContext(2.0, fog, turn))
    assert fog.is_fog
    assert turn.direction.value == "left"
    assert cue.cue_type is CueType.HAPTIC
