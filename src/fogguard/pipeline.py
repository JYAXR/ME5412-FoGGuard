"""Reusable end-to-end replay pipeline for evaluation, scripts, and UI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fogguard.cueing import AdaptiveCueingController
from fogguard.data import WindowReplay, load_csv
from fogguard.detection import FreezeIndexDetector
from fogguard.interfaces import (
    CueingController,
    FoGDetector,
    MotionStateRecognizer,
    TurnIntentRecognizer,
)
from fogguard.motion import AccelerationMotionRecognizer
from fogguard.turning import YawRateTurnRecognizer
from fogguard.types import ControllerContext, CueType, MotionState

RESULT_COLUMNS = (
    "window_start_s",
    "window_end_s",
    "fog_probability",
    "fog_detected",
    "motion_state",
    "motion_confidence",
    "dynamic_accel_rms_mps2",
    "fog_suppressed",
    "effective_fog_detected",
    "turn",
    "turn_confidence",
    "turn_horizon_s",
    "gyro_available",
    "cue",
    "cue_intensity",
    "cue_duration_s",
    "cue_reason",
    "seconds_since_last_cue",
    "truth_fog_label",
    "fog_matches_truth",
    "effective_fog_matches_truth",
    "truth_turn_label",
    "turn_matches_truth",
    "scenario_id",
    "source_dataset",
)


def replay_frame(
    frame: pd.DataFrame,
    *,
    detector: FoGDetector | None = None,
    motion_recognizer: MotionStateRecognizer | None = None,
    turner: TurnIntentRecognizer | None = None,
    controller: CueingController | None = None,
    sample_rate_hz: float = 50.0,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> pd.DataFrame:
    """Run one dataframe through the shared detector, context, and cueing path."""
    detector = detector or FreezeIndexDetector()
    motion_recognizer = motion_recognizer or AccelerationMotionRecognizer()
    turner = turner or YawRateTurnRecognizer()
    controller = controller or AdaptiveCueingController()
    replay = WindowReplay(
        frame,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )

    rows: list[dict[str, object]] = []
    last_cue_time_s: float | None = None
    for window in replay.windows():
        fog = detector.predict(window)
        motion = motion_recognizer.predict(window)
        turn = turner.predict(window)
        fog_suppressed = fog.is_fog and motion.state is MotionState.STATIONARY
        effective_fog_detected = fog.is_fog and not fog_suppressed
        now_s = float(window.timestamp_s[-1])
        seconds_since_last_cue = None if last_cue_time_s is None else now_s - last_cue_time_s
        cue = controller.decide(
            ControllerContext(
                now_s=now_s,
                fog=fog,
                turn=turn,
                seconds_since_last_cue=seconds_since_last_cue,
                motion=motion,
            )
        )
        if cue.cue_type is not CueType.NONE:
            last_cue_time_s = now_s

        metadata = window.metadata
        truth_fog_label = metadata.get("fog_label")
        truth_turn_label = metadata.get("turn_label")
        rows.append(
            {
                "window_start_s": float(window.timestamp_s[0]),
                "window_end_s": now_s,
                "fog_probability": fog.probability,
                "fog_detected": fog.is_fog,
                "motion_state": motion.state.value,
                "motion_confidence": motion.confidence,
                "dynamic_accel_rms_mps2": motion.dynamic_accel_rms_mps2,
                "fog_suppressed": fog_suppressed,
                "effective_fog_detected": effective_fog_detected,
                "turn": turn.direction.value,
                "turn_confidence": turn.confidence,
                "turn_horizon_s": turn.horizon_s,
                "gyro_available": metadata.get("gyro_available", "true").lower() == "true",
                "cue": cue.cue_type.value,
                "cue_intensity": cue.intensity,
                "cue_duration_s": cue.duration_s,
                "cue_reason": cue.reason,
                "seconds_since_last_cue": (
                    seconds_since_last_cue if seconds_since_last_cue is not None else float("nan")
                ),
                "truth_fog_label": truth_fog_label,
                "fog_matches_truth": (
                    None if truth_fog_label is None else fog.is_fog == (truth_fog_label == "1")
                ),
                "effective_fog_matches_truth": (
                    None
                    if truth_fog_label is None
                    else effective_fog_detected == (truth_fog_label == "1")
                ),
                "truth_turn_label": truth_turn_label,
                "turn_matches_truth": (
                    None if truth_turn_label is None else turn.direction.value == truth_turn_label
                ),
                "scenario_id": metadata.get("scenario_id", ""),
                "source_dataset": metadata.get("source_dataset", ""),
            }
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def replay_csv(
    path: str | Path,
    *,
    detector: FoGDetector | None = None,
    motion_recognizer: MotionStateRecognizer | None = None,
    turner: TurnIntentRecognizer | None = None,
    controller: CueingController | None = None,
    sample_rate_hz: float = 50.0,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> pd.DataFrame:
    """Load a canonical CSV and run it through :func:`replay_frame`."""
    source = Path(path)
    result = replay_frame(
        load_csv(source),
        detector=detector,
        motion_recognizer=motion_recognizer,
        turner=turner,
        controller=controller,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    result.insert(0, "source_file", str(source))
    return result
