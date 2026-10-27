"""Shared FoG-STAR model and temporal-integration replay path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fogguard.context import FogStarIntegrationConfig, FogStarTemporalIntegrator
from fogguard.data import iter_fogstar_windows, load_fogstar_frame
from fogguard.interfaces import CueingController
from fogguard.modeling import FogStarMultiTaskModel
from fogguard.motion import AccelerationMotionRecognizer
from fogguard.turning import YawRateTurnRecognizer
from fogguard.types import MotionEstimate, SensorWindow, TurnIntent


def estimate_fogstar_motion(
    window: object,
    *,
    stationary_rms_threshold_mps2: float = 0.30,
) -> MotionEstimate:
    """Estimate physical inactivity from the left ankle without using labels."""
    channel_names = tuple(window.channel_names)
    position = "ankleL" if "ankleL_acc_x" in channel_names else channel_names[0].split("_", 1)[0]
    acc_indices = [channel_names.index(f"{position}_acc_{axis}") for axis in "xyz"]
    gyro_indices = [channel_names.index(f"{position}_gyro_{axis}") for axis in "xyz"]
    samples = window.samples
    sensor_window = SensorWindow(
        timestamp_s=np.arange(len(samples), dtype=float) / window.sample_rate_hz,
        accel_mps2=samples[:, acc_indices] * 9.80665,
        gyro_dps=samples[:, gyro_indices],
        sample_rate_hz=window.sample_rate_hz,
    )
    return AccelerationMotionRecognizer(
        stationary_rms_threshold_mps2=stationary_rms_threshold_mps2
    ).predict(sensor_window)


def estimate_fogstar_yaw_turn(window: object) -> TurnIntent:
    """Keep the yaw-rate recognizer as an explicit agreement-only fallback."""
    channel_names = tuple(window.channel_names)
    position = "back" if "back_gyro_z" in channel_names else channel_names[0].split("_", 1)[0]
    acc_indices = [channel_names.index(f"{position}_acc_{axis}") for axis in "xyz"]
    gyro_indices = [channel_names.index(f"{position}_gyro_{axis}") for axis in "xyz"]
    samples = window.samples
    sensor_window = SensorWindow(
        timestamp_s=np.arange(len(samples), dtype=float) / window.sample_rate_hz,
        accel_mps2=samples[:, acc_indices] * 9.80665,
        gyro_dps=samples[:, gyro_indices],
        sample_rate_hz=window.sample_rate_hz,
    )
    return YawRateTurnRecognizer().predict(sensor_window)


def predict_fogstar_frame(
    frame: pd.DataFrame,
    model: FogStarMultiTaskModel,
    *,
    sensor_positions: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return raw, auditable model outputs; truth is copied only for evaluation."""
    if sensor_positions is None:
        sensor_positions = tuple(
            dict.fromkeys(name.split("_", 1)[0] for name in model.channel_names)
        )
    windows = list(iter_fogstar_windows(frame, sensor_positions=sensor_positions))
    if not windows:
        raise ValueError("No complete valid FoG-STAR windows were found")
    predictions = model.predict_windows(windows)
    rows: list[dict[str, object]] = []
    for window, prediction in zip(windows, predictions, strict=True):
        motion = estimate_fogstar_motion(window)
        yaw_turn = estimate_fogstar_yaw_turn(window)
        rows.append(
            {
                "window_start_s": window.start_s,
                "window_end_s": window.end_s,
                "subject_id": window.subject_id,
                "session_id": window.session_id,
                "task_id": window.task_id,
                "fog_threshold": model.fog_threshold,
                **prediction,
                "motion_state": motion.state.value,
                "motion_confidence": motion.confidence,
                "dynamic_accel_rms_mps2": motion.dynamic_accel_rms_mps2,
                "yaw_turn": yaw_turn.direction.value,
                "yaw_turn_confidence": yaw_turn.confidence,
                "truth_fog_label": window.fog_label,
                "truth_turn_label": window.direction_label,
                "truth_activity_label": window.activity_label,
                "truth_fog_fraction": window.fog_fraction,
                "truth_turn_fraction": window.direction_fraction,
                "truth_activity_fraction": window.activity_fraction,
            }
        )
    return pd.DataFrame(rows)


def integrate_fogstar_predictions(
    predictions: pd.DataFrame,
    *,
    config: FogStarIntegrationConfig | None = None,
    controller: CueingController | None = None,
) -> pd.DataFrame:
    """Apply the shared causal temporal path to raw model predictions."""
    return FogStarTemporalIntegrator(config=config, controller=controller).process(predictions)


def replay_fogstar_frame(
    frame: pd.DataFrame,
    model: FogStarMultiTaskModel,
    *,
    config: FogStarIntegrationConfig | None = None,
    controller: CueingController | None = None,
) -> pd.DataFrame:
    return integrate_fogstar_predictions(
        predict_fogstar_frame(frame, model),
        config=config,
        controller=controller,
    )


def replay_fogstar_csv(
    path: str | Path,
    *,
    model_path: str | Path = "artifacts/fogstar_multitask_baseline.joblib",
    config: FogStarIntegrationConfig | None = None,
    controller: CueingController | None = None,
) -> pd.DataFrame:
    model = FogStarMultiTaskModel.load(model_path)
    sensor_positions = tuple(
        dict.fromkeys(name.split("_", 1)[0] for name in model.channel_names)
    )
    frame = load_fogstar_frame(path, sensor_positions=sensor_positions, require_labels=False)
    result = replay_fogstar_frame(frame, model, config=config, controller=controller)
    result.insert(0, "source_file", str(Path(path)))
    return result
