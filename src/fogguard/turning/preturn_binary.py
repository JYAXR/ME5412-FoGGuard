"""Short-history turn preparation features and a two-stage causal predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fogguard.data.fogstar import sensor_columns
from fogguard.turning.learned import (
    extract_features,
    iter_raw_windows,
    preturn_target,
)

FEATURE_POSITIONS = {
    "left_back_long": ("ankleL", "back"),
    "left_back_short": ("ankleL", "back"),
    "bilateral_back_short": ("ankleL", "ankleR", "back"),
}


def short_features(samples: np.ndarray, channels: tuple[str, ...], *, long_only=False):
    """Rotation, gait energy, slopes and ankle asymmetry ending at decision time.

    Gravity projection is computed from the observed 120-sample history only.
    All short subwindows end at its last sample. No activity or record keys.
    """
    if samples.shape != (120, len(channels)) or not np.isfinite(samples).all():
        raise ValueError("Expected 120 finite samples")
    if long_only:
        return extract_features(samples, channels)
    values, signals = [], {}
    for position in dict.fromkeys(c.split("_")[0] for c in channels):
        acc = samples[:, [channels.index(f"{position}_acc_{a}") for a in "xyz"]]
        gyro = samples[:, [channels.index(f"{position}_gyro_{a}") for a in "xyz"]]
        gravity = acc.mean(0)
        unit = gravity / max(np.linalg.norm(gravity), 1e-8)
        vertical_rate = gyro @ unit
        gyro_norm = np.linalg.norm(gyro, axis=1)
        dynamic_acc = np.linalg.norm(acc - gravity, axis=1)
        acc_norm = np.linalg.norm(acc, axis=1)
        position_signals = [vertical_rate, gyro_norm, dynamic_acc, acc_norm]
        signals[position] = position_signals
        for signal in position_signals:
            for n in (15, 30, 60, 120):
                s = signal[-n:]
                t = np.arange(n) / 60
                t -= t.mean()
                slope = float(t @ s / (t @ t))
                values.extend([s.mean(), s.std(), np.sqrt(np.mean(s**2)),
                               np.quantile(s, 0.1), np.quantile(s, 0.9), slope,
                               np.mean(abs(np.diff(s)))])
                if n <= 60:
                    values.append(float(s.mean() - signal[-2*n:-n].mean()))
    if "ankleL" in signals and "ankleR" in signals:
        for left, right in zip(signals["ankleL"], signals["ankleR"], strict=True):
            for n in (15, 30, 60, 120):
                a, b = left[-n:], right[-n:]
                ea, eb = np.sqrt(np.mean(a**2)), np.sqrt(np.mean(b**2))
                values.extend([a.mean() - b.mean(), (ea - eb) / max(ea + eb, 1e-8),
                               np.mean((a - a.mean()) * (b - b.mean()))
                               / max(a.std() * b.std(), 1e-8)])
    return np.asarray(values, dtype=float)


def dataset(frame: pd.DataFrame, feature_set: str, horizons=(0.5, 1.0)):
    channels = sensor_columns(FEATURE_POSITIONS[feature_set])
    features, rows = [], []
    for row, samples, group, start, stop in iter_raw_windows(frame, channels):
        for horizon in horizons:
            target, onset = preturn_target(group, start, stop, horizon)
            row[f"target_{horizon}"] = target
            row[f"onset_{horizon}"] = onset
        observed = samples if row["input_valid"] else np.zeros_like(samples)
        features.append(short_features(observed, channels, long_only=feature_set.endswith("long")))
        rows.append(row)
    if not rows:
        raise ValueError("No complete windows")
    return np.vstack(features), pd.DataFrame(rows)


@dataclass
class BinaryPreturnModel:
    trigger: object
    direction_model: object
    feature_set: str
    horizon_s: float
    threshold: float

    def predict_features(self, features):
        classes = list(self.trigger.classes_)
        score = self.trigger.predict_proba(features)[:, classes.index(1)]
        direction_scores = self.direction_model.predict_proba(features)
        direction = self.direction_model.classes_[direction_scores.argmax(1)]
        return score, direction, direction_scores.max(1)

    def predict_frame(self, frame: pd.DataFrame):
        channels = sensor_columns(FEATURE_POSITIONS[self.feature_set])
        features, rows = [], []
        for row, samples, _, _, _ in iter_raw_windows(frame, channels):
            observed = samples if row["input_valid"] else np.zeros_like(samples)
            features.append(short_features(observed, channels,
                                           long_only=self.feature_set.endswith("long")))
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        score, direction, confidence = self.predict_features(np.vstack(features))
        valid = result.input_valid.to_numpy()
        triggered = valid & (score >= self.threshold)
        result["turn_score"] = np.where(valid, score, np.nan)
        result["turn_expected"] = triggered
        result["task_prediction"] = np.where(
            valid, np.where(triggered, "turn", "no_turn"), "unknown")
        result["direction"] = np.where(triggered, direction, "unknown")
        result["direction_confidence"] = np.where(triggered, confidence, 0.0)
        result["direction_task"] = "preturn_binary_then_direction"
        result["horizon_s"] = self.horizon_s
        result["target_band_s"] = 0.5
        result["model_version"] = "a2-binary-v1"
        result["prediction_source"] = "real_imu"
        return result

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Expected BinaryPreturnModel")
        return model
