"""Dependency-light multi-task baseline for FoG-STAR windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import FogStarWindow


def extract_window_features(window: FogStarWindow) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Extract time and frequency features while preserving signed yaw evidence."""
    window.validate()
    raw_samples = window.samples
    sos = butter(4, (0.5, 20.0), btype="bandpass", fs=window.sample_rate_hz, output="sos")
    dynamic_samples = sosfiltfilt(sos, raw_samples, axis=0)
    frequencies = np.fft.rfftfreq(len(dynamic_samples), d=1.0 / window.sample_rate_hz)
    spectrum = np.abs(np.fft.rfft(dynamic_samples, axis=0)) ** 2
    gait_band = (frequencies >= 0.5) & (frequencies < 3.0)
    freeze_band = (frequencies >= 3.0) & (frequencies <= 8.0)

    values: list[float] = []
    names: list[str] = []
    for index, channel in enumerate(window.channel_names):
        raw_signal = raw_samples[:, index]
        signal = dynamic_samples[:, index]
        gait_energy = float(spectrum[gait_band, index].sum())
        freeze_energy = float(spectrum[freeze_band, index].sum())
        channel_features = {
            "raw_mean": float(np.mean(raw_signal)),
            "raw_median": float(np.median(raw_signal)),
            "std": float(np.std(signal)),
            "rms": float(np.sqrt(np.mean(signal**2))),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "range": float(np.ptp(signal)),
            "dynamic_median": float(np.median(signal)),
            "iqr": float(np.percentile(signal, 75) - np.percentile(signal, 25)),
            "mean_abs_diff": float(np.mean(np.abs(np.diff(signal)))),
            "gait_energy": gait_energy,
            "freeze_energy": freeze_energy,
            "freeze_ratio": freeze_energy / max(gait_energy, 1e-12),
        }
        for feature_name, value in channel_features.items():
            names.append(f"{channel}__{feature_name}")
            values.append(value)
    return np.asarray(values, dtype=float), tuple(names)


def build_feature_matrix(
    windows: Sequence[FogStarWindow],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    if not windows:
        raise ValueError("At least one FoG-STAR window is required")
    rows: list[NDArray[np.float64]] = []
    expected_names: tuple[str, ...] | None = None
    for window in windows:
        row, names = extract_window_features(window)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("All windows must use identical channels and ordering")
        rows.append(row)
    return np.vstack(rows), expected_names or ()


def create_baseline_classifiers(
    *,
    n_estimators: int = 300,
    random_state: int = 5412,
) -> tuple[RandomForestClassifier, RandomForestClassifier]:
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    common = {
        "n_estimators": n_estimators,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": random_state,
    }
    return (
        RandomForestClassifier(**common, oob_score=True),
        RandomForestClassifier(**common),
    )


@dataclass
class FogStarMultiTaskModel:
    """Independent FoG and direction heads with FoG-priority state output."""

    fog_classifier: Any
    direction_classifier: Any
    channel_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    fog_feature_indices: tuple[int, ...]
    fog_threshold: float = 0.5

    def predict_window(self, window: FogStarWindow) -> dict[str, object]:
        if window.channel_names != self.channel_names:
            raise ValueError(
                f"Model expects channels {self.channel_names}, got {window.channel_names}"
            )
        features, names = extract_window_features(window)
        if names != self.feature_names:
            raise ValueError("Window feature schema does not match the trained model")
        row = features.reshape(1, -1)
        fog_classes = list(self.fog_classifier.classes_)
        fog_index = fog_classes.index(1)
        fog_probability = float(
            self.fog_classifier.predict_proba(row[:, self.fog_feature_indices])[0, fog_index]
        )
        direction_probabilities = self.direction_classifier.predict_proba(row)[0]
        direction_index = int(np.argmax(direction_probabilities))
        direction = str(self.direction_classifier.classes_[direction_index])
        is_fog = fog_probability >= self.fog_threshold
        return {
            "fog_probability": fog_probability,
            "is_fog": is_fog,
            "direction": direction,
            "direction_confidence": float(direction_probabilities[direction_index]),
            "state": "fog" if is_fog else direction,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> FogStarMultiTaskModel:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model
