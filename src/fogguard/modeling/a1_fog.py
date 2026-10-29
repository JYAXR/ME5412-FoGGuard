"""Versioned A1 FoG-only features and inference, independent of direction/control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fogguard.data.fogstar import FogStarWindow, iter_fogstar_windows
from fogguard.modeling.fogstar_baseline import extract_window_features

VARIANTS = ("baseline_2s_v1",)


def a1_features(window: FogStarWindow, variant: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Eleven shared statistics per channel of a completed, filtered 2s window."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown A1 feature variant: {variant}")
    if window.sample_rate_hz != 60 or len(window.samples) != 120:
        raise ValueError("A1 v1 requires exactly 120 samples at 60 Hz")
    full, names = extract_window_features(window)
    keep = [i for i, name in enumerate(names)
            if not name.endswith(("__raw_mean", "__raw_median"))]
    values = list(full[keep])
    result_names = [names[i] for i in keep]
    return np.asarray(values, dtype=float), tuple(result_names)


def a1_matrix(windows: Sequence[FogStarWindow], variant: str):
    if not windows:
        raise ValueError("No windows to extract")
    rows, schema = [], None
    channels = windows[0].channel_names
    for window in windows:
        if window.channel_names != channels:
            raise ValueError("Channel ordering changed between windows")
        values, names = a1_features(window, variant)
        if schema is not None and names != schema:
            raise ValueError("Feature schema changed between windows")
        rows.append(values)
        schema = names
    return np.vstack(rows), schema


@dataclass
class A1FogModel:
    classifier: Any
    variant: str
    channel_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    threshold: float

    def predict_windows(self, windows: Sequence[FogStarWindow]) -> pd.DataFrame:
        raise NotImplementedError("Use one of the three selected concrete detectors")

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        positions = tuple(dict.fromkeys(name.split("_", 1)[0] for name in self.channel_names))
        # Drop labels before window construction: runtime never needs ground truth.
        sensors = frame.drop(columns=[name for name in ("fog", "activity", "fog_severity")
                                      if name in frame])
        windows = list(iter_fogstar_windows(sensors, sensor_positions=positions,
                                           max_interpolation_gap_s=0.0))
        return self.predict_windows(windows)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Not an A1FogModel")
        return model


def prediction_frame(windows, score, threshold, model_id):
    """Shared FoG-only output; time availability is never backdated."""
    return pd.DataFrame({
        "subject_id": [w.subject_id for w in windows], "session_id": [w.session_id for w in windows],
        "task_id": [w.task_id for w in windows], "window_start_s": [w.start_s for w in windows],
        "window_end_s": [w.end_s for w in windows], "available_at_s": [w.end_s for w in windows],
        "fog_probability": score, "fog_threshold": threshold, "is_fog": np.asarray(score) >= threshold,
        "model_version": f"fog_selected_v1/{model_id}", "prediction_source": "train_fit_new_input"})
