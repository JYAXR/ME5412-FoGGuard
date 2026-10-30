"""Causal current-direction and preturn models, independent of A1 contracts.

All features use only a completed IMU window. Activity, FoG and record IDs
never enter features. Future annotations are used only by dataset targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fogguard.data.fogstar import sensor_columns

RAW_KEYS = ["subjectID", "sessionID", "taskID"]
KEYS = ["subject_id", "session_id", "task_id"]
CURRENT_CLASSES = ["straight", "left", "right", "unknown"]
PRETURN_CLASSES = ["no_turn", "left", "right", "unknown"]
ACTIVITY_DIRECTION = {1: "straight", 6: "right", 7: "left"}


def extract_features(samples: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
    """Raw signed statistics, half-second evolution and gravity-projected rotation.

    Acceleration stays in g and angular velocity in deg/s. No axis is assumed
    to be world yaw. Gravity projection uses this window's mean acceleration.
    """
    if samples.shape != (120, len(channels)) or not np.isfinite(samples).all():
        raise ValueError("Expected 120 finite samples with the configured channels")
    centered = samples - samples.mean(axis=0)
    power = abs(np.fft.rfft(centered, axis=0)) ** 2 / len(samples) ** 2
    freq = np.fft.rfftfreq(len(samples), 1 / 60)
    blocks = [
        samples.mean(axis=0), samples.std(axis=0),
        np.sqrt(np.mean(samples**2, axis=0)),
        np.quantile(samples, 0.1, axis=0), np.quantile(samples, 0.9, axis=0),
        np.mean(abs(np.diff(samples, axis=0)), axis=0),
    ]
    for section in np.array_split(samples, 4):
        blocks.extend([section.mean(axis=0), section.std(axis=0)])
    for lo, hi in ((0.5, 3), (3, 8), (8, 20)):
        blocks.append(np.log1p(power[(freq >= lo) & (freq < hi)].sum(axis=0)))
    for position in dict.fromkeys(name.split("_")[0] for name in channels):
        acc = samples[:, [channels.index(f"{position}_acc_{a}") for a in "xyz"]]
        gyro = samples[:, [channels.index(f"{position}_gyro_{a}") for a in "xyz"]]
        gravity = acc.mean(axis=0)
        rotation = gyro @ (gravity / max(np.linalg.norm(gravity), 1e-8))
        blocks.append(np.array([rotation.mean(), rotation.std(),
                                *[x.mean() for x in np.array_split(rotation, 4)]]))
    return np.concatenate(blocks)


def iter_raw_windows(frame: pd.DataFrame, channels: tuple[str, ...]):
    """Fixed per-record grid; missing or irregular samples invalidate a window."""
    required = {"timestamp", *RAW_KEYS, *channels}
    if not required <= set(frame):
        raise ValueError(f"Missing input columns: {sorted(required - set(frame))}")
    if frame[list(required - set(channels))].isna().any().any():
        raise ValueError("Record identities and timestamps must be present")
    for key, group in frame.groupby(RAW_KEYS, sort=False):
        times = group.timestamp.to_numpy(float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps must strictly increase inside each record")
        values = group.loc[:, list(channels)].to_numpy(float)
        for start in range(0, len(group) - 119, 30):
            stop = start + 120
            t = times[start:stop]
            valid = bool(np.isfinite(values[start:stop]).all()
                         and np.all(np.abs(np.diff(t) - 1 / 60) <= 1e-4))
            row = dict(zip(KEYS, map(int, key), strict=True))
            row.update(window_start_s=float(t[0]), window_end_s=float(t[-1]),
                       available_at_s=float(t[-1]), input_valid=valid,
                       input_reason="valid" if valid else "missing_sensor_or_signal_gap")
            yield row, values[start:stop], group, start, stop


def current_target(activity: np.ndarray, purity: float = 0.8) -> str:
    labels = np.array([ACTIVITY_DIRECTION.get(int(a), "unknown") for a in activity])
    names, counts = np.unique(labels, return_counts=True)
    top = str(names[np.argmax(counts)])
    if counts.max() / len(labels) < purity or labels[-1] != top:
        return "unknown"
    return top


def preturn_target(group: pd.DataFrame, start: int, stop: int, horizon: float):
    """First walk->turn onset in [end+h, end+h+0.5); nonwalking history rejects.

    Right-censored future intervals cannot be scored and are marked None.
    Future times and activities never enter extract_features or runtime.
    """
    times = group.timestamp.to_numpy(float)
    activity = group.activity.to_numpy(int)
    end = times[stop - 1]
    upper = end + horizon + 0.5
    if times[-1] < upper - 1e-8:
        return None, None
    # Include the first sample at/after the target interval's end. Otherwise
    # a gap spanning the entire future interval could look like a negative.
    future_stop = min(len(times), np.searchsorted(times, upper, side="left") + 1)
    if np.any(np.abs(np.diff(times[stop - 1:future_stop]) - 1 / 60) > 1e-4):
        return None, None
    if not np.all(activity[start:stop] == 1):
        return "unknown", None
    onsets = np.flatnonzero(np.isin(activity[1:], [6, 7]) & (activity[:-1] == 1)) + 1
    selected = onsets[(times[onsets] >= end + horizon - 1e-8)
                      & (times[onsets] < upper - 1e-8)]
    if not len(selected):
        return "no_turn", None
    index = selected[0]
    return ACTIVITY_DIRECTION[int(activity[index])], float(times[index])


def build_dataset(frame: pd.DataFrame, channels: tuple[str, ...], horizon: float = 0.0,
                  purity: float = 0.8) -> tuple[np.ndarray, pd.DataFrame]:
    rows, features = [], []
    for row, samples, group, start, stop in iter_raw_windows(frame, channels):
        activity = group.activity.to_numpy(int)
        target, onset = (current_target(activity[start:stop], purity), None)
        if horizon:
            target, onset = preturn_target(group, start, stop, horizon)
        row.update(target=target, target_onset_s=onset,
                   direction_task="preturn" if horizon else "current",
                   horizon_s=horizon,
                   back_vertical_rate=float("nan"))
        if row["input_valid"]:
            features.append(extract_features(samples, channels))
            acc = samples[:, [channels.index(f"back_acc_{a}") for a in "xyz"]].mean(0)
            gyro = samples[:, [channels.index(f"back_gyro_{a}") for a in "xyz"]].mean(0)
            row["back_vertical_rate"] = float(gyro @ (acc / max(np.linalg.norm(acc), 1e-8)))
        else:
            features.append(np.zeros(len(extract_features(np.zeros_like(samples), channels))))
        rows.append(row)
    if not rows:
        raise ValueError("No complete windows")
    return np.vstack(features), pd.DataFrame(rows)


@dataclass
class DirectionModel:
    classifier: RandomForestClassifier
    channels: tuple[str, ...]
    horizon_s: float = 0.0
    rejection_threshold: float = 0.0
    feature_indices: tuple[int, ...] | None = None

    def predict_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_indices is not None:
            features = features[:, self.feature_indices]
        probs = self.classifier.predict_proba(features)
        best = probs.argmax(axis=1)
        labels = self.classifier.classes_[best].astype(object)
        confidence = probs[np.arange(len(best)), best]
        labels[confidence < self.rejection_threshold] = "unknown"
        return labels, confidence

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Runtime requires signals and identities only; labels are ignored."""
        rows, features = [], []
        for row, samples, _, _, _ in iter_raw_windows(frame, self.channels):
            rows.append(row)
            features.append(extract_features(samples, self.channels) if row["input_valid"] else
                            extract_features(np.zeros_like(samples), self.channels))
        if not rows:
            return pd.DataFrame()
        output = pd.DataFrame(rows)
        labels, confidence = self.predict_features(np.vstack(features))
        invalid = ~output.input_valid.to_numpy()
        labels[invalid], confidence[invalid] = "unknown", 0.0
        output["task_prediction"] = labels
        output["task_confidence"] = confidence
        # A forecast of no upcoming turn does not establish current straight walking.
        output["direction"] = np.where(labels == "no_turn", "unknown", labels)
        output["direction_confidence"] = np.where(
            np.isin(labels, ["no_turn", "unknown"]), 0.0, confidence)
        output["direction_task"] = "preturn" if self.horizon_s else "current"
        output["horizon_s"] = self.horizon_s
        output["target_band_s"] = 0.5 if self.horizon_s else 0.0
        output["model_version"] = "a2-rf-v1"
        output["prediction_source"] = "real_imu"
        return output

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Expected DirectionModel")
        return model


def default_channels() -> tuple[str, ...]:
    return sensor_columns(("ankleL", "back"))
