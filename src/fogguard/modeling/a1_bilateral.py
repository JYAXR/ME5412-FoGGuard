"""Gyro-only left-anchored windows; compare identical windows with added right IMU."""

from dataclasses import dataclass

import numpy as np

from fogguard.data.fogstar import FogStarWindow, _activity_target, _contiguous_runs
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix, prediction_frame

LEFT = tuple(f"ankleL_gyro_{a}" for a in "xyz")
RIGHT = tuple(f"ankleR_gyro_{a}" for a in "xyz")
DUAL = LEFT + RIGHT


def anchored_gyro_windows(frame, *, require_right):
    channels = DUAL if require_right else LEFT
    if not {"timestamp", *channels} <= set(frame):
        raise ValueError("Missing required ankle gyro/timestamp columns")
    prepared = frame.copy()
    keys = ["subjectID", "sessionID", "taskID"]
    for name, default in zip(keys, ("new", "1", "0"), strict=True):
        if name not in prepared:
            prepared[name] = default
    blocks = prepared[keys].ne(prepared[keys].shift()).any(axis=1).cumsum()
    result = []
    for _, record in prepared.groupby(blocks, sort=False):
        times = record.timestamp.to_numpy(float)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Record timestamps must be finite and increasing")
        valid_left = np.isfinite(record[list(LEFT)].to_numpy(float)).all(axis=1)
        for section in _contiguous_runs(valid_left, times):
            run = record.iloc[section]
            for start in range(0, len(run) - 120 + 1, 30):
                chunk = run.iloc[start:start + 120]
                if not np.allclose(np.diff(chunk.timestamp), 1 / 60, atol=1e-6, rtol=0):
                    raise ValueError("Expected regular 60Hz gyro samples")
                samples = chunk[list(channels)].to_numpy(float)
                if not np.isfinite(samples).all():
                    continue  # right gap: skip, never re-anchor, interpolate or fill.
                fraction = float(chunk.fog.mean()) if "fog" in chunk else None
                activity, activity_fraction = (None, None)
                if "activity" in chunk:
                    activity, activity_fraction = _activity_target(chunk.activity.to_numpy(int), 0.5)
                window = FogStarWindow(samples, channels, 60.0, float(chunk.timestamp.iloc[0]),
                    float(chunk.timestamp.iloc[-1]), *(str(chunk[k].iloc[0]) for k in keys),
                    fog_label=int(fraction >= 0.5) if fraction is not None else None,
                    fog_fraction=fraction, activity_label=activity, activity_fraction=activity_fraction)
                window.validate()
                result.append(window)
    return result


@dataclass
class BilateralFoG(A1FogModel):
    fit_participants: tuple[int, ...] = ()
    threshold_participants: tuple[int, ...] = ()

    def predict_windows(self, windows):
        if self.variant != "bilateral_rf":
            raise ValueError("Unknown ankle model")
        if not windows or any(w.channel_names != self.channel_names for w in windows):
            raise ValueError("No complete aligned gyro windows or wrong channel schema")
        values, names = a1_matrix(windows, "baseline_2s_v1")
        if names != self.feature_names:
            raise ValueError("Ankle feature schema differs")
        scores = self.classifier.predict_proba(values)[:, list(self.classifier.classes_).index(1)]
        frame = prediction_frame(windows, scores, self.threshold, self.variant)
        frame["model_version"] = f"a1_bilateral_v1/{self.variant}"
        frame["prediction_source"] = "joint_train_fit_train_oof_threshold"
        return frame

    def predict_frame(self, frame):
        sensors = frame.drop(columns=[c for c in ("fog", "activity", "fog_severity") if c in frame])
        return self.predict_windows(anchored_gyro_windows(sensors, require_right=True))
