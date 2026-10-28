"""Same-sensor RF/SVM/boosting/CNN experiment; no direction or control changes."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fogguard.data.fogstar import FogStarWindow, iter_fogstar_windows
from fogguard.modeling.a1_fog import a1_matrix

FAMILIES = ("rf", "svm", "hist_gb", "cnn")


def waveform_matrix(windows):
    if not windows:
        raise ValueError("No valid windows")
    channels = windows[0].channel_names
    sos = butter(4, (0.5, 20), btype="bandpass", fs=60, output="sos")
    rows = []
    for window in windows:
        window.validate()
        if (len(window.samples) != 120 or window.sample_rate_hz != 60
                or window.channel_names != channels):
            raise ValueError("CNN requires ordered 120xC windows at 60Hz")
        rows.append(sosfiltfilt(sos, window.samples, axis=0).T)
    return np.asarray(rows, dtype=np.float32)


def make_classifier(family, config):
    seed = config["random_state"]
    options = dict(config[family])
    if family == "rf":
        return RandomForestClassifier(**options, random_state=seed, n_jobs=-1)
    if family == "svm":
        options.pop("score")
        # Do not request libsvm's sample-level probability calibration.
        # Omitting this deprecated option also supports sklearn 1.9's new default.
        options.pop("probability")
        return make_pipeline(StandardScaler(), SVC(**options, random_state=seed, cache_size=512))
    if family == "hist_gb":
        return HistGradientBoostingClassifier(**options, random_state=seed)
    if family == "cnn":
        from fogguard.modeling.a1_cnn import TorchWindowClassifier
        return TorchWindowClassifier(options, seed)
    raise ValueError(f"Unknown classifier: {family}")


def model_scores(model, family, values):
    if family == "svm":
        # Stable monotonic mapping, NOT Platt calibration on random windows.
        return expit(model.decision_function(values))
    return model.predict_proba(values)[:, list(model.classes_).index(1)]


@dataclass
class MultiModelFoG:
    classifier: Any
    family: str
    channel_names: tuple[str, ...]
    threshold: float

    def predict_windows(self, windows: Sequence[FogStarWindow]):
        if not windows or any(w.channel_names != self.channel_names for w in windows):
            raise ValueError("No valid windows or unexpected channel ordering")
        values = (waveform_matrix(windows) if self.family == "cnn"
                  else a1_matrix(windows, "baseline_2s_v1")[0])
        score = model_scores(self.classifier, self.family, values)
        return pd.DataFrame({
            "subject_id": [w.subject_id for w in windows],
            "session_id": [w.session_id for w in windows],
            "task_id": [w.task_id for w in windows],
            "window_start_s": [w.start_s for w in windows],
            "window_end_s": [w.end_s for w in windows],
            "available_at_s": [w.end_s for w in windows],
            "fog_probability": score, "fog_threshold": self.threshold,
            "is_fog": score >= self.threshold, "model_version": f"a1_multimodel_v1/{self.family}",
            "prediction_source": "full_fit_new_input",
        })

    def predict_frame(self, frame):
        positions = tuple(dict.fromkeys(n.split("_", 1)[0] for n in self.channel_names))
        sensors = frame.drop(columns=[n for n in ("fog", "activity", "fog_severity") if n in frame])
        return self.predict_windows(list(iter_fogstar_windows(
            sensors, sensor_positions=positions, max_interpolation_gap_s=0)))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Not a MultiModelFoG")
        return model
