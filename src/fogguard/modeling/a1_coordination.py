"""Fixed short-window bilateral motion proxies, not measured footfall phase."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_fixed_split import prediction_frame
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix

EXTRA_NAMES = tuple("coord__" + name for name in (
    "energy_asymmetry", "absolute_energy_asymmetry", "envelope_difference", "envelope_correlation",
    "peak_correlation", "peak_abs_lag_s", "correlation_025s", "correlation_050s",
    "gait_power_asymmetry", "freeze_power_asymmetry", "gait_spectral_cosine", "gait_peak_frequency_difference"))
EPS = 1e-12


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / (na * nb), -1, 1)) if min(na, nb) > EPS else 0.0


def lag_correlation(a, b, lag):
    if lag > 0:
        return correlation(a[:-lag], b[lag:])
    if lag < 0:
        return correlation(a[-lag:], b[:lag])
    return correlation(a, b)


def coordination_features(window):
    window.validate()
    if window.channel_names != DUAL or window.samples.shape != (120, 6) or window.sample_rate_hz != 60:
        raise ValueError("Coordination v1 requires complete ordered bilateral 120-point/60Hz gyro")
    x = sosfiltfilt(butter(4, (0.5, 20), btype="bandpass", fs=60, output="sos"), window.samples, axis=0)
    a, b = np.linalg.norm(x[:, :3], axis=1), np.linalg.norm(x[:, 3:], axis=1)
    el, er = np.mean(a * a), np.mean(b * b)
    asymmetry = (el - er) / (el + er + EPS)
    lags = sorted(range(-30, 31), key=lambda lag: (abs(lag), lag))
    correlations = [lag_correlation(a, b, lag) for lag in lags]
    best = int(np.argmax(correlations))
    power = np.abs(np.fft.rfft(x, axis=0)) ** 2
    pl, pr = power[:, :3].sum(axis=1), power[:, 3:].sum(axis=1)
    frequencies = np.fft.rfftfreq(120, 1 / 60)
    gait, freeze = (frequencies >= 0.5) & (frequencies < 3), (frequencies >= 3) & (frequencies <= 8)
    gl, gr = pl[gait], pr[gait]
    nl, nr = np.linalg.norm(gl), np.linalg.norm(gr)
    cosine = np.clip(np.dot(gl, gr) / (nl * nr), 0, 1) if min(nl, nr) > EPS else 0
    peak_difference = abs(frequencies[gait][np.argmax(gl)] - frequencies[gait][np.argmax(gr)]) if min(gl.sum(), gr.sum()) > EPS else 0
    bands = [(pl[band].sum() - pr[band].sum()) / (pl[band].sum() + pr[band].sum() + EPS) for band in (gait, freeze)]
    values = np.array([asymmetry, abs(asymmetry), np.mean(abs(a - b)) / (a.mean() + b.mean() + EPS),
        correlations[0], correlations[best], abs(lags[best]) / 60,
        (lag_correlation(a, b, 15) + lag_correlation(a, b, -15)) / 2,
        (lag_correlation(a, b, 30) + lag_correlation(a, b, -30)) / 2,
        *bands, cosine, peak_difference], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite coordination features")
    return values


def coordination_matrix(windows):
    base, names = a1_matrix(windows, "baseline_2s_v1")
    if base.shape[1] != 66:
        raise ValueError("Expected 66 baseline features")
    extra = np.vstack([coordination_features(w) for w in windows])
    return {"bilateral_control": (base, names),
            "coordination_rf": (np.column_stack((base, extra)), names + EXTRA_NAMES)}


@dataclass
class CoordinationFoG(A1FogModel):
    fit_participants: tuple[int, ...] = ()
    threshold_participants: tuple[int, ...] = ()

    def predict_windows(self, windows):
        if not windows or any(w.channel_names != DUAL for w in windows):
            raise ValueError("No complete ordered bilateral gyro windows")
        if self.variant == "bilateral_control":
            values, names = a1_matrix(windows, "baseline_2s_v1")
        elif self.variant == "coordination_rf":
            values, names = coordination_matrix(windows)[self.variant]
        else:
            raise ValueError("Unknown coordination model")
        if names != self.feature_names or self.channel_names != DUAL:
            raise ValueError("Coordination feature/channel schema changed")
        if list(self.classifier.classes_) != [0, 1]:
            raise ValueError("Expected binary FoG classifier")
        scores = self.classifier.predict_proba(values)[:, 1]
        frame = prediction_frame(windows, scores, self.threshold, self.variant)
        frame["model_version"] = f"a1_coordination_v1/{self.variant}"
        frame["prediction_source"] = "train_fit_participant_oof_threshold"
        return frame

    def predict_frame(self, frame):
        sensors = frame.drop(columns=[c for c in ("fog", "activity", "fog_severity") if c in frame])
        return self.predict_windows(anchored_gyro_windows(sensors, require_right=True))
