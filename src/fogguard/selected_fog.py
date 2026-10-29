"""The three retained FoG methods; optional CNN imports stay off RF inference paths."""

import hashlib
import json
from pathlib import Path

from fogguard.data.fogstar import iter_fogstar_windows
from fogguard.modeling.a1_bilateral import BilateralFoG, anchored_gyro_windows
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix

METHODS = ("ankle_history_rf", "bilateral_rf", "ankle_cnn")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def check_lock(folder, lock):
    for path, digest in lock["files"].items():
        if sha(Path(folder) / path) != digest:
            raise ValueError(f"Changed artifact: {path}")


def windows_for(frame, method):
    if method == "bilateral_rf":
        return anchored_gyro_windows(frame, require_right=True)
    if method == "ankle_history_rf":
        return list(iter_fogstar_windows(frame, sensor_positions=("ankleL",), max_interpolation_gap_s=0))
    if method == "ankle_cnn":
        from fogguard.modeling.a1_author_cnn import gyro_windows

        return gyro_windows(frame)
    raise ValueError("Unknown selected method")


def matrix_for(windows, method):
    if method == "bilateral_rf":
        return a1_matrix(windows, "baseline_2s_v1")
    if method == "ankle_history_rf":
        from fogguard.modeling.a1_ankle_history import ankle_matrix

        return ankle_matrix(windows)[:2]
    if method == "ankle_cnn":
        from fogguard.modeling.a1_author_cnn import centered_gyro

        return centered_gyro(windows), ()
    raise ValueError("Unknown selected method")


def wrap_classifier(classifier, method, channels, names, threshold, fit_people, threshold_people):
    args = (classifier, "author_cnn" if method == "ankle_cnn" else method, tuple(channels), tuple(names),
            float(threshold), tuple(fit_people), tuple(threshold_people))
    if method == "bilateral_rf":
        return BilateralFoG(*args)
    if method == "ankle_history_rf":
        from fogguard.modeling.a1_ankle_history import AnkleHistoryFoG

        return AnkleHistoryFoG(*args)
    if method == "ankle_cnn":
        from fogguard.modeling.a1_author_cnn import GyroFoG

        return GyroFoG(*args)
    raise ValueError("Unknown selected method")


def load_selected(path, method):
    if method not in METHODS:
        raise ValueError("Unknown selected method")
    model = A1FogModel.load(Path(path))
    expected = "author_cnn" if method == "ankle_cnn" else method
    if model.variant != expected:
        raise ValueError("Model does not match selected method")
    return model
