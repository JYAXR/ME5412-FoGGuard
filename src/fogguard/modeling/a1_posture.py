"""Causal raw posture/slow-motion summaries and a fixed train-only model grid."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from fogguard.modeling.a1_context import CONTEXT_VARIANT, ContextStream, context_matrix
from fogguard.modeling.a1_fog import A1FogModel

POSTURE_VARIANT = "posture_context_v1"


def stable_correlation(a, b):
    a, b = a - np.mean(a), b - np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(a @ b / denominator, -1, 1)) if denominator > 1e-12 else 0.0


def posture_summary(window):
    """32 raw sensor summaries; no gravity removal or physical pose estimation."""
    rows, vectors = {}, {}
    for position in ("ankleL", "back"):
        acc = window.samples[:, [window.channel_names.index(f"{position}_acc_{a}") for a in "xyz"]]
        gyro = window.samples[:, [window.channel_names.index(f"{position}_gyro_{a}") for a in "xyz"]]
        am, gm = acc.mean(axis=0), gyro.mean(axis=0)
        scale = max(float(np.linalg.norm(am)), 1e-12)
        for axis, value, unit in zip("xyz", am, am / scale, strict=True):
            rows[f"{position}__raw_acc_mean_{axis}"] = float(value)
            rows[f"{position}__acc_mean_unit_{axis}"] = float(unit)
        acc_norm, gyro_norm = np.linalg.norm(acc, axis=1), np.linalg.norm(gyro, axis=1)
        rows[f"{position}__acc_mean_norm"] = float(np.linalg.norm(am))
        rows[f"{position}__acc_magnitude_std"] = float(acc_norm.std())
        for axis, value in zip("xyz", gm, strict=True):
            rows[f"{position}__raw_gyro_mean_{axis}"] = float(value)
        rows[f"{position}__gyro_mean_norm"] = float(np.linalg.norm(gm))
        rows[f"{position}__gyro_raw_rms"] = float(np.sqrt(np.mean(gyro_norm**2)))
        rows[f"{position}__gyro_magnitude_std"] = float(gyro_norm.std())
        vectors[position] = acc_norm, gyro_norm
    for i, sensor in enumerate(("acc", "gyro")):
        a, b = vectors["ankleL"][i], vectors["back"][i]
        rows[f"cross__{sensor}_magnitude_correlation"] = stable_correlation(a, b)
        ra, rb = np.sqrt(np.mean(a**2)), np.sqrt(np.mean(b**2))
        rows[f"cross__{sensor}_rms_balance"] = float((ra - rb) / max(ra + rb, 1e-12))
    return np.asarray(list(rows.values()), float), tuple(rows)


def posture_matrix(windows, variant=POSTURE_VARIANT, *, context=None):
    values, names, links = context_matrix(windows) if context is None else context
    if variant == CONTEXT_VARIANT:
        return values, names, links
    if variant != POSTURE_VARIANT:
        raise ValueError("Unknown posture variant")
    current, schema = [], None
    for window in windows:
        window.validate()
        row, local_names = posture_summary(window)
        if schema is not None and schema != local_names:
            raise ValueError("Posture feature schema changed")
        current.append(row)
        schema = local_names
    current = np.asarray(current)
    if current.shape != (len(windows), 32) or not np.isfinite(current).all():
        raise ValueError("Invalid posture features")
    blocks, result_names = [values, current], [*names, *schema]
    for column, lag in enumerate((2, 4)):
        valid = links[:, column] >= 0
        old = current.copy()
        old[valid] = current[links[valid, column]]
        blocks.extend([old, current - old])
        result_names.extend(f"{n}__lag_{lag}s" for n in schema)
        result_names.extend(f"{n}__delta_{lag}s" for n in schema)
    return np.column_stack(blocks), tuple(result_names), links


def search_classifier(candidate, seed):
    options = dict(candidate["params"], random_state=seed)
    if candidate["family"] == "rf":
        return RandomForestClassifier(**options, n_jobs=-1)
    if candidate["family"] == "extra":
        return ExtraTreesClassifier(**options, n_jobs=-1)
    if candidate["family"] == "hist":
        return HistGradientBoostingClassifier(**options)
    raise ValueError("Unknown posture-search family")


@dataclass
class PostureFoG(A1FogModel):
    candidate_id: str = "posture_rf"

    def predict_windows(self, windows):
        if not windows or any(w.channel_names != self.channel_names for w in windows):
            raise ValueError("No valid windows or unexpected channel ordering")
        values, names, links = posture_matrix(windows, self.variant)
        if names != self.feature_names or list(self.classifier.classes_) != [0, 1]:
            raise ValueError("Feature schema or classes changed")
        score = self.classifier.predict_proba(values)[:, 1]
        return pd.DataFrame({
            "subject_id": [w.subject_id for w in windows], "session_id": [w.session_id for w in windows],
            "task_id": [w.task_id for w in windows], "window_start_s": [w.start_s for w in windows],
            "window_end_s": [w.end_s for w in windows], "available_at_s": [w.end_s for w in windows],
            "fog_probability": score, "fog_threshold": self.threshold, "is_fog": score >= self.threshold,
            "history_2s_available": links[:, 0] >= 0, "history_4s_available": links[:, 1] >= 0,
            "feature_version": self.variant, "candidate_id": self.candidate_id,
            "model_version": f"a1_posture_search_v1/{self.candidate_id}", "prediction_source": "full_fit_new_input",
        })

    def new_stream(self):
        return ContextStream(self)
