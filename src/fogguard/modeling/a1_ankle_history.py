"""Retained 140-feature single-ankle RF with causal history."""
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fogguard.modeling.a1_fog import A1FogModel, a1_matrix

LAG_STEPS = (4, 8)

def record_key(window):
    return window.subject_id, window.session_id, window.task_id

def history_links(windows):
    """Return row indices, resetting at boundaries/gaps; never consult labels."""
    links = np.full((len(windows), len(LAG_STEPS)), -1, dtype=int)
    run_start = 0
    for i, w in enumerate(windows):
        w.validate()
        if (w.sample_rate_hz != 60 or len(w.samples) != 120
                or not np.isclose(w.end_s - w.start_s, 119 / 60, atol=1e-6, rtol=0)):
            raise ValueError("Context requires regular 120-sample, 60Hz windows")
        if i:
            previous = windows[i - 1]
            same_record = record_key(w) == record_key(previous)
            if same_record and w.start_s <= previous.start_s:
                raise ValueError("Windows must be chronological without duplicates")
            if (not same_record or not np.isclose(w.start_s - previous.start_s, 0.5,
                                                 atol=1e-6, rtol=0)):
                run_start = i
        for column, lag in enumerate(LAG_STEPS):
            if i - lag >= run_start:
                links[i, column] = i - lag
    return links

class ContextStream:
    """Nine completed windows of sensor history per independent replay stream."""

    def __init__(self, model):
        self.model = model
        self.windows = deque(maxlen=max(LAG_STEPS) + 1)

    def push(self, window):
        # Check before mutating state; invalid/out-of-order input must fail closed.
        pending = [*self.windows, window]
        history_links(pending)
        if self.windows:
            previous = self.windows[-1]
            if (record_key(window) != record_key(previous)
                    or not np.isclose(window.start_s - previous.start_s, 0.5,
                                      atol=1e-6, rtol=0)):
                pending = [window]
        prediction = self.model.predict_windows(pending).iloc[[-1]].reset_index(drop=True)
        self.windows.clear()
        self.windows.extend(pending[-(max(LAG_STEPS) + 1):])
        return prediction

def ankle_matrix(windows):
    if not windows or any(len(w.channel_names) != 6 or not all(n.startswith("ankleL_")
            for n in w.channel_names) for w in windows):
        raise ValueError("Ankle history RF requires six left-ankle channels")
    base, names = a1_matrix(windows, "baseline_2s_v1")
    links = history_links(windows)
    selected = [i for i, n in enumerate(names) if n.rsplit("__", 1)[-1] in ("rms", "gait_energy", "freeze_energy")]
    current = base[:, selected]
    blocks, schema = [base], list(names)
    for j, lag in enumerate((2, 4)):
        valid = links[:, j] >= 0
        previous = current.copy()
        previous[valid] = current[links[valid, j]]
        blocks.extend([previous, current - previous, valid[:, None].astype(float)])
        schema.extend(f"{names[i]}__lag_{lag}s" for i in selected)
        schema.extend(f"{names[i]}__delta_{lag}s" for i in selected)
        schema.append(f"history_available__lag_{lag}s")
    return np.column_stack(blocks), tuple(schema), links

@dataclass
class AnkleHistoryFoG(A1FogModel):
    fit_participants: tuple[int, ...] = ()
    threshold_participants: tuple[int, ...] = ()
    def predict_windows(self, windows):
        if not windows or any(w.channel_names != self.channel_names for w in windows):
            raise ValueError("No valid windows or unexpected channels")
        values, schema, links = ankle_matrix(windows)
        if schema != self.feature_names:
            raise ValueError("Ankle feature schema differs")
        score = self.classifier.predict_proba(values)[:, list(self.classifier.classes_).index(1)]
        return pd.DataFrame({
            "subject_id": [w.subject_id for w in windows], "session_id": [w.session_id for w in windows],
            "task_id": [w.task_id for w in windows], "window_start_s": [w.start_s for w in windows],
            "window_end_s": [w.end_s for w in windows], "available_at_s": [w.end_s for w in windows],
            "fog_probability": score, "fog_threshold": self.threshold, "is_fog": score >= self.threshold,
            "history_2s_available": links[:, 0] >= 0, "history_4s_available": links[:, 1] >= 0,
            "model_version": "fog_selected_v1/ankle_history_rf", "prediction_source": "train_fit_new_input"})

    def new_stream(self):
        return ContextStream(self)
