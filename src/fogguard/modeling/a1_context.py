"""Fixed causal sensor-history summaries, preserving every two-second decision."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fogguard.modeling.a1_fog import A1FogModel, a1_matrix

CONTEXT_VARIANT = "causal_context_6s_v1"
LAG_STEPS = (4, 8)  # two and four seconds at the fixed 0.5-second stride
STATISTICS = ("rms", "gait_energy", "freeze_energy")


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


def context_matrix(windows, variant=CONTEXT_VARIANT, *, base=None):
    """Current 132 features + two blocks of 36 lag/36 delta/one flag.

    Filtering occurs separately within each completed two-second window. The
    optional base is a trusted training cache from a1_matrix, never label data.
    Missing history is an explicit neutral encoding, not missing-sensor filling.
    """
    if variant not in ("baseline_2s_v1", CONTEXT_VARIANT):
        raise ValueError("Unknown context feature variant")
    links = history_links(windows)
    current, names = a1_matrix(windows, "baseline_2s_v1") if base is None else base
    if current.shape != (len(windows), len(names)) or not np.isfinite(current).all():
        raise ValueError("Invalid feature cache")
    if variant == "baseline_2s_v1":
        return current, names, links
    indices = [i for i, name in enumerate(names) if name.rsplit("__", 1)[-1] in STATISTICS]
    if len(indices) != 36:
        raise ValueError("Context v1 requires twelve channels with three summary statistics")
    summary = current[:, indices]
    blocks, schema = [current], list(names)
    for column, steps in enumerate(LAG_STEPS):
        available = links[:, column] >= 0
        lagged = summary.copy()
        lagged[available] = summary[links[available, column]]
        blocks.extend([lagged, summary - lagged, available[:, None].astype(float)])
        suffix = f"lag_{steps // 2}s"
        schema.extend(f"{names[i]}__{suffix}" for i in indices)
        schema.extend(f"{names[i]}__current_minus_{suffix}" for i in indices)
        schema.append(f"history_available__{suffix}")
    return np.column_stack(blocks), tuple(schema), links


@dataclass
class ContextFoG(A1FogModel):
    """FoG-only batch replay model; create a stream for incremental windows."""

    def predict_windows(self, windows):
        if not windows or any(w.channel_names != self.channel_names for w in windows):
            raise ValueError("No valid windows or unexpected channel ordering")
        values, names, links = context_matrix(windows, self.variant)
        if names != self.feature_names or list(self.classifier.classes_) != [0, 1]:
            raise ValueError("Feature schema or binary classes differ")
        score = self.classifier.predict_proba(values)[:, 1]
        return pd.DataFrame({
            "subject_id": [w.subject_id for w in windows],
            "session_id": [w.session_id for w in windows],
            "task_id": [w.task_id for w in windows],
            "window_start_s": [w.start_s for w in windows],
            "window_end_s": [w.end_s for w in windows],
            "available_at_s": [w.end_s for w in windows],
            "fog_probability": score, "fog_threshold": self.threshold,
            "is_fog": score >= self.threshold, "feature_version": self.variant,
            "model_version": f"a1_context_v1/{self.variant}",
            "history_2s_available": links[:, 0] >= 0,
            "history_4s_available": links[:, 1] >= 0,
            "prediction_source": "full_fit_new_input",
        })

    def new_stream(self):
        return ContextStream(self)


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
