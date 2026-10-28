"""Sensor-availability fallback; missing back data is never filled or imputed."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fogguard.modeling.a1_context import ContextStream, history_links
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix


def ankle_matrix(windows):
    if not windows or any(len(w.channel_names) != 6 or not all(n.startswith("ankleL_")
            for n in w.channel_names) for w in windows):
        raise ValueError("Ankle fallback requires six left-ankle channels")
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
class AnkleFallbackFoG(A1FogModel):
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
            "model_version": "a1_ankle_fallback_v1", "prediction_source": "full_fit_new_input"})

    def new_stream(self):
        return ContextStream(self)


def route_available_predictions(primary, ankle):
    """Emit every primary row, and ankle rows only when no primary is fresh.

    Both inputs contain completed-window scores, not raw partial sensor frames.
    Ignored ankle arrivals do NOT renew a primary decision's half-second hold.
    """
    ids = ["subject_id", "session_id", "task_id"]
    a, b = primary.copy(), ankle.copy()
    a["sensor_route"], b["sensor_route"] = "primary_ankle_back", "fallback_ankle"
    for frame in (a, b):
        if frame.duplicated([*ids, "window_end_s"]).any():
            raise ValueError("Duplicate source decisions")
        if not np.isfinite(frame[["window_start_s", "window_end_s", "fog_probability"]]).all().all():
            raise ValueError("Invalid source decision")
    combined = pd.concat([a, b], ignore_index=True)
    rows = []
    for _, group in combined.groupby(ids, sort=True):
        # Treat CSV round-trip noise below the audit's 1us precision as the
        # same sample time, without changing either source timestamp.
        group = group.assign(_rank=group.sensor_route.map({"primary_ankle_back": 0, "fallback_ankle": 1}),
                             _end_key=group.window_end_s.round(6))
        group = group.sort_values(["_end_key", "_rank"], kind="stable")
        latest_primary = None
        for record in group.to_dict(orient="records"):
            if record["sensor_route"] == "primary_ankle_back":
                latest_primary = record["window_end_s"]
            elif latest_primary is not None and record["window_end_s"] < latest_primary + 0.5 - 1e-6:
                continue
            record.pop("_rank")
            record.pop("_end_key")
            rows.append(record)
    result = pd.DataFrame(rows)
    return result.sort_values("window_end_s", kind="stable").reset_index(drop=True)
