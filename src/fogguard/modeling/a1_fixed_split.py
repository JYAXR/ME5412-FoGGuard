"""Fresh fixed-split models, retaining their TRAIN-only fit for later testing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fogguard.modeling.a1_context import ContextStream, context_matrix
from fogguard.modeling.a1_fallback import ankle_matrix
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix
from fogguard.modeling.a1_multimodel import model_scores, waveform_matrix
from fogguard.modeling.a1_posture import posture_matrix


def feature_bank(windows, *, ankle=False):
    if ankle:
        values, names, _ = ankle_matrix(windows)
        return {"ankle": (values, names)}
    current, names = a1_matrix(windows, "baseline_2s_v1")
    context = context_matrix(windows, base=(current, names))
    posture = posture_matrix(windows, context=context)
    return {"tabular": (current, names), "waveform": (waveform_matrix(windows), ()),
            "context": context[:2], "posture": posture[:2]}


def feature_kind(model_id):
    return {"cnn_2s": "waveform", "rf_context": "context", "extra_posture": "posture", "rf_ankle": "ankle"}.get(model_id, "tabular")


def scores_for(classifier, model_id, values):
    return model_scores(classifier, "svm" if model_id == "svm_2s" else "rf", values)


def prediction_frame(windows, score, threshold, model_id):
    return pd.DataFrame({
        "subject_id": [w.subject_id for w in windows], "session_id": [w.session_id for w in windows],
        "task_id": [w.task_id for w in windows], "window_start_s": [w.start_s for w in windows],
        "window_end_s": [w.end_s for w in windows], "available_at_s": [w.end_s for w in windows],
        "fog_probability": score, "fog_threshold": threshold, "is_fog": np.asarray(score) >= threshold,
        "model_version": f"a1_fixed_split_v1/{model_id}", "prediction_source": "fixed_split_new_input"})


@dataclass
class FixedSplitFoG(A1FogModel):
    fit_participants: tuple[int, ...] = ()
    threshold_participants: tuple[int, ...] = ()

    def predict_windows(self, windows):
        if not windows or any(w.channel_names != self.channel_names for w in windows):
            raise ValueError("No valid windows or unexpected channel schema")
        model_id = self.variant
        bank = feature_bank(windows, ankle=model_id == "rf_ankle")
        values, names = bank[feature_kind(model_id)]
        if names != self.feature_names:
            raise ValueError("Fixed-split feature schema differs")
        return prediction_frame(windows, scores_for(self.classifier, model_id, values), self.threshold, model_id)

    def new_stream(self):
        return ContextStream(self)
