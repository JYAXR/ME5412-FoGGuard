"""Completed-window invariant representation ablation; binary FoG only."""

from dataclasses import dataclass

from fogguard.evaluation.external_failure import NAMES, summarize_windows
from fogguard.modeling.a1_bilateral import DUAL, anchored_gyro_windows
from fogguard.modeling.a1_fixed_split import prediction_frame
from fogguard.modeling.a1_fog import A1FogModel, a1_matrix

FEATURES = tuple("invariant__" + name for name in NAMES)


def feature_matrix(windows, variant):
    if not windows or any(w.channel_names != DUAL or w.samples.shape != (120, 6) or w.sample_rate_hz != 60 for w in windows):
        raise ValueError("Expected complete ordered bilateral 120-point/60Hz gyro windows")
    if variant == "axis_control":
        return a1_matrix(windows, "baseline_2s_v1")
    if variant == "invariant_rf":
        return summarize_windows(windows), FEATURES
    raise ValueError("Unknown invariant experiment variant")


@dataclass
class InvariantFoG(A1FogModel):
    fit_participants: tuple[int, ...] = ()
    threshold_participants: tuple[int, ...] = ()

    def predict_windows(self, windows):
        values, names = feature_matrix(windows, self.variant)
        if self.channel_names != DUAL or names != self.feature_names or list(self.classifier.classes_) != [0, 1]:
            raise ValueError("Frozen channel/feature/class schema changed")
        scores = self.classifier.predict_proba(values)[:, 1]
        frame = prediction_frame(windows, scores, self.threshold, self.variant)
        frame["model_version"] = f"a1_invariant_v1/{self.variant}"
        frame["prediction_source"] = "train_fit_participant_oof_threshold"
        return frame

    def predict_frame(self, frame):
        sensors = frame.drop(columns=[c for c in ("fog", "activity", "fog_severity") if c in frame])
        return self.predict_windows(anchored_gyro_windows(sensors, require_right=True))
