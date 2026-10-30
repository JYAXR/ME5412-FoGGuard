"""Optional within-person study using the unchanged bilateral RF representation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from fogguard.modeling.a1_bilateral import anchored_gyro_windows
from fogguard.modeling.a1_fog import a1_matrix, prediction_frame

KEYS = ["subject_id", "session_id", "task_id"]


def plan_blocks(records):
    """Split record identities before featurization; no labels or scores used.

    Multiple sessions use leave-one-session-out, otherwise leave-one-record-out.
    Session/record IDs are grouping keys, not verified chronological dates.
    """
    if not set(KEYS) <= set(records):
        raise ValueError("Record identities are required")
    if records[KEYS].isna().any().any():
        raise ValueError("Record identities must be present")
    records = records[KEYS].astype(str).drop_duplicates()
    folds = []
    for person, group in records.groupby("subject_id", sort=True):
        sessions = sorted(group.session_id.unique())
        multi = len(sessions) > 1
        blocks = sessions if multi else sorted(group.task_id.unique())
        for number, block in enumerate(blocks):
            mask = group.session_id.eq(block) if multi else group.task_id.eq(block)
            encode = lambda frame: [list(k) for k in frame.itertuples(index=False, name=None)]
            folds.append({"subject_id": person, "fold_id": f"p{person}-b{number + 1}",
                          "split_kind": "session_held_out" if multi else "record_held_out",
                          "adaptation_records": encode(group.loc[~mask]),
                          "evaluation_records": encode(group.loc[mask])})
    return folds


def record_mask(meta, identities):
    allowed = set(map(tuple, identities))
    return np.array([tuple(map(str, row)) in allowed for row in
                     meta[KEYS].itertuples(index=False, name=None)], dtype=bool)


def adaptation_status(meta, fold, config):
    adapt = record_mask(meta, fold["adaptation_records"])
    evaluate = record_mask(meta, fold["evaluation_records"])
    if np.any(adapt & evaluate):
        raise ValueError("Adaptation/evaluation records overlap")
    y = meta.loc[adapt, "truth_fog_label"].to_numpy(int)
    positive, negative = int(y.sum()), int((y == 0).sum())
    reason = "eligible"
    if not evaluate.any():
        reason = "no_supported_evaluation_windows"
    elif positive < config["minimum_adaptation_positive_windows"]:
        reason = "insufficient_adaptation_positive_windows"
    elif negative < config["minimum_adaptation_negative_windows"]:
        reason = "insufficient_adaptation_negative_windows"
    return {"reason": reason, "eligible": reason == "eligible",
            "adaptation_windows": len(y), "adaptation_positive_windows": positive,
            "adaptation_negative_windows": negative, "evaluation_windows": int(evaluate.sum())}


def choose_calibration(labels, scores, grid):
    """Caller must pass adaptation observations only, never held-out targets."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if set(labels) != {0, 1} or len(labels) != len(scores):
        raise ValueError("Calibration requires aligned positive and negative adaptation examples")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Calibration scores must be finite probabilities")
    candidates = []
    for threshold in np.linspace(grid["start"], grid["stop"], grid["count"]):
        predicted = scores >= threshold
        candidates.append({"threshold": float(threshold),
                           "f1": float(f1_score(labels, predicted, zero_division=0)),
                           "balanced_accuracy": float(balanced_accuracy_score(labels, predicted))})
    best = max(candidates, key=lambda r: (r["f1"], r["balanced_accuracy"], r["threshold"]))
    return best, candidates


@dataclass
class PersonalStudyModel:
    classifier: RandomForestClassifier
    feature_names: tuple[str, ...]
    threshold: float
    strategy: str
    fit_participants: tuple[str, ...]
    adaptation_records: tuple[tuple[str, ...], ...] = ()

    def predict_features(self, values):
        return self.classifier.predict_proba(values)[:, list(self.classifier.classes_).index(1)]

    def predict_frame(self, frame):
        sensors = frame.drop(columns=[c for c in ("fog", "activity", "fog_severity") if c in frame])
        windows = anchored_gyro_windows(sensors, require_right=True)
        values, names = a1_matrix(windows, "baseline_2s_v1")
        if names != self.feature_names:
            raise ValueError("Feature schema differs from the fitted model")
        result = prediction_frame(windows, self.predict_features(values), self.threshold,
                                  f"personal_study/{self.strategy}")
        result["prediction_source"] = f"exploratory_train_pool/{self.strategy}"
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix == ".json":
            calibration = json.loads(path.read_text("utf-8"))
            base = cls.load(path.parent / calibration["base_model"])
            return replace(base, threshold=calibration["selected"]["threshold"],
                           strategy="population_calibrated",
                           adaptation_records=tuple(map(tuple, calibration["adaptation_records"])))
        result = joblib.load(path)
        if not isinstance(result, cls):
            raise TypeError("Not a personal-study bilateral model")
        return result


def fit_model(values, labels, names, config, *, strategy, fit_people, adaptation_records=()):
    if set(np.asarray(labels, int)) != {0, 1}:
        raise ValueError("Binary FoG training needs both classes")
    forest = RandomForestClassifier(**config["rf"], random_state=config["random_state"])
    forest.fit(values, labels)
    return PersonalStudyModel(forest, tuple(names), config["fixed_threshold"], strategy,
                              tuple(map(str, fit_people)), tuple(map(tuple, adaptation_records)))
