"""Auditable grouped inner validation for A1; outer test rows are never accepted."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold


def make_forest(config):
    return RandomForestClassifier(n_estimators=config["n_estimators"],
                                  min_samples_leaf=config["min_samples_leaf"],
                                  class_weight=config["class_weight"],
                                  random_state=config["random_state"], n_jobs=-1)


def grouped_threshold(x, y, groups, config):
    """Select a threshold using held-out training participants in 3 inner folds."""
    groups = np.asarray(groups).astype(str)
    y = np.asarray(y, dtype=int)
    probability = np.full(len(y), np.nan)
    seen = np.zeros(len(y), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True,
                                    random_state=config["random_state"])
    folds = []
    for training, validation in splitter.split(x, y, groups):
        train_people, val_people = set(groups[training]), set(groups[validation])
        if train_people & val_people or set(y[training]) != {0, 1}:
            raise ValueError("Leaking groups or missing training class")
        model = make_forest(config).fit(x[training], y[training])
        probability[validation] = model.predict_proba(x[validation])[:, 1]
        seen[validation] += 1
        folds.append({"train": sorted(train_people), "validation": sorted(val_people)})
    if not np.all(seen == 1) or not np.isfinite(probability).all():
        raise ValueError("Inner predictions must cover each training row exactly once")
    grid = config["threshold_grid"]
    candidates = np.linspace(grid["start"], grid["stop"], grid["count"])
    scores = []
    for threshold in candidates:
        pred = probability >= threshold
        scores.append({"threshold": float(threshold),
                       "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                       "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro",
                                                  zero_division=0))})
    selected = max(scores, key=lambda row: (row["balanced_accuracy"], row["macro_f1"],
                                          row["threshold"]))
    return selected["threshold"], {"folds": folds, "threshold_scores": scores,
                                    "selected": selected}
