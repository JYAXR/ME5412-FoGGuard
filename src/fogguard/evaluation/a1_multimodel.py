"""Identical grouped inner folds and an explicit FoG-F1 threshold objective."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from fogguard.modeling.a1_multimodel import make_classifier, model_scores


def inner_grouped_scores(values, labels, groups, family, config):
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True,
                                    random_state=config["random_state"])
    scores, seen = np.full(len(labels), np.nan), np.zeros(len(labels), dtype=int)
    audits = []
    for training, validation in splitter.split(values, labels, groups):
        train_people, validation_people = set(groups[training]), set(groups[validation])
        if train_people & validation_people or set(labels[training]) != {0, 1}:
            raise ValueError("Group leakage or missing binary class")
        model = make_classifier(family, config).fit(values[training], labels[training])
        scores[validation] = model_scores(model, family, values[validation])
        seen[validation] += 1
        row = {"train": sorted(train_people), "validation": sorted(validation_people)}
        if family == "cnn":
            row["training_loss"] = model.loss_history_
            row["parameters"] = model.parameter_count_
        audits.append(row)
    if not np.all(seen == 1) or not np.isfinite(scores).all():
        raise ValueError("Incomplete grouped inner OOF")
    grid = config["threshold_grid"]
    results = []
    for threshold in np.linspace(grid["start"], grid["stop"], grid["count"]):
        decision = scores >= threshold
        results.append({"threshold": float(threshold),
                        "fog_f1": float(f1_score(labels, decision, zero_division=0)),
                        "balanced_accuracy": float(balanced_accuracy_score(labels, decision))})
    selected = max(results, key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
    return selected["threshold"], {"folds": audits, "threshold_scores": results,
                                    "selected": selected}
