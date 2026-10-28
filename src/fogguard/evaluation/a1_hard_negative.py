"""Participant-cross-fitted hard-negative weighting, isolated inside training data."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from fogguard.modeling.a1_multimodel import make_classifier, model_scores


def require_training_only(groups, forbidden):
    people = set(np.asarray(groups).astype(str))
    if people & set(map(str, forbidden)):
        raise ValueError("Forbidden validation/test participant entered training or mining")
    return people


def mining_scores(values, labels, groups, config, *, forbidden=()):
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels, dtype=int)
    people = require_training_only(groups, forbidden)
    scores, seen = np.full(len(labels), np.nan), np.zeros(len(labels), int)
    splitter = StratifiedGroupKFold(n_splits=config["mining_folds"], shuffle=True,
                                    random_state=config["random_state"])
    audit = []
    for training, validation in splitter.split(values, labels, groups):
        a, b = set(groups[training]), set(groups[validation])
        if a & b or a | b != people or set(labels[training]) != {0, 1}:
            raise ValueError("Invalid mining participant split or missing class")
        model = make_classifier("rf", config).fit(values[training], labels[training])
        scores[validation] = model_scores(model, "rf", values[validation])
        seen[validation] += 1
        audit.append({"train": sorted(a), "validation": sorted(b)})
    if not np.all(seen == 1) or not np.isfinite(scores).all():
        raise ValueError("Each mining row needs exactly one participant-held-out score")
    return scores, audit


def negative_weights(labels, fractions, scores, config):
    labels, fractions, scores = (np.asarray(a) for a in (labels, fractions, scores))
    if not (len(labels) == len(fractions) == len(scores)):
        raise ValueError("Weight inputs must align")
    if (not np.isfinite(fractions).all() or not np.isfinite(scores).all()
            or not np.isin(labels, [0, 1]).all()
            or not ((fractions >= 0) & (fractions <= 1)).all()
            or not ((scores >= 0) & (scores <= 1)).all()
            or not np.array_equal(labels, (fractions >= 0.5).astype(int))):
        raise ValueError("Invalid labels, fractions or mining scores")
    if config["negative_weight"] < 1:
        raise ValueError("Negative weight must be at least one")
    hard = (labels == 0) & (fractions == 0) & (scores >= config["mining_threshold"])
    weights = np.ones(len(labels), float)
    weights[hard] = config["negative_weight"]
    return weights, hard


def fit_weighted(values, labels, fractions, groups, variant, config, *, forbidden=()):
    people = require_training_only(groups, forbidden)
    audit = {"training_people": sorted(people), "forbidden_people": sorted(map(str, forbidden))}
    if variant == "baseline":
        return make_classifier("rf", config).fit(values, labels), audit, None
    if variant != "hard_negative":
        raise ValueError("Unknown weighting variant")
    scores, folds = mining_scores(values, labels, groups, config, forbidden=forbidden)
    weights, hard = negative_weights(labels, fractions, scores, config)
    model = make_classifier("rf", config).fit(values, labels, sample_weight=weights)
    audit.update(mining_folds=folds, weighted_negative_windows=int(hard.sum()),
                 training_windows=len(labels), negative_weight=config["negative_weight"])
    return model, audit, {"score": scores, "weight": weights, "hard": hard}


def select_weighted_threshold(values, labels, fractions, groups, variant, config,
                              *, outer_test_people=()):
    groups = np.asarray(groups).astype(str)
    require_training_only(groups, outer_test_people)
    scores, seen = np.full(len(labels), np.nan), np.zeros(len(labels), int)
    splitter = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True,
                                    random_state=config["random_state"])
    audits = []
    for train, validation in splitter.split(values, labels, groups):
        val_people = sorted(set(groups[validation]))
        forbidden = sorted(set(map(str, outer_test_people)) | set(val_people))
        model, audit, _ = fit_weighted(values[train], labels[train], fractions[train],
                                       groups[train], variant, config, forbidden=forbidden)
        scores[validation] = model_scores(model, "rf", values[validation])
        seen[validation] += 1
        audits.append({"train": sorted(set(groups[train])), "validation": val_people,
                       "training_audit": audit})
    if not np.all(seen == 1) or not np.isfinite(scores).all():
        raise ValueError("Incomplete grouped inner predictions")
    grid = config["threshold_grid"]
    rows = []
    for threshold in np.linspace(grid["start"], grid["stop"], grid["count"]):
        decision = scores >= threshold
        rows.append({"threshold": float(threshold),
                     "fog_f1": float(f1_score(labels, decision, zero_division=0)),
                     "balanced_accuracy": float(balanced_accuracy_score(labels, decision))})
    selected = max(rows, key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
    return selected["threshold"], {"folds": audits, "threshold_scores": rows, "selected": selected}
