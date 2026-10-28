"""Nested participant-only feature/model/threshold selection, fixed before testing."""

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from fogguard.modeling.a1_posture import search_classifier


def nested_candidate_scores(matrices, labels, groups, config, *, forbidden=(), progress=None):
    groups, labels = np.asarray(groups).astype(str), np.asarray(labels, int)
    if set(groups) & set(map(str, forbidden)):
        raise ValueError("Test people entered inner selection")
    splitter = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True,
                                    random_state=config["random_state"])
    splits = list(splitter.split(next(iter(matrices.values())), labels, groups))
    folds = [{"train": sorted(set(groups[a])), "validation": sorted(set(groups[b]))} for a, b in splits]
    candidates, score_arrays = [], {}
    grid = config["threshold_grid"]
    for candidate in config["candidates"]:
        if progress:
            progress(candidate["id"])
        x = matrices[candidate["features"]]
        scores, seen = np.full(len(labels), np.nan), np.zeros(len(labels), int)
        for train, validation in splits:
            if set(groups[train]) & set(groups[validation]) or set(labels[train]) != {0, 1}:
                raise ValueError("Inner leakage or missing class")
            model = search_classifier(candidate, config["random_state"]).fit(x[train], labels[train])
            scores[validation] = model.predict_proba(x[validation])[:, 1]
            seen[validation] += 1
        if not np.all(seen == 1) or not np.isfinite(scores).all():
            raise ValueError("Incomplete candidate OOF")
        options = []
        for threshold in np.linspace(grid["start"], grid["stop"], grid["count"]):
            decision = scores >= threshold
            options.append({"threshold": float(threshold), "fog_f1": float(f1_score(labels, decision, zero_division=0)),
                            "balanced_accuracy": float(balanced_accuracy_score(labels, decision))})
        best = max(options, key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"]))
        candidates.append({"candidate_id": candidate["id"], "features": candidate["features"],
                           "selected_threshold": best, "threshold_scores": options,
                           "average_precision": float(average_precision_score(labels, scores))})
        score_arrays[candidate["id"]] = scores
    selected = max(enumerate(candidates), key=lambda pair: (
        pair[1]["selected_threshold"]["fog_f1"], pair[1]["average_precision"], -pair[0]))[1]
    return {"training_people": sorted(set(groups)), "inner_folds": folds,
            "candidates": candidates, "selected_candidate_id": selected["candidate_id"]}, score_arrays
