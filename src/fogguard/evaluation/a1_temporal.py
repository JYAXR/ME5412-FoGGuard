"""TRAIN-only participant checkpoint plans, independently of held-out labels."""

import numpy as np
from sklearn.model_selection import train_test_split


def nested_plan(groups, labels, folds, *, checkpoint_people, seed):
    groups, labels = np.asarray(groups, int), np.asarray(labels, int)
    plans, held_seen = [], []
    for fold in folds:
        fit_people = sorted(map(int, fold["fit_people"]))
        held_people = sorted(map(int, fold["held_out_people"]))
        if set(fit_people) & set(held_people) or set(fit_people) | set(held_people) != set(groups):
            raise ValueError("Outer folds must partition TRAIN people")
        # No held-out labels enter the inner participant split.
        strata = [int(labels[groups == p].any()) for p in fit_people]
        train, checkpoint = train_test_split(fit_people, test_size=checkpoint_people, stratify=strata,
                                             random_state=seed + int(fold["fold"]))
        train, checkpoint = sorted(map(int, train)), sorted(map(int, checkpoint))
        if set(labels[np.isin(groups, train)]) != {0, 1} or set(labels[np.isin(groups, checkpoint)]) != {0, 1}:
            raise ValueError("Missing class in inner fit or checkpoint; do not redraw")
        plans.append({"fold": int(fold["fold"]), "fit_people": fit_people, "held_out_people": held_people,
                      "inner_fit_people": train, "checkpoint_people": checkpoint})
        held_seen.extend(held_people)
    if sorted(held_seen) != sorted(set(groups)):
        raise ValueError("Each TRAIN person must be held out exactly once")
    return plans


def person_indices(groups, people):
    return np.flatnonzero(np.isin(np.asarray(groups, int), people))


def assert_history_isolation(links, groups, selected, allowed):
    selected_links = links[selected]
    sources = selected_links[selected_links >= 0]
    if not set(np.asarray(groups)[sources]) <= set(allowed):
        raise ValueError("History includes a forbidden participant")
