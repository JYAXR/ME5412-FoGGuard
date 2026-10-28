"""Protect real FoG, ambiguous boundary labels and nested held-out participants."""

import json
from pathlib import Path

import numpy as np
import pytest

from fogguard.evaluation.a1_hard_negative import (
    fit_weighted,
    mining_scores,
    negative_weights,
    select_weighted_threshold,
)


def config():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "config/a1_hard_negative_v1.json")
                     .read_text(encoding="utf-8"))
    cfg["rf"]["n_estimators"] = 4
    cfg["threshold_grid"]["count"] = 5
    return cfg


def test_only_pure_confident_training_negatives_are_upweighted():
    y = np.array([0, 0, 1, 1, 0])
    fraction = np.array([0, 0.2, 0.6, 1.0, 0])
    score = np.array([0.8, 0.9, 0.9, 0.1, 0.1])
    weights, hard = negative_weights(y, fraction, score, config())
    np.testing.assert_array_equal(weights, [2, 1, 1, 1, 1])
    np.testing.assert_array_equal(hard, [True, False, False, False, False])


def test_mining_excludes_external_people_and_uses_held_out_groups():
    rng = np.random.default_rng(9)
    x, y, g = rng.normal(size=(48, 4)), np.tile([0, 1], 24), np.repeat(np.arange(6), 8)
    scores, folds = mining_scores(x, y, g, config(), forbidden=[99])
    assert np.isfinite(scores).all()
    seen = set()
    for fold in folds:
        assert not set(fold["train"]) & set(fold["validation"])
        assert "99" not in fold["train"] + fold["validation"]
        assert not seen & set(fold["validation"])
        seen |= set(fold["validation"])
    assert seen == set(map(str, range(6)))
    with pytest.raises(ValueError, match="Forbidden"):
        mining_scores(x, y, g, config(), forbidden=[1])


def test_nested_mining_never_sees_inner_validation_or_outer_test_people():
    rng = np.random.default_rng(10)
    x, y, g = rng.normal(size=(72, 4)), np.tile([0, 1], 36), np.repeat(np.arange(9), 8)
    _, audit = select_weighted_threshold(x, y, y.astype(float), g, "hard_negative", config(),
                                         outer_test_people=[99])
    for inner in audit["folds"]:
        forbidden = set(inner["validation"]) | {"99"}
        assert not forbidden & set(inner["training_audit"]["training_people"])
        for mine in inner["training_audit"]["mining_folds"]:
            assert not forbidden & set(mine["train"] + mine["validation"])


def test_mismatched_truth_fraction_and_unknown_variant_reject():
    with pytest.raises(ValueError, match="Invalid"):
        negative_weights([0, 1], [0, 0], [0.4, 0.7], config())
    with pytest.raises(ValueError, match="Unknown"):
        fit_weighted(np.zeros((2, 2)), np.array([0, 1]), np.array([0, 1]), [1, 2],
                     "unknown", config())
