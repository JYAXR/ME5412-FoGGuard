"""Saved search must be backed by actual inner scores and disjoint people."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score


def verifier():
    spec = importlib.util.spec_from_file_location("verify_posture_test", Path(__file__).resolve().parents[1]
                                                 / "scripts/verify_a1_posture_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample():
    frame = pd.DataFrame({"subject_id": [1, 1, 2, 2], "truth_fog_label": [0, 0, 1, 1],
                          "score__a": [0.1, 0.4, 0.6, 0.9], "score__b": [0.6, 0.1, 0.8, 0.2]})
    cfg = {"candidates": [{"id": "a", "features": "x"}, {"id": "b", "features": "y"}],
           "threshold_grid": {"start": 0.25, "stop": 0.75, "count": 3}}
    rows = []
    for candidate in cfg["candidates"]:
        score, truth = frame[f"score__{candidate['id']}"], frame.truth_fog_label
        options = [{"threshold": float(t), "fog_f1": float(f1_score(truth, score >= t)),
                    "balanced_accuracy": float(balanced_accuracy_score(truth, score >= t))}
                   for t in np.linspace(0.25, 0.75, 3)]
        rows.append({"candidate_id": candidate["id"], "features": candidate["features"],
                     "threshold_scores": options, "selected_threshold": max(options,
                       key=lambda r: (r["fog_f1"], r["balanced_accuracy"], r["threshold"])),
                     "average_precision": float(average_precision_score(truth, score))})
    return cfg, frame, {"training_people": ["1", "2"], "candidates": rows, "selected_candidate_id": "a"}


def test_verifier_rejects_unbacked_metrics_and_wrong_configuration_choice():
    cfg, frame, audit = sample()
    module = verifier()
    module.verify_candidates(audit, frame, cfg)
    wrong = copy.deepcopy(audit)
    wrong["selected_candidate_id"] = "b"
    with pytest.raises(ValueError, match="configuration choice"):
        module.verify_candidates(wrong, frame, cfg)
    wrong = frame.copy()
    wrong["score__a"] = 0.9
    with pytest.raises(ValueError, match="metric mismatch"):
        module.verify_candidates(audit, wrong, cfg)


def test_verifier_rejects_test_people_in_inner_partition():
    module = verifier()
    audit = {"training_people": ["1", "2"], "inner_folds": [
        {"train": ["1"], "validation": ["2"]}, {"train": ["2"], "validation": ["1"]}]}
    module.verify_groups(audit, {"1", "2"}, {"3"})
    with pytest.raises(ValueError, match="training people"):
        module.verify_groups(audit, {"1", "2"}, {"2"})
    audit["inner_folds"][0]["train"].append("2")
    with pytest.raises(ValueError, match="Leaking"):
        module.verify_groups(audit, {"1", "2"}, {"3"})
