"""Post-hoc diagnostics preserve time availability and distinguish truth references from ceilings."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fogguard.evaluation.external_failure import (
    event_opportunities,
    invariant_summaries,
    select_failures,
    window_groups,
)


def prediction_grid(a, b, score):
    start = np.arange(0, 8, 0.5)
    end = start + 119 / 60
    points = start[:, None] + np.arange(120)[None, :] / 60
    labels = ((points >= a - 1e-9) & (points < b - 1e-9)).mean(axis=1) >= 0.5
    return pd.DataFrame({"record_id": 1, "window_start_s": start, "window_end_s": end,
        "available_at_s": end + 1 / 6, "fog_probability": score, "is_fog": score >= 0.23,
        "truth_fog_label": labels})


def test_rotations_sign_flips_and_zeros_have_the_declared_invariants():
    x = np.random.default_rng(41).normal(size=(6, 120, 6))
    q = np.linalg.qr(np.random.default_rng(42).normal(size=(3, 3)))[0]
    moved = np.concatenate((x[:, :, :3] @ q, x[:, :, 3:] @ q.T), axis=2)
    np.testing.assert_allclose(invariant_summaries(x), invariant_summaries(moved), atol=1e-10)
    np.testing.assert_allclose(invariant_summaries(x), invariant_summaries(x * [-1, 1, -1, -1, 1, -1]), atol=1e-12)
    np.testing.assert_array_equal(invariant_summaries(np.zeros((1, 120, 6))), np.zeros((1, 10)))


def test_short_event_label_reference_is_not_a_model_performance_ceiling():
    a, b = 3.0, 3.8
    predictions = prediction_grid(a, b, 0.8)
    events = pd.DataFrame([{"record_id": 1, "source_person": 6, "start_s": a, "end_s": b,
                            "duration_s": b - a, "covered": True, "alarm_s": b - a}])
    row = event_opportunities(events, predictions).iloc[0]
    assert row.timely_self_majority_windows == 0 and not row.label_reference_covered
    assert row.covered and row.alarm_s == pytest.approx(0.8)


def test_long_event_can_be_missed_despite_timely_positive_weak_labels():
    a, b = 3.0, 7.0
    predictions = prediction_grid(a, b, 0.1)
    events = pd.DataFrame([{"record_id": 1, "source_person": 6, "start_s": a, "end_s": b,
                            "duration_s": b - a, "covered": False, "alarm_s": 0.0}])
    row = event_opportunities(events, predictions).iloc[0]
    assert row.timely_self_majority_windows > 0 and row.label_reference_covered
    assert row.missed_despite_self_majority and row.max_timely_score == pytest.approx(0.1)
    assert row.label_reference_overlap_s < b - a  # real availability, not a backfilled window.


def test_purity_boundary_and_fixed_failure_selection():
    assert list(window_groups([0, 0.49, 0.5, 1])) == ["pure_nonfog", "mixed", "mixed", "pure_fog"]
    events = pd.DataFrame({"source_person": [6, 6, 7], "record_id": [4, 3, 7], "start_s": [1., 3., 5.],
        "end_s": [10., 7., 8.], "duration_s": [9., 4., 3.], "covered": [False] * 3,
        "timely_self_majority_windows": [0, 2, 1]})
    selected = select_failures(events)
    assert [s["record_id"] for s in selected] == [3, 7]


def test_diagnostic_code_does_not_fit_predict_or_select_thresholds():
    root = Path(__file__).parents[1]
    for file in ("scripts/diagnose_stanford_failure.py", "src/fogguard/evaluation/external_failure.py"):
        tree = ast.parse((root / file).read_text("utf-8"))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in {"fit", "fit_transform", "predict", "predict_proba", "predict_windows"}]
