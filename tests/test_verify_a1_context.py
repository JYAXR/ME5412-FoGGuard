"""Corrupted historical references must fail the independent CSV audit."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load_verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_a1_context.py"
    spec = importlib.util.spec_from_file_location("verify_context_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame():
    return pd.DataFrame({"subject_id": [1] * 9 + [2] * 5, "session_id": 1, "task_id": 1,
                         "window_start_s": np.arange(14) * 0.5,
                         "window_end_s": np.arange(14) * 0.5 + 119 / 60,
                         "lag_2s_row": [-1] * 4 + list(range(5)) + [-1] * 4 + [9],
                         "lag_4s_row": [-1] * 8 + [0] + [-1] * 5})


def test_independent_verifier_rejects_future_and_cross_person_links():
    verifier = load_verifier()
    original = frame()
    verifier.verify_history(original)
    wrong = original.copy()
    wrong.loc[8, "lag_4s_row"] = 9
    with pytest.raises(ValueError, match="history reference"):
        verifier.verify_history(wrong)
    wrong = original.copy()
    wrong.loc[9, "lag_2s_row"] = 5
    with pytest.raises(ValueError, match="history reference"):
        verifier.verify_history(wrong)


def test_independent_verifier_rejects_bad_time_and_duplicate():
    verifier = load_verifier()
    wrong = frame()
    wrong.loc[8, "window_end_s"] += 1
    with pytest.raises(ValueError, match="timing"):
        verifier.verify_history(wrong)
    wrong = frame()
    wrong.loc[8, ["window_start_s", "window_end_s"]] = wrong.loc[
        7, ["window_start_s", "window_end_s"]].to_numpy()
    with pytest.raises(ValueError, match="Duplicate"):
        verifier.verify_history(wrong)
