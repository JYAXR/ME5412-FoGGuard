"""The saved-policy verifier must reject wrong budgets and causal decisions."""

import copy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_a1_event.py"
    spec = importlib.util.spec_from_file_location("event_verify_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_verifier_rejects_wrong_people_budget_and_infeasible_choice():
    module = verifier()
    config = {"threshold_grid": {"start": 0.3, "stop": 0.6, "count": 2},
              "confirmation_choices": [1, 2], "false_duration_budget_multiplier": 1.1,
              "require_control_time_coverage": True}
    options = [{"threshold": t, "confirmation_windows": c, "fog_f1": f, "balanced_accuracy": 0.8,
                "covered_raw_events": e, "false_alarm_s": fp, "true_alarm_s": tp}
               for t, c, f, e, fp, tp in [(0.3, 1, 0.7, 4, 100, 50), (0.3, 2, 0.65, 5, 105, 51),
                                         (0.6, 1, 0.6, 3, 50, 40), (0.6, 2, 0.5, 2, 10, 30)]]
    selection = {"selection_people": ["1"], "options": options, "control": options[0],
                 "selected": options[1], "false_alarm_budget_s": 110, "feasible_count": 2}
    module.verify_selection(selection, config, ["1"])
    with pytest.raises(ValueError, match="people"):
        module.verify_selection(selection, config, ["2"])
    wrong = copy.deepcopy(selection)
    wrong["false_alarm_budget_s"] = 200
    with pytest.raises(ValueError, match="budget"):
        module.verify_selection(wrong, config, ["1"])
    wrong = copy.deepcopy(selection)
    wrong["selected"] = options[-1]
    with pytest.raises(ValueError, match="choice"):
        module.verify_selection(wrong, config, ["1"])


def test_independent_gate_does_not_backdate_or_carry_across_gaps():
    module = verifier()
    frame = pd.DataFrame({"subject_id": 1, "session_id": 1, "task_id": [1, 1, 1, 1, 2],
                          "window_end_s": [2, 2.5, 4, 4.5, 5], "fog_probability": 0.9,
                          "fog_threshold": 0.3, "event_confirmation_windows": 2})
    assert module.independent_gate(frame).tolist() == [False, True, False, True, False]
