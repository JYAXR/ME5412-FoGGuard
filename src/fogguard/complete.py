"""Integrated simulation: original temporal integrator -> optional control -> saved outputs."""

from dataclasses import replace

import pandas as pd

from fogguard.context import FogStarTemporalIntegrator
from fogguard.cueing.march import MarchConfig, MarchGuidanceController
from fogguard.cueing.replay import replay_guidance
from fogguard.cueing.scenarios import _predictions, _steps, build_guidance_cases


def run_complete_pipeline(predictions, *, responses=None, config=None):
    """Predictions may be saved real-model candidates or explicitly constructed cases.

    Truth remains an evaluation annotation. This does not claim new online shared types.
    """
    events = FogStarTemporalIntegrator().process(predictions)
    commands = replay_guidance(events, responses=responses, config=config or MarchConfig(),
                               controller_type=MarchGuidanceController)
    return events, commands


def complete_cases():
    cases = build_guidance_cases()
    for name in ("march_recovery", "march_no_response", "march_missing_input",
                 "march_unknown_direction", "march_wrong_foot"):
        predictions = _predictions(name, positive=range(4, 21), count=48)
        responses = [{"scenario_id": name, "timestamp_s": 5.0, "foot": "none",
                      "forward_m": None, "lateral_m": None, "response_status": "no_response",
                      "delay_s": None, "reliability": 1.0, "source": "simulated"}]
        if name == "march_recovery":
            responses += _steps(name, [5.5, 6, 6.5, 7], lengths=[0, 0, 0, 0]).to_dict("records")
            responses += _steps(name, [7.5 + 0.5*i for i in range(14)]).to_dict("records")
        if name == "march_missing_input":
            responses += [{**responses[0], "timestamp_s": 5.5, "response_status": "missing_input"}]
        if name == "march_unknown_direction":
            predictions.loc[8:14, "direction"] = "other"
        if name == "march_wrong_foot":
            responses += [{**row, "foot": "right"} for row in _steps(
                name, [5.5, 6, 6.5, 7], lengths=[0, 0, 0, 0]).to_dict("records")]
        cases[name] = (FogStarTemporalIntegrator().process(predictions), pd.DataFrame(responses))
    return cases


def strategy_configs(base=None):
    full = base or MarchConfig(enable_adaptive_spacing=True)
    return {
        "fixed": replace(full, enable_response_policy=False, enable_fading=False,
                         enable_adaptive_spacing=False, enable_march_fallback=False),
        "countdown": replace(full, enable_march_fallback=False),
        "march": full,
        "no_fade": replace(full, enable_fading=False),
        "no_spacing": replace(full, enable_adaptive_spacing=False),
    }
