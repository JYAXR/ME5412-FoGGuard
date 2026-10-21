"""Explicitly constructed model candidates and responses for strategy acceptance.

These are not sensor/model accuracy experiments. All candidates pass through
Robot 2's actual temporal integrator; truth columns are only annotations.
"""

from __future__ import annotations

import pandas as pd

from fogguard.context import FogStarTemporalIntegrator


def _predictions(
    name: str, *, direction: str = "straight", positive: range = range(4, 17),
    count: int = 48,
) -> pd.DataFrame:
    times = [2 + index * 0.5 for index in range(count)]
    return pd.DataFrame({
        "subject_id": f"constructed-{name}", "session_id": "demo", "task_id": name,
        "scenario_id": name, "source_dataset": "constructed_guidance_case",
        "window_start_s": [value - 2 for value in times], "window_end_s": times,
        "fog_probability": [0.85 if index in positive else 0.1 for index in range(count)],
        "fog_threshold": 0.5, "is_fog": [index in positive for index in range(count)],
        "direction": direction, "direction_confidence": 0.9,
        "truth_fog_label": [int(index in positive) for index in range(count)],
    })


def _steps(name: str, times: list[float], *, lengths: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame([
        {"scenario_id": name, "timestamp_s": time, "foot": "left" if index % 2 == 0 else "right",
         "forward_m": lengths[index] if lengths else 0.40,
         "lateral_m": 0.10 if index % 2 == 0 else -0.10,
         "response_status": "success", "delay_s": 0.0, "reliability": 1.0,
         "source": "simulated"}
        for index, time in enumerate(times)
    ])


def build_guidance_cases(
    provided_responses: pd.DataFrame | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame | None]]:
    """Return stable event timelines plus separately identified simulated responses."""
    raw = {
        "normal": _predictions("normal", positive=range(0)),
        "normal_right_turn": _predictions("normal_right_turn", direction="right", positive=range(0)),
        "straight_fog": _predictions("straight_fog"),
        "left_fog": _predictions("left_fog", direction="left"),
        "right_fog": _predictions("right_fog", direction="right"),
        "voluntary_stop_observation": _predictions("voluntary_stop_observation"),
        "fog_recovery": _predictions("fog_recovery", positive=range(4, 13)),
        "single_candidate": _predictions("single_candidate", positive=range(4, 5)),
        "short_dip": _predictions("short_dip", positive=range(4, 19)),
        "unknown_direction": _predictions("unknown_direction"),
        "stale_input": _predictions("stale_input"),
        "successful_recovery": _predictions("successful_recovery", positive=range(4, 13)),
        "refreeze_during_fade": _predictions("refreeze_during_fade", positive=range(4, 21)),
        "persistent_no_response": _predictions("persistent_no_response", positive=range(4, 43)),
        "adaptive_spacing": _predictions("adaptive_spacing", positive=range(4, 19)),
        "recovery_interrupted": _predictions("recovery_interrupted", positive=range(4, 13)),
    }
    raw["voluntary_stop_observation"]["motion_state"] = "stationary"
    raw["voluntary_stop_observation"]["motion_confidence"] = 0.9
    raw["voluntary_stop_observation"]["truth_fog_label"] = 0
    raw["short_dip"].loc[10, ["fog_probability", "is_fog"]] = [0.1, False]
    raw["refreeze_during_fade"].loc[11:12, ["fog_probability", "is_fog"]] = [0.1, False]
    raw["unknown_direction"].loc[9:13, "direction"] = "other"
    raw["stale_input"]["received_at_s"] = raw["stale_input"]["window_end_s"]
    raw["stale_input"].loc[10, "received_at_s"] += 2.0
    responses = {
        "successful_recovery": _steps("successful_recovery", [5 + 0.5 * i for i in range(16)]),
        "refreeze_during_fade": _steps("refreeze_during_fade", [5 + 0.5 * i for i in range(24)]),
        "persistent_no_response": pd.DataFrame([{
            "scenario_id": "persistent_no_response", "timestamp_s": 5.0,
            "foot": "none", "forward_m": None, "lateral_m": None,
            "response_status": "no_response", "delay_s": None,
            "reliability": 1.0, "source": "simulated",
        }]),
        "adaptive_spacing": _steps(
            "adaptive_spacing", [5 + 0.5 * i for i in range(16)],
            lengths=[0.48, 0.50, 0.52, 0.50] * 4,
        ),
        "recovery_interrupted": pd.DataFrame([
            *_steps("recovery_interrupted", [5 + 0.5 * i for i in range(8)]).to_dict("records"),
            {
                "scenario_id": "recovery_interrupted", "timestamp_s": 9.0,
                "foot": "none", "forward_m": None, "lateral_m": None,
                "response_status": "unable", "delay_s": None,
                "reliability": 1.0, "source": "simulated",
            },
        ]),
    }
    if provided_responses is not None:
        for scenario, group in provided_responses.groupby("scenario_id", sort=False):
            # Original timestamps are 12.5–14.8 s: use a matching candidate timeline.
            raw[scenario] = _predictions(scenario, positive=range(20, 31), count=57)
            responses[scenario] = group.copy()
    return {
        name: (FogStarTemporalIntegrator().process(frame), responses.get(name))
        for name, frame in raw.items()
    }
