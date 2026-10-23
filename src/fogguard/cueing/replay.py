"""CSV adapter shared by CLI, tests and presentation consumers; truth is not read."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from fogguard.cueing.contracts import GuidanceConfig, GuidanceInput, StepResponse
from fogguard.cueing.guidance import GuidanceController
from fogguard.data.responses import validate_step_responses

REQUIRED_EVENT_COLUMNS = {
    "subject_id", "session_id", "window_end_s", "event_id", "event_phase",
    "stable_fog_detected", "turn", "turn_confidence", "input_fresh",
}


def _number(value: object) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError("Expected a boolean, not a truthy string or missing value")


def replay_guidance(
    events: pd.DataFrame,
    *,
    config: GuidanceConfig | None = None,
    responses: pd.DataFrame | None = None,
    finalize: bool = True,
    controller_type: type[GuidanceController] = GuidanceController,
) -> pd.DataFrame:
    """Replay Robot 2 outputs. Responses match scenario_id, never truth_*.

    Timestamps in a response file are availability times, not scheduled future
    outcomes; delay_s is descriptive and is not added to timestamp_s again.
    A configured response stream is assumed available until missing_input or a
    low-reliability row explicitly suspends it. Silence may then trigger timeout.
    """
    missing = sorted(REQUIRED_EVENT_COLUMNS - set(events.columns))
    if missing:
        raise ValueError(f"Guidance events missing columns: {missing}")
    if events.empty:
        raise ValueError("Guidance replay requires at least one event row")
    events = events.copy()
    for name in ("subject_id", "session_id"):
        if events[name].isna().any() or events[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} must identify every event row")
    for name, default in (("task_id", "default"), ("scenario_id", "default")):
        if name not in events:
            events[name] = default
        if events[name].isna().any():
            raise ValueError(f"{name} must not contain missing identifiers")

    response_groups: dict[str, list[StepResponse]] = {}
    if responses is not None:
        responses = validate_step_responses(responses)
        for scenario, group in responses.groupby("scenario_id", sort=False):
            response_groups[str(scenario)] = [
                StepResponse(
                    timestamp_s=float(row.timestamp_s), foot=row.foot,
                    forward_m=None if pd.isna(row.forward_m) else float(row.forward_m),
                    lateral_m=None if pd.isna(row.lateral_m) else float(row.lateral_m),
                    response_status=row.response_status, reliability=float(row.reliability),
                ) for row in group.itertuples()
            ]
        unmatched = set(response_groups) - set(events["scenario_id"].astype(str))
        if unmatched:
            raise ValueError(f"Response scenarios have no event match: {sorted(unmatched)}")
        for scenario in response_groups:
            group = events.loc[events.scenario_id.astype(str).eq(scenario)]
            if len(group[["subject_id", "session_id", "task_id"]].drop_duplicates()) != 1:
                raise ValueError("Each response scenario must map to exactly one subject/session/task")

    controller = controller_type(config)
    sequence: tuple[str, ...] | None = None
    seen_sequences: set[tuple[str, ...]] = set()
    response_index = 0
    output = []
    last_row = None

    def append(row: dict, command, terminal: bool = False) -> None:
        record = dict(row)
        for name, value in command.as_dict().items():
            if name == "targets":
                value = json.dumps(value, ensure_ascii=False, allow_nan=False)
            record[f"guidance_{name}"] = value
        record["guidance_terminal"] = terminal
        output.append(record)

    for row in events.to_dict(orient="records"):
        current = tuple(str(row[name]) for name in (
            "subject_id", "session_id", "task_id", "scenario_id",
        ))
        if current != sequence:
            if current in seen_sequences:
                raise ValueError("Event sequences must be contiguous; do not interleave sessions")
            if last_row is not None:
                terminal = controller.finish()
                if terminal is not None:
                    append(last_row, terminal, True)
            seen_sequences.add(current)
            sequence = current
            response_index = 0
            controller = controller_type(
                config, responses_configured=current[-1] in response_groups,
            )
        now = _number(row["window_end_s"])
        due = []
        source = response_groups.get(current[-1], [])
        while response_index < len(source) and source[response_index].timestamp_s <= now:
            due.append(source[response_index])
            response_index += 1
        quality_reason = ""
        try:
            fresh = _boolean(row["input_fresh"])
            stable = _boolean(row["stable_fog_detected"])
        except ValueError:
            fresh = stable = False
            quality_reason = "invalid_boolean_field"
        event_id = row["event_id"]
        if pd.isna(event_id):
            event_id = None
        elif isinstance(event_id, (float, np.floating)) and event_id.is_integer():
            event_id = str(int(event_id))
        else:
            event_id = str(event_id)
        item = GuidanceInput(
            now_s=now, sequence=current, event_id=event_id,
            event_phase=str(row["event_phase"]), stable_fog_detected=stable,
            turn=str(row["turn"]), turn_confidence=_number(row["turn_confidence"]),
            input_fresh=fresh, input_age_s=_number(row.get("input_age_s", 0.0)),
            quality_reason=quality_reason,
        )
        append(row, controller.step(item, tuple(due)))
        last_row = row
    if finalize:
        terminal = controller.finish()
        if terminal is not None:
            append(last_row, terminal, True)
    return pd.DataFrame(output)


def summarize_guidance(result: pd.DataFrame) -> dict[str, object]:
    """Behavior counts, not clinical effects or FoG accuracy without truth."""
    starts = result.loc[result.guidance_action.eq("start")]
    stops = result.loc[result.guidance_action.eq("stop")]
    durations = []
    unmatched = 0
    for row in starts.itertuples():
        end = stops.loc[stops.guidance_command_id.eq(row.guidance_command_id)]
        if end.empty:
            unmatched += 1
        else:
            durations.append(max(0.0, float(end.iloc[0].guidance_timestamp_s)
                                 - row.guidance_timestamp_s))
    latencies = []
    unprompted = 0
    confirmed = result.loc[
        result.event_phase.eq("confirmed") & ~result.guidance_terminal
    ]
    group_keys = ["subject_id", "session_id", "task_id", "scenario_id", "event_id"]
    for key, group in confirmed.groupby(group_keys, dropna=False, sort=False):
        matching = starts
        for name, value in zip(group_keys[:-1], key[:-1], strict=True):
            matching = matching.loc[matching[name].eq(value)]
        matching = matching.loc[matching.event_id.eq(key[-1])]
        if matching.empty:
            unprompted += 1
        else:
            latencies.append(float(matching.iloc[0].window_end_s - group.window_end_s.iloc[0]))
    return {
        "input_rows": int((~result.guidance_terminal).sum()),
        "start_commands": len(starts), "stop_commands": len(stops),
        "active_duration_s": round(sum(durations), 6), "unclosed_commands": unmatched,
        "confirmed_events_without_start": unprompted,
        "mean_confirmed_to_start_delay_s": (
            round(sum(latencies) / len(latencies), 6) if latencies else None
        ),
        "stop_reasons": dict(Counter(stops.guidance_reason)),
        "states": list(dict.fromkeys(result.guidance_control_state)),
        "scope": "engineering behavior; synthetic responses do not establish efficacy",
    }
