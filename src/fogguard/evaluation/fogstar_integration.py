"""Event and direction metrics for Robot 2 FoG-STAR integration output."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _runs(
    rows: pd.DataFrame,
    positive: Iterable[bool],
    *,
    maximum_gap_s: float,
) -> list[dict[str, float]]:
    flags = list(positive)
    times = rows["window_end_s"].to_numpy(dtype=float)
    result: list[dict[str, float]] = []
    start_index: int | None = None
    for index, is_positive in enumerate(flags):
        starts_after_gap = index > 0 and times[index] - times[index - 1] > maximum_gap_s
        if start_index is not None and (not is_positive or starts_after_gap):
            result.append(
                {"start_s": float(times[start_index]), "end_s": float(times[index - 1])}
            )
            start_index = None
        if is_positive and start_index is None:
            start_index = index
    if start_index is not None:
        result.append({"start_s": float(times[start_index]), "end_s": float(times[-1])})
    return result


def evaluate_fogstar_integration(
    result: pd.DataFrame,
    *,
    post_event_grace_s: float = 2.0,
    maximum_gap_s: float = 0.75,
) -> dict[str, object]:
    """Evaluate confirmed events from subject-held-out window predictions.

    Truth events are formed from consecutive truth-positive windows. Delay is
    measured from the first truth-positive window's output time to the first
    confirmed output, rather than claiming sample-accurate clinical latency.
    """
    required = {
        "window_start_s",
        "window_end_s",
        "subject_id",
        "session_id",
        "truth_fog_label",
        "truth_turn_label",
        "event_id",
        "event_phase",
        "turn",
        "cue",
        "is_fog",
        "fog_suppressed",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"Integrated FoG-STAR results are missing columns: {missing}")
    if post_event_grace_s < 0:
        raise ValueError("post_event_grace_s must not be negative")

    group_columns = ["subject_id", "session_id"]
    if "task_id" in result:
        group_columns.append("task_id")
    truth_events: list[dict[str, object]] = []
    predicted_events: list[dict[str, object]] = []
    recording_seconds = 0.0

    for key, group in result.groupby(group_columns, sort=False, dropna=False):
        ordered = group.sort_values("window_end_s")
        key_tuple = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, map(str, key_tuple), strict=True))
        recording_seconds += max(
            0.0,
            float(ordered["window_end_s"].iloc[-1] - ordered["window_start_s"].iloc[0]),
        )
        for event in _runs(
            ordered,
            ordered["truth_fog_label"].astype(int).eq(1),
            maximum_gap_s=maximum_gap_s,
        ):
            truth_events.append({**identity, **event})

        active = ordered[ordered["event_id"].notna()]
        for event_id, event_rows in active.groupby("event_id", sort=False):
            confirmed = event_rows[event_rows["event_phase"] == "confirmed"]
            if confirmed.empty:
                continue
            predicted_events.append(
                {
                    **identity,
                    "event_id": int(event_id),
                    "confirm_s": float(confirmed["window_end_s"].iloc[0]),
                    "end_s": float(event_rows["window_end_s"].iloc[-1]),
                }
            )

    used_predictions: set[int] = set()
    delays: list[float] = []
    event_details: list[dict[str, object]] = []
    for truth in truth_events:
        match_index = next(
            (
                index
                for index, prediction in enumerate(predicted_events)
                if index not in used_predictions
                and all(prediction[name] == truth[name] for name in group_columns)
                and float(prediction["confirm_s"])
                <= float(truth["end_s"]) + post_event_grace_s
                and float(prediction["end_s"]) >= float(truth["start_s"])
            ),
            None,
        )
        if match_index is None:
            event_details.append({**truth, "detected": False, "delay_s": None})
            continue
        used_predictions.add(match_index)
        delay = max(
            0.0,
            float(predicted_events[match_index]["confirm_s"]) - float(truth["start_s"]),
        )
        delays.append(delay)
        event_details.append({**truth, "detected": True, "delay_s": delay})

    false_events = len(predicted_events) - len(used_predictions)
    evaluated_direction = result[result["truth_turn_label"].isin(["straight", "left", "right"])]
    direction_coverage = float((evaluated_direction["turn"] != "unknown").mean())
    covered = evaluated_direction[evaluated_direction["turn"] != "unknown"]
    direction_macro_f1 = (
        float(
            f1_score(
                covered["truth_turn_label"],
                covered["turn"],
                labels=["straight", "left", "right"],
                average="macro",
                zero_division=0,
            )
        )
        if not covered.empty
        else 0.0
    )
    direction_comparison: dict[str, object] = {
        "final_direction_coverage": direction_coverage,
        "final_direction_macro_f1_when_accepted": direction_macro_f1,
    }
    if {"yaw_turn", "yaw_turn_confidence"} <= set(result.columns):
        yaw_covered = evaluated_direction[evaluated_direction["yaw_turn"] != "unknown"]
        direction_comparison.update(
            {
                "yaw_baseline_coverage": float(len(yaw_covered) / len(evaluated_direction)),
                "yaw_baseline_macro_f1_when_accepted": float(
                    f1_score(
                        yaw_covered["truth_turn_label"],
                        yaw_covered["yaw_turn"],
                        labels=["straight", "left", "right"],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "yaw_baseline_default_role": "comparison_only",
            }
        )
    non_fog = result[result["truth_fog_label"].astype(int).eq(0)]
    cue_summary: dict[str, object] = {
        "n_cues": int(result["cue"].ne("none").sum()),
        "n_cues_on_non_fog_windows": int(non_fog["cue"].ne("none").sum()),
        "n_raw_fog_windows_suppressed_as_stationary": int(result["fog_suppressed"].sum()),
    }
    if "truth_activity_label" in result:
        non_fog_stand = non_fog[non_fog["truth_activity_label"].eq(3)]
        all_stand = result[result["truth_activity_label"].eq(3)]
        all_walk = result[result["truth_activity_label"].eq(1)]
        cue_summary.update(
            {
                "stationary_proxy_rate_on_stand_windows": float(
                    all_stand["motion_state"].eq("stationary").mean()
                ),
                "stationary_proxy_rate_on_walk_windows": float(
                    all_walk["motion_state"].eq("stationary").mean()
                ),
                "n_non_fog_stand_windows": len(non_fog_stand),
                "n_non_fog_stand_raw_fog_windows": int(non_fog_stand["is_fog"].sum()),
                "n_non_fog_stand_suppressed_windows": int(
                    non_fog_stand["fog_suppressed"].sum()
                ),
                "n_cues_on_non_fog_stand_windows": int(
                    non_fog_stand["cue"].ne("none").sum()
                ),
            }
        )
    return {
        "scope": "subject-held-out FoG-STAR window predictions after causal integration",
        "latency_definition": "first confirmed output minus first truth-positive window output",
        "n_truth_events": len(truth_events),
        "n_detected_events": len(delays),
        "n_missed_events": len(truth_events) - len(delays),
        "event_recall": float(len(delays) / len(truth_events)) if truth_events else 0.0,
        "n_predicted_events": len(predicted_events),
        "n_false_positive_events": false_events,
        "false_positive_events_per_minute": (
            float(false_events / (recording_seconds / 60.0)) if recording_seconds else 0.0
        ),
        "mean_confirmation_delay_s": float(np.mean(delays)) if delays else None,
        "median_confirmation_delay_s": float(np.median(delays)) if delays else None,
        "direction_coverage": direction_coverage,
        "direction_macro_f1_when_accepted": direction_macro_f1,
        "direction_comparison": direction_comparison,
        "cue_summary": cue_summary,
        "events": event_details,
    }
