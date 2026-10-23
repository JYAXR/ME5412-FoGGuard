"""Additive Automation 2 evaluation; existing recognition/control is unchanged.

Window labels are evaluation annotations only. No metric is fed to a controller.
Latency starts at the first truth-positive WINDOW OUTPUT, not clinical onset.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

SEQUENCE = ["subject_id", "session_id", "task_id", "scenario_id"]


def confusion(truth, prediction) -> dict:
    """Explicit counts; absent denominators are unavailable rather than zero."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Nonempty, equally shaped label vectors required")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Labels must be binary")
    tp = int(((truth == 1) & (prediction == 1)).sum())
    tn = int(((truth == 0) & (prediction == 0)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    return {"windows": len(truth), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def truth_runs(group: pd.DataFrame, maximum_gap_s: float = 0.75) -> list[dict]:
    """Contiguous positive output windows within ONE sequence."""
    times = group.window_end_s.to_numpy(dtype=float)
    if len(times) > 1 and (np.diff(times) <= 0).any():
        raise ValueError("Evaluation input must be strictly chronological within a sequence")
    runs = []
    current = None
    for time, label in zip(times, group.truth_fog_label.astype(int), strict=True):
        if current is not None and (not label or time - current["end_s"] > maximum_gap_s):
            runs.append(current)
            current = None
        if label:
            if current is None:
                current = {"start_s": float(time), "end_s": float(time)}
            current["end_s"] = float(time)
    if current is not None:
        runs.append(current)
    return runs


def match_onsets(truth_events: list[dict], starts: list[float], grace_s: float = 2.0) -> dict:
    """One-to-one onset matching; an earlier false alarm cannot become a hit.

    Only a start inside [truth_start, truth_end + grace] matches. Delays are
    output-window delays. This is stricter than upstream overlap-based matching.
    """
    if grace_s < 0 or not np.isfinite(grace_s):
        raise ValueError("grace_s must be finite and nonnegative")
    used = set()
    details = []
    for event in truth_events:
        candidates = [(i, time) for i, time in enumerate(starts) if i not in used
                      and event["start_s"] <= time <= event["end_s"] + grace_s]
        match = min(candidates, key=lambda item: item[1]) if candidates else None
        if match is not None:
            used.add(match[0])
        details.append({**event, "detected": match is not None,
                        "delay_s": match[1] - event["start_s"] if match else None})
    delays = [row["delay_s"] for row in details if row["detected"]]
    return {"truth_events": len(truth_events), "hits": len(used),
            "misses": len(truth_events) - len(used),
            "unmatched_starts": len(starts) - len(used),
            "mean_delay_s": float(np.mean(delays)) if delays else None,
            "details": details}


def evaluate_timeline(events: pd.DataFrame, commands: pd.DataFrame) -> dict:
    """Evaluate real input rows separately from synthetic terminal stop records."""
    required = {*SEQUENCE, "window_end_s", "truth_fog_label", "is_fog",
                "stable_fog_detected", "event_phase", "event_id"}
    if not required <= set(events):
        raise ValueError(f"Missing evaluation columns: {sorted(required - set(events))}")
    windows = commands.loc[~commands.guidance_terminal].copy()
    if len(windows) != len(events):
        raise ValueError("Every event input must have exactly one nonterminal command row")
    expected = events.reset_index(drop=True)
    actual = windows.reset_index(drop=True)
    for name in [*SEQUENCE, "window_end_s"]:
        if not expected[name].equals(actual[name]):
            raise ValueError(f"Input/command alignment mismatch: {name}")
    raw = confusion(events.truth_fog_label, events.is_fog)
    stable = confusion(events.truth_fog_label, events.stable_fog_detected)
    n_truth = n_hits = n_missed = false_events = 0
    cue_hits = cue_false = 0
    delays = []
    cue_delays = []
    details = []
    for key, group in events.groupby(SEQUENCE, sort=False, dropna=False):
        truth = truth_runs(group)
        confirmed = group.loc[group.event_phase.eq("confirmed")]
        onsets = confirmed.groupby("event_id", sort=False).window_end_s.first().tolist()
        matched = match_onsets(truth, onsets)
        n_truth += matched["truth_events"]
        n_hits += matched["hits"]
        n_missed += matched["misses"]
        false_events += matched["unmatched_starts"]
        delays.extend(row["delay_s"] for row in matched["details"] if row["detected"])
        chosen = commands
        for name, value in zip(SEQUENCE, key, strict=True):
            chosen = chosen.loc[chosen[name].eq(value)]
        starts = chosen.loc[chosen.guidance_action.eq("start")]
        cue_match = match_onsets(truth, starts.window_end_s.tolist())
        cue_hits += cue_match["hits"]
        cue_false += cue_match["unmatched_starts"]
        cue_delays.extend(row["delay_s"] for row in cue_match["details"] if row["detected"])
        for detection, guidance in zip(matched["details"], cue_match["details"], strict=True):
            active_truth = chosen.loc[
                ~chosen.guidance_terminal & chosen.guidance_visual_enabled
                & chosen.window_end_s.between(detection["start_s"], detection["end_s"])]
            details.append({**dict(zip(SEQUENCE, key, strict=True)), **detection,
                            "guidance_start_matched": guidance["detected"],
                            "guidance_start_delay_s": guidance["delay_s"],
                            "guidance_active_truth_windows": len(active_truth)})
    starts = windows.loc[windows.guidance_action.eq("start")]
    stopped = commands.loc[commands.guidance_action.eq("stop")]
    return {"raw": raw, "stable": stable, "truth_events": n_truth,
            "confirmed_event_hits": n_hits, "confirmed_event_misses": n_missed,
            "unmatched_confirmed_event_starts": false_events,
            "mean_confirmation_delay_s": float(np.mean(delays)) if delays else None,
            "prompted_truth_events": cue_hits, "unprompted_truth_events": n_truth - cue_hits,
            "unmatched_guidance_starts": cue_false,
            "mean_truth_window_to_guidance_delay_s": (
                float(np.mean(cue_delays)) if cue_delays else None),
            "start_commands": len(starts),
            "starts_on_non_fog_windows": int(starts.truth_fog_label.astype(int).eq(0).sum()),
            "active_non_fog_windows": int((windows.truth_fog_label.astype(int).eq(0)
                                           & windows.guidance_visual_enabled).sum()),
            "terminal_stops": int(stopped.guidance_terminal.sum()),
            "fade_completed_stops": int(stopped.guidance_reason.eq(
                "simulated_recovery_fade_completed").sum()),
            "upstream_recovery_stops": int(stopped.guidance_reason.eq(
                "detection_recovery_completed").sum()),
            "timeout_stops": int(stopped.guidance_reason.eq("episode_timeout").sum()),
            "stop_reasons": dict(Counter(stopped.guidance_reason)), "events": details,
            "latency_definition": "first matching output minus first truth-positive window output",
            "matching": "one-to-one onset inside truth run through end+2s; no pre-onset match",
            "active_non_fog_note": "includes intended recovery tails; not all are false alarms"}


def audit_commands(commands: pd.DataFrame) -> list[dict]:
    """Independent lifecycle/modality checks; return failures without hiding them."""
    issues = []
    active = {}
    seen = set()

    def issue(index, code):
        issues.append({"row": int(index), "code": code})

    for index, row in enumerate(commands.to_dict("records")):
        key = tuple(str(row[name]) for name in SEQUENCE) + (str(row["guidance_command_id"]),)
        action = row["guidance_action"]
        visual = bool(row["guidance_visual_enabled"])
        audio = bool(row["guidance_audio_enabled"])
        targets = json.loads(row["guidance_targets"])
        if visual != audio:
            issue(index, "modalities_not_synchronized")
        if action in {"none", "stop"} and (
            visual or audio or targets or row["guidance_visual_opacity"] != 0
            or row["guidance_audio_volume"] != 0
        ):
            issue(index, "disabled_output_still_active")
        if action == "start":
            if key in seen or any(k[:-1] == key[:-1] for k in active):
                issue(index, "duplicate_or_overlapping_start")
            active[key] = row
            seen.add(key)
        elif action in {"hold", "update"}:
            if key not in active:
                issue(index, "update_without_start")
            else:
                old = active[key]
                if row["guidance_cue_started_at_s"] != old["guidance_cue_started_at_s"]:
                    issue(index, "episode_clock_reset")
                if action == "hold" and targets != json.loads(old["guidance_targets"]):
                    issue(index, "hold_moved_targets")
                active[key] = row
        elif action == "stop":
            if key not in active:
                issue(index, "stop_without_start")
            active.pop(key, None)
        elif action != "none":
            issue(index, "unknown_action")
        if visual:
            if not targets or row["guidance_audio_mode"] == "off":
                issue(index, "active_output_incomplete")
            for target in targets:
                if not np.isfinite([target["x_m"], target["y_m"], target["yaw_deg"]]).all():
                    issue(index, "nonfinite_target")
    for key in active:
        issues.append({"row": None, "code": "unclosed_command", "key": list(key)})
    return issues
