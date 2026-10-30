"""Optional causal cue replay with honest movement-only feedback provenance.

The original GuidanceController remains the default. This module does not
construct StepResponse or infer actual footfalls or forward distance.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from fogguard.cueing.contracts import FootTarget, GuidanceConfig, GuidanceInput
from fogguard.cueing.guidance import GuidanceController
from fogguard.cueing.targets import make_targets
from fogguard.motion.proxy_feedback import KEYS, MotionProxyConfig


class _ProxyBaseController(GuidanceController):
    def cancel_on_proxy_fade(self, now):
        return self._stop(now, "proxy_motion_fade_completed")


def _flag(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false", "0", "1"}:
        return value.lower() in {"true", "1"}
    raise ValueError("Expected an explicit boolean")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _validate_inputs(perception, feedback, config):
    required = {*KEYS, "available_at_s", "window_end_s", "event_id", "event_phase",
                "stable_fog_detected", "turn", "turn_confidence", "input_fresh",
                "input_age_s", "fog_observation_is_new"}
    needed_feedback = {*KEYS, "feedback_window_start_s", "feedback_window_end_s",
                       "feedback_available_at_s", "observed_duration_s", "feedback_valid",
                       "feedback_state", "feedback_source", "feedback_kind"}
    if perception.empty or not required <= set(perception) or not needed_feedback <= set(feedback):
        raise ValueError("Perception or feedback schema is incomplete")
    p, f = perception.copy(), feedback.copy()
    for frame in (p, f):
        for key in KEYS:
            if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
                raise ValueError("Missing record identity")
            frame[key] = frame[key].astype(str)
    keys = p[list(KEYS)].drop_duplicates()
    if len(keys) != 1 or (not f.empty and (len(f[list(KEYS)].drop_duplicates()) != 1
            or not all(f[key].eq(keys.iloc[0][key]).all() for key in KEYS))):
        raise ValueError("Each replay requires exactly one matched record")
    for name, frame in (("available_at_s", p), ("feedback_available_at_s", f)):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        values = frame[name].to_numpy(float)
        if not np.isfinite(values).all() or (np.diff(values) <= 0).any():
            raise ValueError("Input times must be finite, unique and increasing")
    p["window_end_s"] = pd.to_numeric(p.window_end_s, errors="raise")
    if not np.allclose(p.window_end_s, p.available_at_s, atol=1e-8, rtol=0):
        raise ValueError("Perception window_end_s must be the decision-clock alias")
    if not f.empty:
        if not f.feedback_source.eq("derived_proxy").all() or not f.feedback_kind.eq("movement_only").all():
            raise ValueError("This adapter accepts only explicitly declared movement-only proxy feedback")
        if not f.feedback_state.isin(["movement_evidence", "no_movement_evidence", "unknown"]).all():
            raise ValueError("Invalid motion proxy state")
        f["feedback_valid"] = f.feedback_valid.map(_flag)
        if (f.feedback_valid != f.feedback_state.ne("unknown")).any():
            raise ValueError("Proxy validity and state disagree")
        for col in ("feedback_window_start_s", "feedback_window_end_s", "observed_duration_s"):
            f[col] = pd.to_numeric(f[col], errors="raise")
        if (not np.isfinite(f[["feedback_window_start_s", "feedback_window_end_s",
                               "observed_duration_s"]].to_numpy(float)).all()
                or (f.feedback_window_end_s < f.feedback_window_start_s).any()
                or (f.feedback_available_at_s < f.feedback_window_end_s - 1e-8).any()
                or (f.observed_duration_s < 0).any()
                or (f.observed_duration_s > f.feedback_window_end_s - f.feedback_window_start_s + 1e-8).any()
                or (f.loc[~f.feedback_valid, "observed_duration_s"] != 0).any()):
            raise ValueError("Invalid proxy observation duration or time")
        spans = f.feedback_window_end_s - f.feedback_window_start_s
        full_span = (config.window_samples - 1) / config.sample_rate_hz
        if ((np.abs(spans[f.feedback_valid] - full_span) > config.time_tolerance_s).any()
                or (np.abs(f.loc[f.feedback_valid, "observed_duration_s"]
                           - spans[f.feedback_valid]) > config.time_tolerance_s).any()
                or (f.feedback_window_start_s.to_numpy()[1:]
                    < f.feedback_window_end_s.to_numpy()[:-1] - 1e-8).any()):
            raise ValueError("Feedback windows must be full, non-overlapping observations")
        # Every feedback arrival must be a decision tick, or valid evidence might be skipped.
        if not np.isin(f.feedback_available_at_s.round(6), p.available_at_s.round(6)).all():
            raise ValueError("Decision clock must include each feedback availability")
    return p, f


def replay_proxy_guidance(perception, feedback, *, strategy="march", proxy_config=None,
                          guidance_config=None, scenario_id="proxy"):
    """Process one record and emit commands and auditable per-tick proxy evidence.

    fixed ignores feedback. feedback, countdown and march require valid current feedback;
    unknown feedback withdraws cues and resets the consecutive inactivity timer.
    Alternating march targets are prompts, never claimed observed footfalls.
    """
    if strategy not in {"fixed", "feedback", "countdown", "march"}:
        raise ValueError("Unknown proxy strategy")
    cfg = proxy_config or MotionProxyConfig()
    p, f = _validate_inputs(perception, feedback, cfg)
    gc = guidance_config or GuidanceConfig()
    base = _ProxyBaseController(replace(gc, enable_response_policy=False), responses_configured=False)
    records = f.to_dict("records")
    pointer = 0
    latest = None
    no_motion_s = 0.0
    last_no_motion_end = None
    motion_streak = 0
    mode, mode_started, fade_started = "GUIDING", None, None
    fallback_used = False
    current_id = None
    previous_exposed = None
    rearm_required, new_confirmations = False, 0
    output = []

    def append(row, command, *, terminal=False, new=False, ready=False):
        command = replace(command, schema_version="guidance-proxy-experiment-v1",
                          response_source=("not_used" if strategy == "fixed" else
                                           "derived_proxy" if latest is not None else "unavailable"))
        data = dict(row)
        data["scenario_id"] = scenario_id
        data["proxy_strategy"] = strategy
        data["proxy_fading_enabled"] = gc.enable_fading
        data.update({"proxy_feedback_state": latest["feedback_state"] if latest else "unavailable",
                     "proxy_feedback_valid_now": ready, "proxy_observation_is_new": new,
                     "proxy_no_motion_valid_s": no_motion_s,
                     "proxy_movement_streak": motion_streak,
                     "proxy_fallback_used": fallback_used,
                     "proxy_feedback_source": "derived_proxy" if latest else "unavailable"})
        for key, value in command.as_dict().items():
            data["guidance_" + key] = json.dumps(value, ensure_ascii=False, allow_nan=False) if key == "targets" else value
        data["guidance_terminal"] = terminal
        output.append(data)

    for row in p.to_dict("records"):
        now = _number(row["available_at_s"])
        new = False
        while pointer < len(records) and records[pointer]["feedback_available_at_s"] <= now + 1e-9:
            latest = records[pointer]
            pointer += 1
            new = True
        feedback_age = round(now - latest["feedback_window_end_s"], 9) if latest else float("nan")
        ready = bool(latest is not None and latest["feedback_valid"] and
                     -1e-8 <= feedback_age < cfg.source_lifetime_s - 1e-8)
        phase, turn = str(row["event_phase"]), str(row["turn"])
        confidence, age = _number(row["turn_confidence"]), _number(row["input_age_s"])
        p_fresh = (_flag(row["input_fresh"]) and math.isfinite(age) and 0 <= age <= gc.maximum_input_age_s
                   and turn in {"straight", "left", "right"} and math.isfinite(confidence)
                   and gc.minimum_direction_confidence <= confidence <= 1)
        gate = p_fresh if strategy == "fixed" else p_fresh and ready
        gate_reason = ("perception_unavailable" if not p_fresh else
                       "feedback_unavailable" if strategy != "fixed" and not ready else "")
        if strategy != "fixed":
            if not gate:
                rearm_required, new_confirmations = True, 0
            elif rearm_required:
                if phase != "confirmed":
                    rearm_required, new_confirmations = False, 0
                else:
                    if _flag(row["fog_observation_is_new"]):
                        new_confirmations += 1
                    if new_confirmations >= 2:
                        rearm_required, new_confirmations = False, 0
                    else:
                        gate = False
                        gate_reason = "waiting_two_new_fog_observations"
        event_id = row["event_id"]
        if pd.isna(event_id):
            event_id = None
        elif isinstance(event_id, (float, np.floating)) and event_id.is_integer():
            event_id = str(int(event_id))
        else:
            event_id = str(event_id)
        item = GuidanceInput(now_s=now, sequence=tuple(row[k] for k in KEYS) + (scenario_id,),
            event_id=event_id, event_phase=phase,
            stable_fog_detected=_flag(row["stable_fog_detected"]), turn=turn,
            turn_confidence=confidence, input_fresh=gate, input_age_s=age,
            quality_reason=gate_reason if not gate else "")
        command = base.step(item)
        if not command.visual_enabled:
            no_motion_s, last_no_motion_end, motion_streak = 0.0, None, 0
            mode, mode_started, fade_started = "GUIDING", None, None
            current_id, previous_exposed, fallback_used = None, None, False
            append(row, command, new=new, ready=ready)
            continue
        if command.command_id != current_id:
            current_id = command.command_id
            no_motion_s, last_no_motion_end, motion_streak = 0.0, None, 0
            mode, mode_started, fade_started, fallback_used = "GUIDING", None, None, False
            previous_exposed = None
        reason = command.reason
        if strategy != "fixed" and new and ready and latest["feedback_window_start_s"] >= command.cue_started_at_s - 1e-8:
            if latest["feedback_state"] == "no_movement_evidence":
                if (last_no_motion_end is not None and
                        latest["feedback_window_start_s"] - last_no_motion_end > 1.5 / cfg.sample_rate_hz + 1e-8):
                    no_motion_s = 0.0
                no_motion_s += latest["observed_duration_s"]
                last_no_motion_end = latest["feedback_window_end_s"]
                motion_streak = 0
            elif latest["feedback_state"] == "movement_evidence":
                no_motion_s, last_no_motion_end = 0.0, None
                motion_streak += 1
        if mode in {"MARCH_FALLBACK", "RESET_CUE"} and phase != "confirmed":
            mode, mode_started = "GUIDING", None
            reason = "perception_recovery_ends_fallback"
        if (strategy in {"countdown", "march"} and phase == "confirmed" and mode == "GUIDING"
                and not fallback_used and no_motion_s >= cfg.valid_no_movement_to_fallback_s):
            mode = "MARCH_FALLBACK" if strategy == "march" else "RESET_CUE"
            mode_started, fallback_used = now, True
            reason = "valid_no_motion_enter_march" if strategy == "march" else "valid_no_motion_enter_countdown"
        if mode in {"MARCH_FALLBACK", "RESET_CUE"} and now - mode_started >= cfg.fallback_duration_s:
            mode, mode_started = "GUIDING", None
            reason = "fallback_timeout_return_forward_unverified"
        if phase == "confirmed" and fade_started is not None:
            fade_started = None
            reason = "perception_reconfirmed_restore_intensity"
        if (strategy != "fixed" and gc.enable_fading and phase == "recovery"
                and mode == "GUIDING" and motion_streak >= 2):
            if fade_started is None:
                fade_started = now
                reason = "perception_recovery_with_motion_proxy_begin_fade"
            if now - fade_started >= gc.fade_seconds:
                command = base.cancel_on_proxy_fade(now)
                no_motion_s, last_no_motion_end, motion_streak = 0.0, None, 0
                mode, mode_started, fade_started = "GUIDING", None, None
                current_id, previous_exposed, fallback_used = None, None, False
                append(row, command, new=new, ready=ready)
                continue
        # Direction is applied to configured targets, not to a measured pose.
        targets = make_targets(gc, command_id=command.command_id, first_index=0,
                               first_foot=gc.first_foot, direction=turn,
                               step_length_m=gc.step_length_m)
        audio_mode = command.audio_mode
        state = "FADING_PROXY" if fade_started is not None else mode
        if mode == "MARCH_FALLBACK":
            parity = int((now - mode_started + 1e-9) // (60 / gc.cadence_bpm)) % 2
            feet = ("left", "right") if parity == 0 else ("right", "left")
            targets = tuple(FootTarget(f"{command.command_id}/march-{parity}-{foot}", foot,
                                       0.0, (1 if foot == "left" else -1)*gc.step_width_m/2, 0.0)
                            for foot in feet)
            audio_mode = "march_alternating"
        elif mode == "RESET_CUE":
            audio_mode = "countdown_3_2_1_go"
        factor = 1.0 if fade_started is None else max(0.0, 1 - (now-fade_started)/gc.fade_seconds)
        exposed = replace(command, schema_version="guidance-proxy-experiment-v1", control_state=state,
                          reason=reason, targets=targets, audio_mode=audio_mode,
                          visual_opacity=gc.opacity*factor, audio_volume=gc.volume*factor,
                          response_source="derived_proxy" if strategy != "fixed" else "not_used",
                          mode_started_at_s=mode_started if mode_started is not None else command.mode_started_at_s)
        if exposed.action != "start":
            fields = ("control_state", "targets", "visual_opacity", "audio_volume", "audio_mode")
            changed = previous_exposed is None or any(getattr(exposed, name) != getattr(previous_exposed, name)
                                                       for name in fields)
            exposed = replace(exposed, action="update" if changed else "hold")
        previous_exposed = exposed
        append(row, exposed, new=new, ready=ready)
    terminal = base.finish()
    if terminal is not None:
        append(p.iloc[-1].to_dict(), terminal, terminal=True)
    return pd.DataFrame(output)
