"""Causal no-response timer, safety withdrawal, fallbacks and truthful provenance."""

import json

import numpy as np
import pandas as pd
import pytest

from fogguard.cueing.proxy_replay import replay_proxy_guidance
from fogguard.evaluation.automation2 import audit_commands
from fogguard.motion.proxy_feedback import GYRO, estimate_motion_feedback
from fogguard.perception import fuse_perception


def fixture(*, motion=0.0, recover=False):
    ends = np.arange(2, 16, 0.5)
    scores = np.where((ends >= 9) & recover, 0.1, 0.9)
    common = {"subject_id": "1", "session_id": "1", "task_id": "1",
              "window_start_s": ends - 119/60, "window_end_s": ends,
              "available_at_s": ends, "model_version": "fixture",
              "prediction_source": "constructed_test"}
    fog = pd.DataFrame(common).assign(fog_probability=scores, fog_threshold=0.5, is_fog=scores >= 0.5)
    direction = pd.DataFrame(common).assign(direction="straight", direction_confidence=0.9,
                                             direction_task="current", horizon_s=0.0)
    times = np.arange(16 * 60) / 60
    sensors = pd.DataFrame({"timestamp": times, "subjectID": 1, "sessionID": 1, "taskID": 1})
    for col in GYRO:
        sensors[col] = motion
    feedback = estimate_motion_feedback(sensors)
    ticks = np.unique(np.round(np.r_[np.arange(0, 16, 0.1), feedback.feedback_available_at_s], 9))
    clock = pd.DataFrame({"subject_id": "1", "session_id": "1", "task_id": "1", "timestamp_s": ticks})
    perception = fuse_perception(fog, direction, clock=clock)
    return perception, feedback


def test_no_motion_timer_only_counts_new_complete_postcue_windows():
    perception, feedback = fixture()
    result = replay_proxy_guidance(perception, feedback, strategy="march")
    start = result.loc[result.guidance_action.eq("start")].iloc[0].guidance_timestamp_s
    entry = result.loc[result.guidance_reason.eq("valid_no_motion_enter_march")].iloc[0]
    assert entry.guidance_timestamp_s > start + 3
    assert entry.proxy_no_motion_valid_s >= 3
    assert entry.proxy_no_motion_valid_s < 3.5
    # Repeated timer/held rows never accumulate new no-motion evidence.
    prior = result.proxy_no_motion_valid_s.shift(fill_value=0)
    increase = result.proxy_no_motion_valid_s > prior + 1e-9
    assert result.loc[increase, "proxy_observation_is_new"].all()
    assert (result.loc[increase, "proxy_no_motion_valid_s"] - prior[increase] <= 29/60 + 1e-9).all()
    assert not audit_commands(result)


def test_same_inputs_countdown_and_march_switch_at_same_evidence():
    perception, feedback = fixture()
    countdown = replay_proxy_guidance(perception, feedback, strategy="countdown")
    marching = replay_proxy_guidance(perception, feedback, strategy="march")
    fixed = replay_proxy_guidance(perception, feedback, strategy="fixed")
    c = countdown.loc[countdown.guidance_reason.eq("valid_no_motion_enter_countdown")].iloc[0]
    m = marching.loc[marching.guidance_reason.eq("valid_no_motion_enter_march")].iloc[0]
    assert c.guidance_timestamp_s == m.guidance_timestamp_s
    assert c.proxy_no_motion_valid_s == m.proxy_no_motion_valid_s
    assert not fixed.guidance_control_state.isin(["MARCH_FALLBACK", "RESET_CUE"]).any()
    feet = [json.loads(value)[0]["foot"] for value in marching.loc[
        marching.guidance_control_state.eq("MARCH_FALLBACK"), "guidance_targets"]]
    assert set(feet) == {"left", "right"}
    assert marching.guidance_reason.eq("fallback_timeout_return_forward_unverified").any()
    assert marching.guidance_stable_steps.eq(0).all()
    assert not audit_commands(countdown) and not audit_commands(marching) and not audit_commands(fixed)


def test_unknown_feedback_withdraws_cue_and_resets_timer():
    perception, feedback = fixture()
    feedback.loc[feedback.feedback_available_at_s.between(4.4, 4.6),
                 ["feedback_valid", "feedback_state", "observed_duration_s"]] = [False, "unknown", 0]
    result = replay_proxy_guidance(perception, feedback, strategy="march")
    at_failure = result.loc[result.proxy_observation_is_new & result.proxy_feedback_state.eq("unknown")]
    assert len(at_failure) == 1
    assert not at_failure.guidance_visual_enabled.any()
    assert at_failure.proxy_no_motion_valid_s.eq(0).all()
    assert not audit_commands(result)


def test_recovery_motion_proxy_fades_without_claiming_steps():
    perception, feedback = fixture(motion=20.0, recover=True)
    result = replay_proxy_guidance(perception, feedback, strategy="march")
    fading = result.loc[result.guidance_control_state.eq("FADING_PROXY")]
    assert len(fading) >= 2
    assert fading.guidance_visual_opacity.min() < fading.guidance_visual_opacity.max()
    assert result.guidance_stable_steps.eq(0).all()
    assert not audit_commands(result)


def test_reject_unmatched_records_bad_source_and_missing_feedback_ticks():
    perception, feedback = fixture()
    with pytest.raises(ValueError, match="matched record"):
        replay_proxy_guidance(perception, feedback.assign(subject_id="someone-else"))
    with pytest.raises(ValueError, match="movement-only"):
        replay_proxy_guidance(perception, feedback.assign(feedback_source="simulated"))
    time = feedback.feedback_available_at_s.iloc[0]
    without_tick = perception.loc[~np.isclose(perception.available_at_s, time, atol=1e-8, rtol=0)]
    with pytest.raises(ValueError, match="[Dd]ecision clock"):
        replay_proxy_guidance(without_tick, feedback)


def test_reject_overlapping_or_inflated_feedback_duration():
    perception, feedback = fixture()
    overlap = feedback.copy()
    overlap.loc[1, "feedback_window_start_s"] = overlap.loc[0, "feedback_window_end_s"] - .1
    with pytest.raises(ValueError, match="duration|non-overlapping"):
        replay_proxy_guidance(perception, overlap)
    inflated = feedback.copy()
    inflated.loc[1, "observed_duration_s"] = 20
    with pytest.raises(ValueError, match="duration"):
        replay_proxy_guidance(perception, inflated)


def test_rearm_after_unknown_requires_two_new_fog_observations():
    perception, feedback = fixture()
    feedback.loc[feedback.feedback_available_at_s.between(4.4, 4.6),
                 ["feedback_valid", "feedback_state", "observed_duration_s"]] = [False, "unknown", 0]
    result = replay_proxy_guidance(perception, feedback, strategy="march")
    waiting = result.loc[result.guidance_reason.eq("waiting_two_new_fog_observations")]
    assert not waiting.empty and not waiting.guidance_visual_enabled.any()
    # Feedback becomes valid just before 5 s; there is only one new FoG input
    # between 5.0 and 5.5, no matter how many 0.1-s timer rows occur.
    between = result.loc[result.available_at_s.ge(5.0) & result.available_at_s.lt(5.5)]
    assert not between.guidance_visual_enabled.any()
    later = result.loc[result.available_at_s.ge(5.5) & result.guidance_action.eq("start")]
    assert not later.empty


def test_confirmed_new_turn_changes_footprint_geometry():
    perception, feedback = fixture()
    perception.loc[perception.available_at_s.ge(4), "turn"] = "left"
    result = replay_proxy_guidance(perception, feedback, strategy="fixed")
    left = result.loc[result.available_at_s.ge(4) & result.guidance_visual_enabled].iloc[0]
    targets = json.loads(left.guidance_targets)
    assert targets[0]["yaw_deg"] > 0
    assert left.guidance_action == "update"


def test_refreeze_during_proxy_fade_restores_intensity():
    perception, feedback = fixture(motion=20.0)
    recovery = perception.available_at_s.ge(9) & perception.available_at_s.lt(9.8)
    perception.loc[recovery, "event_phase"] = "recovery"
    perception.loc[recovery, "stable_fog_detected"] = False
    result = replay_proxy_guidance(perception, feedback, strategy="march")
    assert result.guidance_control_state.eq("FADING_PROXY").any()
    restore = result.loc[result.guidance_reason.eq("perception_reconfirmed_restore_intensity")]
    assert not restore.empty
    assert restore.guidance_visual_opacity.iloc[0] == pytest.approx(1.0)


def test_disable_proxy_fading_keeps_full_intensity_until_base_exit():
    from dataclasses import replace

    from fogguard.cueing.contracts import GuidanceConfig

    perception, feedback = fixture(motion=20.0, recover=True)
    enabled = replay_proxy_guidance(perception, feedback, strategy="march")
    disabled = replay_proxy_guidance(perception, feedback, strategy="march",
                                     guidance_config=replace(GuidanceConfig(), enable_fading=False))
    assert enabled.guidance_control_state.eq("FADING_PROXY").any()
    assert not disabled.guidance_control_state.eq("FADING_PROXY").any()
    assert disabled.loc[disabled.guidance_visual_enabled, "guidance_visual_opacity"].eq(1).all()
    assert not audit_commands(disabled)


def test_feedback_baseline_uses_same_gate_without_fallback():
    perception, feedback = fixture()
    marching = replay_proxy_guidance(perception, feedback, strategy="march")
    feedback_only = replay_proxy_guidance(perception, feedback, strategy="feedback")
    assert marching.guidance_control_state.eq("MARCH_FALLBACK").any()
    assert not feedback_only.guidance_control_state.isin(["MARCH_FALLBACK", "RESET_CUE"]).any()
    assert marching.guidance_action.eq("start").sum() == feedback_only.guidance_action.eq("start").sum()
    feedback.loc[feedback.feedback_available_at_s.between(4.4, 4.6),
                 ["feedback_valid", "feedback_state", "observed_duration_s"]] = [False, "unknown", 0]
    result = replay_proxy_guidance(perception, feedback, strategy="feedback")
    assert result.guidance_reason.eq("feedback_unavailable").any()
    assert not result.loc[~result.proxy_feedback_valid_now, "guidance_visual_enabled"].any()
