import math
from dataclasses import replace

import pytest

from fogguard.cueing.contracts import GuidanceConfig, GuidanceInput, StepResponse
from fogguard.cueing.guidance import GuidanceController
from fogguard.cueing.targets import make_targets


def context(time=2.0, phase="confirmed", **overrides):
    return replace(GuidanceInput(
        now_s=time, sequence=("subject", "session", "task", "scenario"), event_id="1",
        event_phase=phase, stable_fog_detected=phase == "confirmed",
        turn="straight", turn_confidence=0.9, input_fresh=True,
    ), **overrides)


def response(time, foot="left", status="success", reliability=1.0, forward=0.4):
    return StepResponse(time, foot, forward, 0.1, status, reliability)


def test_guidance_starts_once_and_keeps_targets_and_clock_without_landings():
    controller = GuidanceController()
    start = controller.step(context())
    hold = controller.step(context(2.5, turn="left"))
    assert start.action == "start"
    assert start.visual_enabled and start.audio_enabled
    assert hold.action == "hold"
    assert start.targets == hold.targets
    assert hold.cue_started_at_s == start.cue_started_at_s == 2.0
    assert start.command_id == hold.command_id


def test_short_dip_does_not_restart_an_event_and_recovery_stops_both_modalities():
    controller = GuidanceController()
    start = controller.step(context())
    observe = controller.step(context(2.5, "recovery"))
    resumed = controller.step(context(3.0))
    stopped = controller.step(context(3.5, "normal", event_id=None))
    assert observe.control_state == "EXIT_OBSERVE"
    assert resumed.command_id == start.command_id
    assert resumed.action == "update"
    assert stopped.action == "stop"
    assert not stopped.audio_enabled and not stopped.visual_enabled
    assert stopped.targets == ()


@pytest.mark.parametrize("changes,reason", [
    ({"input_fresh": False}, "stale_input"),
    ({"input_age_s": 2.0}, "stale_input"),
    ({"input_age_s": -1}, "stale_input"),
    ({"turn": "unknown"}, "direction_unavailable"),
    ({"turn_confidence": 0.1}, "direction_unavailable"),
    ({"turn_confidence": float("nan")}, "direction_unavailable"),
    ({"stable_fog_detected": False}, "inconsistent_event_contract"),
    ({"event_id": None}, "inconsistent_event_contract"),
    ({"event_phase": "safe_none"}, "event_unavailable"),
])
def test_fault_stops_immediately_even_inside_refractory(changes, reason):
    controller = GuidanceController()
    controller.step(context())
    stopped = controller.step(context(2.5, **changes))
    assert stopped.action == "stop"
    assert stopped.control_state == "SAFE_NONE"
    assert stopped.reason == reason
    assert not stopped.audio_enabled and not stopped.visual_enabled


def test_rearming_after_unknown_requires_two_fresh_confirmed_inputs():
    controller = GuidanceController(GuidanceConfig(refractory_seconds=0))
    controller.step(context())
    controller.step(context(2.5, turn="unknown"))
    waiting = controller.step(context(3.0))
    restart = controller.step(context(3.5))
    assert waiting.action == "none"
    assert waiting.reason == "waiting_fresh_reconfirmation"
    assert restart.action == "start"


def test_refractory_controls_only_new_starts_and_boundary_is_inclusive():
    controller = GuidanceController()
    controller.step(context())
    assert controller.step(context(2.5, "normal", event_id=None)).action == "stop"
    for t in (3, 3.5, 4, 4.5):
        assert controller.step(context(t, event_id="2")).reason == "start_refractory"
    assert controller.step(context(5, event_id="2")).action == "start"


@pytest.mark.parametrize("time,reason", [
    (3.0, "input_gap"), (1.0, "time_not_strictly_increasing"),
    (2.0, "time_not_strictly_increasing"), (float("nan"), "invalid_timestamp"),
])
def test_bad_times_revoke_active_command_without_advancing_invalid_clock(time, reason):
    controller = GuidanceController()
    controller.step(context())
    stopped = controller.step(context(time))
    assert stopped.action == "stop"
    assert stopped.reason == reason
    assert math.isfinite(stopped.timestamp_s) and stopped.timestamp_s >= 2.0


def test_sequence_switch_cancels_old_command_before_new_start():
    controller = GuidanceController()
    first = controller.step(context())
    sequence = ("another", "session", "task", "scenario")
    stopped = controller.step(context(0.5, sequence=sequence))
    new = controller.step(context(1.0, sequence=sequence))
    assert stopped.action == "stop" and stopped.command_id == first.command_id
    assert new.action == "start" and new.command_id != first.command_id


def test_single_reset_then_timeout_latches_until_a_new_event():
    config = GuidanceConfig(maximum_episode_s=4.0, response_timeout_s=1.0, reset_duration_s=1)
    controller = GuidanceController(config, responses_configured=True)
    controller.step(context())
    outputs = [controller.step(context(2 + i * 0.5)) for i in range(1, 11)]
    assert sum(item.control_state == "RESET_CUE" and item.action == "update"
               for item in outputs) == 1
    assert any(item.reason == "episode_timeout" and item.action == "stop" for item in outputs)
    assert outputs[-1].reason == "event_finished_waiting_new_event"
    assert controller.step(context(7.5, event_id="2")).action == "start"


def test_missing_response_disables_fallback_and_does_not_invent_steps():
    controller = GuidanceController(GuidanceConfig(response_timeout_s=1), responses_configured=True)
    controller.step(context())
    unavailable = controller.step(context(2.5), (response(2.5, status="missing_input"),))
    late = controller.step(context(3.0))
    assert unavailable.response_source == "unavailable_fixed_fallback"
    assert late.control_state == "GUIDING" and late.stable_steps == 0
    assert late.targets == unavailable.targets


def test_source_health_before_start_survives_stops_and_later_events():
    controller = GuidanceController(GuidanceConfig(response_timeout_s=1, refractory_seconds=0),
                                    responses_configured=True)
    controller.step(context(1.0, "normal", event_id=None), (
        response(1.0, status="missing_input"),
    ))
    for t in (1.5, 2.0, 2.5, 3.0):
        command = controller.step(context(t))
        assert command.control_state == "GUIDING"
        assert command.response_source == "unavailable_fixed_fallback"
    controller.step(context(3.5, "normal", event_id=None))
    for t in (4.0, 4.5, 5.0, 5.5):
        command = controller.step(context(t, event_id="2"))
        assert command.control_state == "GUIDING"
        assert command.response_source == "unavailable_fixed_fallback"


def test_no_response_source_never_assumes_failure_or_actual_recovery():
    controller = GuidanceController(GuidanceConfig(response_timeout_s=1))
    controller.step(context())
    for i in range(1, 10):
        assert controller.step(context(2 + 0.5 * i)).control_state == "GUIDING"
    stop = controller.step(context(7, "normal", event_id=None))
    assert stop.reason == "detection_recovery_completed"


def test_response_moves_only_next_target_and_noisy_step_cannot_trigger_fade():
    controller = GuidanceController(responses_configured=True)
    start = controller.step(context())
    low = controller.step(context(2.5), (response(2.5, reliability=0.1),))
    assert start.targets == low.targets and low.stable_steps == 0
    valid = controller.step(context(3.0), (response(3.0),))
    assert valid.targets[0].foot == "right"
    assert start.targets[0].target_id != valid.targets[0].target_id
    held = controller.step(context(3.5, turn="right"))
    assert held.targets == valid.targets


def test_duplicate_and_future_response_rejected():
    controller = GuidanceController(responses_configured=True)
    controller.step(context())
    with pytest.raises(ValueError, match="Future"):
        controller.step(context(2.5), (response(3.0),))
    # A rejected future record does not advance the replay clock.
    controller.step(context(2.5), (response(2.5),))
    with pytest.raises(ValueError, match="once"):
        controller.step(context(3.0), (response(2.5),))


def test_unconfigured_response_cannot_silently_enable_recovery_policy():
    controller = GuidanceController()
    with pytest.raises(ValueError, match="explicitly configured"):
        controller.step(context(), (response(2.0),))


@pytest.mark.parametrize("status,foot,forward", [
    ("unable", "none", None), ("no_response", "none", None),
    ("success", "left", 0.4), ("success", "right", 0.01),
])
def test_interrupted_recovery_revokes_fade_and_requires_new_stable_steps(status, foot, forward):
    controller = GuidanceController(responses_configured=True)
    controller.step(context())
    controller.step(context(2.5), (response(2.5),))
    controller.step(context(3.0), (response(3.0, "right"),))
    assert controller.step(context(3.5, "recovery"), (response(3.5),)).control_state == "FADING"
    interrupted = controller.step(context(4.0, "recovery"), (
        response(4.0, foot, status, forward=forward),
    ))
    assert interrupted.stable_steps == 0
    assert interrupted.control_state == "EXIT_OBSERVE"
    assert interrupted.audio_volume == GuidanceConfig().volume
    # The old fade deadline cannot finish a recovery after contrary evidence.
    later = controller.step(context(4.5, "recovery"))
    assert later.control_state == "EXIT_OBSERVE"
    assert later.reason != "simulated_recovery_fade_completed"
    restarted_evidence = controller.step(context(5.0, "recovery"), (
        response(5.0, "right"),
    ))
    assert restarted_evidence.stable_steps == 1
    assert restarted_evidence.control_state == "RECOVERING"


def test_rejected_response_batch_is_atomic_and_does_not_move_targets_or_health():
    controller = GuidanceController(responses_configured=True)
    original = controller.step(context())
    with pytest.raises(ValueError, match="Future"):
        controller.step(context(2.5), (
            response(2.25, status="missing_input"), response(3.0),
        ))
    unchanged = controller.step(context(2.5))
    assert unchanged.targets == original.targets
    assert unchanged.response_source == "simulated"
    with pytest.raises(ValueError, match="once"):
        controller.step(context(3.0), (response(3.0), response(3.0, "right")))
    accepted = controller.step(context(3.0), (response(3.0),))
    assert accepted.stable_steps == 1
    assert accepted.targets[0].foot == "right"


def test_response_cannot_be_reused_across_episode_boundaries():
    controller = GuidanceController(GuidanceConfig(refractory_seconds=0), responses_configured=True)
    controller.step(context(), (response(2.0),))
    controller.step(context(2.5, "normal", event_id=None))
    # It may remain recovering because of the recent step; a core fault ends it.
    controller.step(context(3.0, turn="unknown"))
    controller.step(context(3.5))
    with pytest.raises(ValueError, match="once"):
        controller.step(context(4.0), (response(2.0),))
    assert controller.step(context(4.0)).action == "start"


def test_fading_requires_stable_alternating_steps_and_reconfirmation_restores_intensity():
    controller = GuidanceController(responses_configured=True)
    controller.step(context())
    controller.step(context(2.5), (response(2.5),))
    recovery = controller.step(context(3.0, "recovery"), (response(3.0, "right"),))
    fade = controller.step(context(3.5, "recovery"), (response(3.5),))
    reduced = controller.step(context(4.0, "recovery"), (response(4.0, "right"),))
    restored = controller.step(context(4.5))
    assert recovery.control_state == "RECOVERING"
    assert fade.control_state == reduced.control_state == "FADING"
    assert reduced.audio_volume < fade.audio_volume
    assert restored.audio_volume == GuidanceConfig().volume
    assert restored.visual_opacity == GuidanceConfig().opacity
    assert restored.reason == "fog_reconfirmed_restore_intensity"
    assert restored.action == "update"


def test_fade_finishes_then_new_confirmation_of_same_event_can_restart():
    config = GuidanceConfig(fade_seconds=0.5, refractory_seconds=0)
    controller = GuidanceController(config, responses_configured=True)
    controller.step(context())
    controller.step(context(2.5), (response(2.5),))
    controller.step(context(3.0), (response(3.0, "right"),))
    controller.step(context(3.5, "recovery"), (response(3.5),))
    stop = controller.step(context(4.0, "recovery"), (response(4.0, "right"),))
    assert stop.reason == "simulated_recovery_fade_completed"
    assert controller.step(context(4.5)).action == "start"


def test_adaptive_spacing_waits_for_stable_steps_then_stays_bounded():
    config = GuidanceConfig(enable_adaptive_spacing=True)
    controller = GuidanceController(config, responses_configured=True)
    controller.step(context())
    lengths = []
    for i in range(1, 9):
        foot = "left" if i % 2 else "right"
        command = controller.step(context(2 + i * 0.5), (
            response(2 + i * 0.5, foot, forward=0.8),
        ))
        lengths.append(command.step_length_m)
    assert lengths[:2] == [0.4, 0.4]
    assert lengths[2] == pytest.approx(0.43)
    assert max(lengths) <= config.maximum_step_length_m
    assert max(b - a for a, b in zip([0.4] + lengths[:-1], lengths, strict=True)) <= (
        config.maximum_step_adjustment_m + 1e-9
    )


def test_turn_targets_are_mirrored_with_opposite_configured_start_foot():
    left = make_targets(GuidanceConfig(), command_id="a", first_index=0,
                        first_foot="left", direction="left", step_length_m=0.4)
    right = make_targets(GuidanceConfig(), command_id="b", first_index=0,
                         first_foot="right", direction="right", step_length_m=0.4)
    for a, b in zip(left, right, strict=True):
        assert a.x_m == pytest.approx(b.x_m)
        assert a.y_m == pytest.approx(-b.y_m)
        assert a.yaw_deg == pytest.approx(-b.yaw_deg)


@pytest.mark.parametrize("values", [
    {"cadence_bpm": float("nan")}, {"refractory_seconds": -1},
    {"target_count": 0}, {"stable_steps": 2.5}, {"first_foot": "unknown"},
    {"step_length_m": 1.0}, {"enable_fading": "false"}, {"volume": 1.5},
])
def test_invalid_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        GuidanceConfig(**values)
