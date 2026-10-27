from dataclasses import replace

import pandas as pd
import pytest

from fogguard.cueing.contracts import GuidanceInput, StepResponse
from fogguard.cueing.guidance import GuidanceController
from fogguard.cueing.march import MarchConfig, MarchGuidanceController
from fogguard.cueing.replay import replay_guidance
from fogguard.cueing.scenarios import build_guidance_cases


def context(time, **changes):
    return replace(GuidanceInput(time, ("s", "session", "task", "case"), "1", "confirmed",
                                 True, "straight", 0.9, True), **changes)


def step(time, foot="left", forward=0.0, status="success", reliability=1.0):
    return StepResponse(time, foot, forward, 0.1, status, reliability)


def marching():
    controller = MarchGuidanceController(responses_configured=True)
    controller.step(context(2))
    output = controller.step(context(2.5), (step(2.5, "none", status="no_response"),))
    return controller, output


def test_stationary_targets_and_four_steps_return_to_forward_without_false_recovery():
    controller, start = marching()
    assert start.control_state == "MARCH_FALLBACK"
    assert start.audio_mode == "march_alternating"
    assert all(target.x_m == 0 for target in start.targets)
    assert start.stable_steps == 0
    for index, foot in enumerate(["left", "right", "left", "right"]):
        now = 3 + index * 0.5
        output = controller.step(context(now), (step(now, foot),))
        assert output.command_id == start.command_id
        assert output.stable_steps == 0
    assert output.control_state == "GUIDING"
    assert output.reason == "march_completed_return_forward"
    assert output.targets[0].x_m == pytest.approx(0.4)
    assert output.visual_opacity == 1


@pytest.mark.parametrize("change", [{"turn": "unknown"}, {"input_fresh": False},
                                   {"turn_confidence": 0.1}])
def test_safety_cancels_marching_immediately(change):
    controller, _ = marching()
    output = controller.step(context(3, **change))
    assert output.action == "stop" and output.control_state == "SAFE_NONE"
    assert not output.visual_enabled and not output.audio_enabled and not output.targets


def test_repeated_foot_and_forward_displacement_do_not_complete_marching():
    controller, _ = marching()
    for now in (3, 3.5, 4, 4.5):
        output = controller.step(context(now), (step(now, "right", forward=0.4),))
        assert output.control_state == "MARCH_FALLBACK"
        assert output.stable_steps == 0


def test_missing_response_returns_fixed_targets_and_does_not_fake_success():
    controller, _ = marching()
    output = controller.step(context(3), (step(3, "none", status="missing_input"),))
    assert output.control_state == "GUIDING"
    assert output.response_source == "unavailable_fixed_fallback"
    assert output.targets[0].x_m > 0
    assert output.stable_steps == 0


def test_hold_does_not_restart_audio_or_move_stationary_targets():
    controller, start = marching()
    held = controller.step(context(3))
    assert held.action == "hold"
    assert held.targets == start.targets
    assert held.mode_started_at_s == start.mode_started_at_s


def test_march_timeout_returns_forward_then_episode_failure_locks_event():
    controller, _ = marching()
    outputs = [controller.step(context(3 + index * 0.5)) for index in range(32)]
    assert sum(row.reason == "march_timeout_fixed_guidance" for row in outputs) == 1
    assert any(row.reason == "episode_timeout" and row.action == "stop" for row in outputs)
    assert not any(row.action == "start" for row in outputs)


def test_future_and_duplicate_responses_are_rejected_without_state_change():
    controller, _ = marching()
    with pytest.raises(ValueError, match="Future"):
        controller.step(context(3), (step(4),))
    controller.step(context(3), (step(3),))
    with pytest.raises(ValueError, match="once"):
        controller.step(context(3.5), (step(3),))


@pytest.mark.parametrize("settings", [{"march_required_steps": 1},
                                      {"march_required_steps": 2.5},
                                      {"march_maximum_forward_m": 0.3}])
def test_invalid_march_parameters_fail(settings):
    with pytest.raises(ValueError):
        MarchConfig(**settings)


def test_disabled_extension_preserves_every_original_scenario_output():
    config = MarchConfig(enable_march_fallback=False)
    for events, responses in build_guidance_cases().values():
        original = replay_guidance(events, config=config, responses=responses,
                                   controller_type=GuidanceController)
        extended = replay_guidance(events, config=config, responses=responses,
                                   controller_type=MarchGuidanceController)
        pd.testing.assert_frame_equal(original, extended)


def test_truth_annotations_cannot_change_march_commands():
    events, responses = build_guidance_cases()["persistent_no_response"]
    original = replay_guidance(events, config=MarchConfig(), responses=responses,
                               controller_type=MarchGuidanceController)
    events = events.copy()
    events["truth_fog_label"] = 1 - events["truth_fog_label"]
    changed = replay_guidance(events, config=MarchConfig(), responses=responses,
                              controller_type=MarchGuidanceController)
    columns = [name for name in original if name.startswith("guidance_")]
    pd.testing.assert_frame_equal(original[columns], changed[columns])
